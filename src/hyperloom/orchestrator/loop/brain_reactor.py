# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A reactor turn's model call. The orchestration turn is the brain's: it is lent the orchestration backend and says
which intents the router handles. Any other role's backend is called directly; the Critic's asks the brain itself."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

from hyperloom.brain_api import (
    BrainContractError,
    BrainTurn,
    CallSite,
    LlmCallError,
    LlmReply,
    StateSnapshot,
    TickBatch,
    ToolCall,
    TurnFailed,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.trace.context_events import PromptSnapshotTracker, record_prompt_snapshot
from hyperloom.inference_optimizer.trace.trajectory_trace import EVENT_LLM_CALL, llm_call_summary, trajectory_span
from hyperloom.orchestrator.brain_link import resolve_brain

from ..roles.base import Backend, BackendTurnResult

ORCHESTRATION_ROLE = "orchestration"
# Snapshot key the brain reads the composed orchestration turn under.
_TURN_KEY = "orchestration_turn"


@dataclass(frozen=True)
class ReactorCall:
    """A reactor turn's model result, holding the intents to route, and the id of its ``llm.call`` span."""

    result: BackendTurnResult
    span_id: str | None


@dataclass(frozen=True)
class ReactorTurn:
    """What one reactor turn asks its model: the composed prompts, the turn's request, the tools, and its call id."""

    agent_name: str
    backend: Backend
    prompt: str
    system_prompt: str | None
    request: str
    tools: list[str]
    call_id: str


@dataclass(frozen=True)
class _ModelAsk:
    """What one backend turn is asked; ``max_turns=0`` leaves the turn count to the backend's default."""

    prompt: str
    system_prompt: str | None
    tools: list[str] | None
    max_turns: int = 0


async def _model_call(snapshots: PromptSnapshotTracker, turn: ReactorTurn, ask: _ModelAsk) -> ReactorCall:
    """One backend turn inside its ``llm.call`` span, its prompt snapshot recorded first."""
    with trajectory_span(
        EVENT_LLM_CALL,
        call_id=turn.call_id,
        attributes={"name": turn.agent_name, "model": getattr(turn.backend, "model", None)},
    ) as call_span:
        record_prompt_snapshot(
            snapshots.observe(turn.agent_name, prompt=ask.prompt, system_prompt=ask.system_prompt, tools=ask.tools)
        )
        result: BackendTurnResult = await turn.backend.run(
            prompt=ask.prompt, system_prompt=ask.system_prompt, tools=ask.tools, max_turns=ask.max_turns
        )
        call_span.finish(**llm_call_summary(result.metadata))
    return ReactorCall(result, call_span.span_id)


class _OrchestrationModel:
    """The orchestration backend, lent to the brain for one turn; a failed call reaches the brain as LlmCallError."""

    role = ORCHESTRATION_ROLE

    def __init__(self, snapshots: PromptSnapshotTracker, turn: ReactorTurn, port: "_OrchestrationPort") -> None:
        self._snapshots = snapshots
        self._turn = turn
        self._port = port
        #: The last call's result and span, for the body to trace and route.
        self.call: ReactorCall | None = None

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: Sequence[str] | None = None,
        max_turns: int | None = None,
        **ignored: Any,
    ) -> LlmReply:
        """One orchestration backend turn; the backend has no cached prefix and sets its own output cap."""
        ask = _ModelAsk(prompt, system_prompt, None if tools is None else list(tools), max_turns or 0)
        try:
            self.call = await _model_call(self._snapshots, self._turn, ask)
        except Exception as exc:  # every backend failure is restored for the reactor's own handlers
            raise self._port.own(LlmCallError(str(exc))) from exc
        result = self.call.result
        intents = [{"intent_type": i.type.value, "payload": i.payload} for i in result.intents]
        return LlmReply(intents=intents, raw_text=result.raw_text or "")


class _OrchestrationPort:
    """What one orchestration turn lends the brain: the orchestration model, and no inline tools."""

    def __init__(self, snapshots: PromptSnapshotTracker, turn: ReactorTurn) -> None:
        self._owned: list[TurnFailed] = []
        self.model = _OrchestrationModel(snapshots, turn, self)
        self.llm = {ORCHESTRATION_ROLE: self.model}

    async def call(self, call: ToolCall) -> dict[str, Any]:
        raise TurnFailed(f"an orchestration turn cannot call {call.name!r}")

    def own(self, failure: TurnFailed) -> TurnFailed:
        """Remember ``failure`` as one this port raised for a body error, so the body's error can be restored."""
        self._owned.append(failure)
        return failure

    def body_error(self, failure: TurnFailed) -> BaseException:
        """The body's own error when the brain let one of this port's failures through, else the brain's failure."""
        if any(failure is owned for owned in self._owned) and failure.__cause__ is not None:
            return failure.__cause__
        return failure


def _routed_intent(call: ToolCall) -> Intent:
    """The intent an ``intent.route`` call hands the router."""
    if call.name != "intent.route":
        raise BrainContractError(
            f"an orchestration turn routes intents and nothing else; the brain asked for {call.name}"
        )
    try:
        return Intent(type=IntentType(call.args["intent_type"]), payload=dict(call.args["payload"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise BrainContractError(f"the brain routed a malformed intent {dict(call.args)!r}: {exc}") from exc


class ReactorModels:
    """Runs each reactor turn's model call, and keeps what the brain returned from its last orchestration turn."""

    def __init__(self, snapshots: PromptSnapshotTracker) -> None:
        self._snapshots = snapshots
        self._brain_state: bytes | None = None

    async def run(self, turn: ReactorTurn, *, tick: int, phase: str) -> ReactorCall:
        """The turn's model result and span; a failed backend call raises the backend's own error."""
        if turn.agent_name != ORCHESTRATION_ROLE:
            prompt = f"{turn.prompt}\n\n{turn.request}" if turn.request else turn.prompt
            return await _model_call(self._snapshots, turn, _ModelAsk(prompt, turn.system_prompt, turn.tools))
        port = _OrchestrationPort(self._snapshots, turn)
        brain_turn = await self._ask_brain(port, turn, tick=tick, phase=phase)
        intents = [_routed_intent(call) for call in brain_turn.tool_calls]
        called = port.model.call
        result = called.result if called is not None else BackendTurnResult()
        return ReactorCall(replace(result, intents=intents), called.span_id if called is not None else None)

    async def _ask_brain(self, port: _OrchestrationPort, turn: ReactorTurn, *, tick: int, phase: str) -> BrainTurn:
        failure: BaseException | None = None
        try:
            composed = {
                "system_prompt": turn.system_prompt,
                "prompt": turn.prompt,
                "request": turn.request,
                "tools": turn.tools,
                "max_turns": 0,
            }
            batch = TickBatch(
                call_site=CallSite.LLM_TURN,
                tick=tick,
                now_unix=time.time(),
                events=(),
                snapshot=StateSnapshot(phase=phase, data={_TURN_KEY: composed}),
                brain_state=self._brain_state,
            )
            brain_turn = await resolve_brain().on_llm_turn(batch, port)
        except TurnFailed as exc:
            failure = port.body_error(exc)
        if failure is not None:
            # Raised outside the handler, so a restored body error keeps the chain it was raised with.
            raise failure
        self._brain_state = brain_turn.brain_state
        return brain_turn


__all__ = ["ORCHESTRATION_ROLE", "ReactorCall", "ReactorModels", "ReactorTurn"]

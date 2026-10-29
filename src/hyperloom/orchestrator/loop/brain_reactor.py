# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A reactor turn's model call. The orchestration turn is the brain's: it composes the prompt from the body's view, is
lent the orchestration backend, and says which intents the router handles. Any other role's backend is called directly
with the prompt the body composed; the Critic's asks the brain itself."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

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

if TYPE_CHECKING:
    from ..state.shared_state import SharedState
    from .conversation import ConversationCollaborator, TurnPrompt

ORCHESTRATION_ROLE = "orchestration"
# Snapshot key the brain reads the orchestration turn under.
_TURN_KEY = "orchestration_turn"


@dataclass(frozen=True)
class ReactorCall:
    """A reactor turn's model result, holding the intents to route, the id of its ``llm.call`` span, and the part of
    the prompt it carried in the system prompt."""

    result: BackendTurnResult
    span_id: str | None
    context: str = ""


@dataclass(frozen=True)
class ReactorTurn:
    """What one reactor turn asks its model: its prompts or view, the turn's request, the tools, and its call id."""

    agent_name: str
    backend: Backend
    prompt: "TurnPrompt"
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
        prefix: str | None = None,
        tools: Sequence[str] | None = None,
        max_turns: int | None = None,
        **ignored: Any,
    ) -> LlmReply:
        """One orchestration backend turn; the backend sets its own output cap.

        A backend that caches its system prompt carries ``prefix`` there; any other reads it ahead of the prompt.
        """
        context = ""
        if prefix is not None and getattr(self._turn.backend, "system_prompt_cached", False):
            system_prompt, context = f"{system_prompt}\n\n{prefix}", prefix
        elif prefix is not None:
            prompt = f"{prefix}\n{prompt}"
        ask = _ModelAsk(prompt, system_prompt, None if tools is None else list(tools), max_turns or 0)
        try:
            called = await _model_call(self._snapshots, self._turn, ask)
            self.call = replace(called, context=context)
        except Exception as exc:  # every backend failure is restored for the reactor's own handlers
            raise self._port.own(LlmCallError(str(exc))) from exc
        result = self.call.result
        intents = [{"intent_type": i.type.value, "payload": i.payload} for i in result.intents]
        return LlmReply(intents=intents, raw_text=result.raw_text or "")


class _OrchestrationPort:
    """What one orchestration turn lends the brain: the orchestration model, and the inbox cursor its prompt moves."""

    def __init__(
        self, snapshots: PromptSnapshotTracker, turn: ReactorTurn, conversation: "ConversationCollaborator"
    ) -> None:
        self._owned: list[TurnFailed] = []
        self._conversation = conversation
        self.model = _OrchestrationModel(snapshots, turn, self)
        self.llm = {ORCHESTRATION_ROLE: self.model}

    async def call(self, call: ToolCall) -> dict[str, Any]:
        """Note the last inbox message the composed prompt showed; nothing else answers inline."""
        if call.name != "inbox.rendered":
            raise TurnFailed(f"an orchestration turn cannot call {call.name!r}")
        self._conversation.mark_rendered(ORCHESTRATION_ROLE, int(call.args["seq"]))
        return {"accepted": True}

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

    async def run(
        self, turn: ReactorTurn, state: "SharedState", conversation: "ConversationCollaborator"
    ) -> ReactorCall:
        """The turn's model result and span; a failed backend call raises the backend's own error."""
        prompts = turn.prompt
        if turn.agent_name != ORCHESTRATION_ROLE:
            prompt = f"{prompts.prompt}\n\n{turn.request}" if turn.request else prompts.prompt
            called = await _model_call(self._snapshots, turn, _ModelAsk(prompt, prompts.system_prompt, turn.tools))
            return replace(called, context=prompts.context)
        port = _OrchestrationPort(self._snapshots, turn, conversation)
        brain_turn = await self._ask_brain(port, turn, state)
        intents = [_routed_intent(call) for call in brain_turn.tool_calls]
        called = port.model.call
        if called is None:
            return ReactorCall(BackendTurnResult(intents=intents), None)
        return replace(called, result=replace(called.result, intents=intents))

    async def _ask_brain(self, port: _OrchestrationPort, turn: ReactorTurn, state: "SharedState") -> BrainTurn:
        failure: BaseException | None = None
        try:
            reported = {
                "system_prompt": turn.prompt.system_prompt,
                "view": turn.prompt.view,
                "request": turn.request,
                "tools": turn.tools,
                "max_turns": 0,
            }
            batch = TickBatch(
                call_site=CallSite.LLM_TURN,
                tick=int(state.tick or 0),
                now_unix=time.time(),
                events=(),
                snapshot=StateSnapshot(phase=state.phase or "", data={_TURN_KEY: reported}),
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

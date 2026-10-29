# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestration reactor turn goes through the brain; its backend's own errors and intents come back unchanged."""

from __future__ import annotations

import pytest

from hyperloom.brain_api import BrainContractError, BrainTurn, ToolCall, TurnFailed
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType, NoIntentEmitted
from hyperloom.inference_optimizer.trace.context_events import PromptSnapshotTracker
from hyperloom.orchestrator.loop import brain_reactor
from hyperloom.orchestrator.loop.brain_reactor import ReactorModels, ReactorTurn
from hyperloom.orchestrator.roles.base import BackendError, BackendTurnResult, LLMCallFailed

_INTENTS = [
    Intent(IntentType.PROPOSE_ACTION, {"action_name": "explore", "predicted_gain_pct": 2.5, "params": {"k": [1, 2]}}),
    Intent(IntentType.SEND_MESSAGE, {"topic": "note"}),
]


class _Backend:
    model = "m"

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _turn(backend, agent="orchestration", request=""):
    return ReactorTurn(agent, backend, "ctx", "sys", request, ["emit_intent"], "call-1")


async def _run(backend, **kwargs):
    return await ReactorModels(PromptSnapshotTracker()).run(_turn(backend, **kwargs), tick=4, phase="FRAMEWORK_AGENT")


async def test_the_brain_asks_the_backend_the_composed_turn_and_routes_its_intents() -> None:
    backend = _Backend(BackendTurnResult(intents=list(_INTENTS), raw_text="r", metadata={"model": "m"}))
    called = await _run(backend, request="answer this")
    assert backend.calls == [
        {"prompt": "ctx\n\nanswer this", "system_prompt": "sys", "tools": ["emit_intent"], "max_turns": 0}
    ]
    assert called.result == BackendTurnResult(intents=_INTENTS, raw_text="r", metadata={"model": "m"})
    assert called.span_id


@pytest.mark.parametrize(
    "error", [BackendError("down"), LLMCallFailed("api"), NoIntentEmitted("none"), RuntimeError("bug")]
)
async def test_a_backend_failure_reaches_the_reactor_as_it_was_raised(error) -> None:
    with pytest.raises(type(error)) as caught:
        await _run(_Backend(error))
    assert caught.value is error


async def test_another_role_calls_its_backend_without_the_brain(monkeypatch) -> None:
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: pytest.fail("the brain was asked"))
    backend = _Backend(BackendTurnResult(intents=list(_INTENTS)))
    called = await _run(backend, agent="critic", request="q")
    assert backend.calls[0]["prompt"] == "ctx\n\nq"
    assert called.result.intents == _INTENTS


class _Brain:
    def __init__(self, turn):
        self.turn = turn

    async def on_llm_turn(self, batch, port):
        if isinstance(self.turn, BaseException):
            raise self.turn
        return self.turn


async def test_a_brain_that_asks_for_anything_but_routing_breaks_the_contract(monkeypatch) -> None:
    turn = BrainTurn(tool_calls=(ToolCall("task.prune", {"family": "f", "reason": "r"}),))
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: _Brain(turn))
    with pytest.raises(BrainContractError, match="task.prune"):
        await _run(_Backend(BackendTurnResult()))


async def test_a_brain_that_routes_without_asking_the_model_routes_its_own_intents(monkeypatch) -> None:
    turn = BrainTurn(tool_calls=(ToolCall("intent.route", {"intent_type": "alert", "payload": {"severity": "low"}}),))
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: _Brain(turn))
    backend = _Backend(BackendTurnResult())
    called = await _run(backend)
    assert backend.calls == []
    assert called.result.intents == [Intent(IntentType.ALERT, {"severity": "low"})]
    assert called.span_id is None


async def test_a_brain_that_fails_the_turn_fails_it_with_its_own_reason(monkeypatch) -> None:
    failure = TurnFailed("cannot decide")
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: _Brain(failure))
    with pytest.raises(TurnFailed) as caught:
        await _run(_Backend(BackendTurnResult()))
    assert caught.value is failure

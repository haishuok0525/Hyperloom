# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestration reactor turn goes through the brain; its backend's own errors and intents come back unchanged."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.brain_api import BrainContractError, BrainTurn, ToolCall, TurnFailed
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType, NoIntentEmitted
from hyperloom.inference_optimizer.trace.context_events import PromptSnapshotTracker
from hyperloom.orchestrator.loop import brain_reactor
from hyperloom.orchestrator.loop.brain_reactor import ReactorModels, ReactorTurn
from hyperloom.orchestrator.loop.conversation import TurnPrompt
from hyperloom.orchestrator.roles.base import BackendError, BackendTurnResult, LLMCallFailed

_INTENTS = [
    Intent(IntentType.PROPOSE_ACTION, {"action_name": "explore", "predicted_gain_pct": 2.5, "params": {"k": [1, 2]}}),
    Intent(IntentType.SEND_MESSAGE, {"topic": "note"}),
]
_STATE = SimpleNamespace(tick=4, phase="FRAMEWORK_AGENT")


class _Backend:
    model = "m"

    def __init__(self, outcome, *, cached=False):
        self.outcome = outcome
        self.system_prompt_cached = cached
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Conversation:
    def __init__(self):
        self.rendered: dict[str, int] = {}

    def mark_rendered(self, agent_name, seq):
        self.rendered[agent_name] = seq


class _Brain:
    """Composes ``turn`` / ``context``, reports inbox seq 9, asks the model once, routes what it emitted."""

    def __init__(self, failure=None):
        self.failure = failure
        self.batches: list = []

    async def on_llm_turn(self, batch, port):
        self.batches.append(batch)
        if self.failure is not None:
            raise self.failure
        await port.call(ToolCall("inbox.rendered", {"seq": 9}))
        reply = await port.llm["orchestration"].run("turn", system_prompt="sys", prefix="context", tools=["t"])
        return BrainTurn(tool_calls=tuple(ToolCall("intent.route", i) for i in reply.intents))


def _turn(backend, agent="orchestration", request=""):
    prompts = TurnPrompt("sys", "ctx", "cached-ctx", view={"inbox": []} if agent == "orchestration" else None)
    return ReactorTurn(agent, backend, prompts, request, ["emit_intent"], "call-1")


async def _run(backend, conversation=None, **kwargs):
    models = ReactorModels(PromptSnapshotTracker())
    return await models.run(_turn(backend, **kwargs), _STATE, conversation or _Conversation())


@pytest.fixture
def brain(monkeypatch):
    stub = _Brain()
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: stub)
    return stub


async def test_the_brain_is_given_the_view_and_its_intents_are_routed(brain) -> None:
    backend = _Backend(BackendTurnResult(intents=list(_INTENTS), raw_text="r", metadata={"model": "m"}))
    conversation = _Conversation()
    called = await _run(backend, conversation, request="answer this")
    reported = brain.batches[0].snapshot.data["orchestration_turn"]
    assert reported == {
        "system_prompt": "sys",
        "view": {"inbox": []},
        "request": "answer this",
        "tools": ["emit_intent"],
        "max_turns": 0,
    }
    assert (brain.batches[0].tick, brain.batches[0].snapshot.phase) == (4, "FRAMEWORK_AGENT")
    assert called.result == BackendTurnResult(intents=_INTENTS, raw_text="r", metadata={"model": "m"})
    assert called.span_id
    assert conversation.rendered == {"orchestration": 9}


@pytest.mark.parametrize(
    ("cached", "system_prompt", "prompt", "context"),
    [(False, "sys", "context\nturn", ""), (True, "sys\n\ncontext", "turn", "context")],
)
async def test_the_cacheable_prefix_goes_where_the_backend_caches(
    brain, cached, system_prompt, prompt, context
) -> None:
    backend = _Backend(BackendTurnResult(), cached=cached)
    called = await _run(backend)
    assert backend.calls == [{"prompt": prompt, "system_prompt": system_prompt, "tools": ["t"], "max_turns": 0}]
    assert called.context == context


@pytest.mark.parametrize(
    "error", [BackendError("down"), LLMCallFailed("api"), NoIntentEmitted("none"), RuntimeError("bug")]
)
async def test_a_backend_failure_reaches_the_reactor_as_it_was_raised(brain, error) -> None:
    with pytest.raises(type(error)) as caught:
        await _run(_Backend(error))
    assert caught.value is error


async def test_another_role_calls_its_backend_with_the_prompt_the_body_composed(monkeypatch) -> None:
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: pytest.fail("the brain was asked"))
    backend = _Backend(BackendTurnResult(intents=list(_INTENTS)))
    called = await _run(backend, agent="critic", request="q")
    assert backend.calls == [{"prompt": "ctx\n\nq", "system_prompt": "sys", "tools": ["emit_intent"], "max_turns": 0}]
    assert called.result.intents == _INTENTS
    assert called.context == "cached-ctx"


class _FixedBrain:
    def __init__(self, turn):
        self.turn = turn

    async def on_llm_turn(self, batch, port):
        if isinstance(self.turn, BaseException):
            raise self.turn
        return self.turn


async def test_a_brain_that_asks_for_anything_but_routing_breaks_the_contract(monkeypatch) -> None:
    turn = BrainTurn(tool_calls=(ToolCall("task.prune", {"family": "f", "reason": "r"}),))
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: _FixedBrain(turn))
    with pytest.raises(BrainContractError, match="task.prune"):
        await _run(_Backend(BackendTurnResult()))


async def test_a_brain_that_routes_without_asking_the_model_routes_its_own_intents(monkeypatch) -> None:
    turn = BrainTurn(tool_calls=(ToolCall("intent.route", {"intent_type": "alert", "payload": {"severity": "low"}}),))
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: _FixedBrain(turn))
    backend = _Backend(BackendTurnResult())
    called = await _run(backend)
    assert backend.calls == []
    assert called.result.intents == [Intent(IntentType.ALERT, {"severity": "low"})]
    assert called.span_id is None


async def test_a_brain_that_fails_the_turn_fails_it_with_its_own_reason(monkeypatch) -> None:
    failure = TurnFailed("cannot decide")
    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: _FixedBrain(failure))
    with pytest.raises(TurnFailed) as caught:
        await _run(_Backend(BackendTurnResult()))
    assert caught.value is failure


async def test_an_orchestration_turn_answers_no_query_inline(monkeypatch) -> None:
    class _Asking:
        async def on_llm_turn(self, batch, port):
            await port.call(ToolCall("query.get_gaps", {}))

    monkeypatch.setattr(brain_reactor, "resolve_brain", lambda: _Asking())
    with pytest.raises(TurnFailed, match="query.get_gaps"):
        await _run(_Backend(BackendTurnResult()))

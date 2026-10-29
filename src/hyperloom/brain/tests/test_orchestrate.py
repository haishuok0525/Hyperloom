# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestration turn of Brain v0, against a fake body: what it asks the model, and what it routes."""

from __future__ import annotations

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain.orchestrate import ORCHESTRATION_KEY
from hyperloom.brain_api import BrainTurn, CallSite, LlmCallError, LlmReply, StateSnapshot, TickBatch, TurnFailed

_INTENTS = (
    {"intent_type": "delegate", "payload": {"action_name": "explore"}},
    {"intent_type": "send_message", "payload": {"topic": "note", "body": "x"}},
)


class _Model:
    role = "orchestration"

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    async def run(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _Port:
    def __init__(self, model):
        self.llm = {"orchestration": model}

    async def call(self, call):
        raise AssertionError(f"unexpected tool call {call.name}")


def _batch(**turn):
    composed = {"system_prompt": "sys", "prompt": "ctx", "request": "", "tools": ["t1"], "max_turns": 0, **turn}
    return TickBatch(
        call_site=CallSite.LLM_TURN,
        tick=3,
        now_unix=0.0,
        events=(),
        snapshot=StateSnapshot("FRAMEWORK_AGENT", {ORCHESTRATION_KEY: composed}),
        brain_state=b"kept",
    )


async def test_the_model_is_asked_the_composed_turn_and_every_intent_is_routed_in_order() -> None:
    model = _Model(LlmReply(intents=_INTENTS, raw_text="r"))
    turn = await create_brain({}).on_llm_turn(_batch(), _Port(model))
    assert model.calls == [{"prompt": "ctx", "system_prompt": "sys", "tools": ["t1"], "max_turns": 0}]
    assert [(c.name, dict(c.args)) for c in turn.tool_calls] == [("intent.route", i) for i in _INTENTS]
    assert turn.brain_state == b"kept"


async def test_a_request_follows_the_composed_prompt() -> None:
    model = _Model(LlmReply())
    await create_brain({}).on_llm_turn(_batch(request="answer this"), _Port(model))
    assert model.calls[0]["prompt"] == "ctx\n\nanswer this"


async def test_a_turn_without_intents_routes_nothing() -> None:
    turn = await create_brain({}).on_llm_turn(_batch(), _Port(_Model(LlmReply(raw_text="nothing"))))
    assert turn == BrainTurn(brain_state=b"kept")


async def test_a_failed_model_call_reaches_the_body_as_it_was_raised() -> None:
    failure = LlmCallError("backend down")
    with pytest.raises(LlmCallError) as caught:
        await create_brain({}).on_llm_turn(_batch(), _Port(_Model(failure)))
    assert caught.value is failure


async def test_a_turn_the_body_did_not_compose_fails() -> None:
    batch = TickBatch(call_site=CallSite.LLM_TURN, tick=0, now_unix=0.0, events=(), snapshot=StateSnapshot(""))
    with pytest.raises(TurnFailed, match=ORCHESTRATION_KEY):
        await create_brain({}).on_llm_turn(batch, _Port(_Model(LlmReply())))

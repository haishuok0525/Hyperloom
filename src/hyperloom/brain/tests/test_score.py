# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Advisory proposal scoring at CallSite.SCORE."""

from __future__ import annotations

import json

from hyperloom.brain import create_brain
from hyperloom.brain.score import PROPOSAL_SCORE_KEY
from hyperloom.brain_api import CallSite, LlmReply, StateSnapshot, TickBatch


class _Model:
    def __init__(self, reply: LlmReply | Exception) -> None:
        self.role = "m"
        self._reply = reply
        self.prompts: list[str] = []

    async def run(self, prompt: str, **_kwargs):
        self.prompts.append(prompt)
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply


class _Port:
    def __init__(self, *models: _Model) -> None:
        self.llm = {m.role: m for m in models}

    async def call(self, call):
        raise AssertionError(f"unexpected {call.name}")


def _batch(request: dict) -> TickBatch:
    return TickBatch(
        CallSite.SCORE,
        1,
        0.0,
        (),
        StateSnapshot("FRAMEWORK_AGENT", {PROPOSAL_SCORE_KEY: request}),
        b"kept",
    )


async def test_score_records_per_model_results_and_errors() -> None:
    ok = _Model(
        LlmReply(raw_text=json.dumps({"scores": {"proposal_0": {"score": 7, "reason": "likely win"}}}))
    )
    ok.role = "good"
    bad = _Model(RuntimeError("gateway 400"))
    bad.role = "bad"
    turn = await create_brain({}).on_score(
        _batch(
            {
                "gap": {"domain": "d"},
                "proposals": [{"name": "p"}],
                "models": ["good", "bad"],
                "max_completion_tokens": 4096,
            }
        ),
        _Port(ok, bad),
    )
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "score.record"
    args = turn.tool_calls[0].args
    assert args["models"]["good"]["p"]["score"] == 7.0
    assert "bad" in args["errors"]
    assert "THROUGHPUT" in ok.prompts[0]


async def test_score_duplicate_names_are_an_input_error() -> None:
    turn = await create_brain({}).on_score(
        _batch(
            {
                "gap": {},
                "proposals": [{"name": "dup"}, {"name": "dup"}],
                "models": ["m"],
            }
        ),
        _Port(),
    )
    assert turn.tool_calls[0].args["errors"]["input"]
    assert turn.tool_calls[0].args["models"] == {}

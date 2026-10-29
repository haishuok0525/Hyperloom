# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestration turn of Brain v0, against a fake body: the prompt it composes, what it asks, what it routes."""

from __future__ import annotations

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain.orchestrate import ORCHESTRATION_KEY, compose_prompt
from hyperloom.brain.orchestrate import prompt as orchestrate_prompt
from hyperloom.brain_api import BrainTurn, CallSite, LlmCallError, LlmReply, StateSnapshot, TickBatch, TurnFailed

_INTENTS = (
    {"intent_type": "delegate", "payload": {"action_name": "explore"}},
    {"intent_type": "send_message", "payload": {"topic": "note", "body": "x"}},
)


def _view(**overrides):
    view = {
        "session_dir": "/s",
        "phase": "FRAMEWORK_AGENT",
        "mission": "mission",
        "time_budget": None,
        "tick": "tick 3",
        "shared_state": "ledger",
        "target_gap_pct": "1.50",
        "constraints": ["=== Constraints ===", "c"],
        "resource_pools": "pools",
        "advisories": {
            "warm_start": "warm",
            "gaps": "",
            "target_gap": "",
            "proposal_scores": "",
            "priors_match": "",
            "intervention_mix": "",
        },
        "specialist_rounds": [],
        "bottleneck": None,
        "acceptance": {"keep_threshold": 1.0, "macro_cycle": 0, "entries": []},
        "discarded_escalate_hint": {"hint": "", "ts": ""},
        "denial_summary": "",
        "proposal_queue": None,
        "plateau": "",
        "trajectory_digest": "",
        "inbox": [],
    }
    view.update(overrides)
    return view


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
    def __init__(self, model=None):
        self.llm = {"orchestration": model}
        self.rendered: list[int] = []

    async def call(self, call):
        assert call.name == "inbox.rendered", call.name
        self.rendered.append(call.args["seq"])
        return {"accepted": True}


def _batch(view=None, **turn):
    reported = {"system_prompt": "sys", "view": view or _view(), "request": "", "tools": ["t1"], "max_turns": 0}
    reported.update(turn)
    return TickBatch(
        call_site=CallSite.LLM_TURN,
        tick=3,
        now_unix=0.0,
        events=(),
        snapshot=StateSnapshot("FRAMEWORK_AGENT", {ORCHESTRATION_KEY: reported}),
        brain_state=b"kept",
    )


async def test_the_session_context_and_the_moving_turn_are_split_in_reading_order() -> None:
    prompt = await compose_prompt(_view(denial_summary="denied", plateau="flat", trajectory_digest="digest"), _Port())
    assert prompt.context == "\n".join(
        ["SESSION_DIR=/s", "=== Constraints ===", "c", "=== Resource pools ===", "pools"]
        + ["=== Warm start (Recipe KB T0) ===", "warm"]
    )
    assert prompt.turn == "\n".join(
        [
            "=== Phase ===",
            "FRAMEWORK_AGENT",
            "=== Mission progress ===",
            "mission",
            "=== Tick ===",
            "tick 3",
            "=== Shared session state ===",
            "ledger",
            "target_gap_pct=1.50",
            "denied",
            "=== Plateau advisory ===",
            "flat",
            "=== Trajectory review (advisory) ===",
            "digest",
            "=== Inbox for orchestration ===",
            "(no new messages)",
        ]
    )


async def test_the_proposal_queue_leads_with_the_predictor_and_ends_with_the_kb_evidence() -> None:
    queue = {"predictor": "P", "untested": "U", "kb_evidence": "KB"}
    prompt = await compose_prompt(_view(proposal_queue=queue), _Port())
    assert "=== Untested proposals (current cycle) ===\nP\n\nU\nKB\n=== Inbox" in prompt.turn


@pytest.mark.parametrize(
    ("remaining", "closing", "warned"), [(4.0, False, True), (4.0, True, False), (6.0, False, False)]
)
async def test_the_deadline_warning_appears_only_near_the_end_outside_the_closing_phase(
    remaining, closing, warned
) -> None:
    budget = {"elapsed_min": 55.0, "remaining_min": remaining, "budget_min": 60, "closing_phase": closing}
    prompt = await compose_prompt(_view(time_budget=budget), _Port())
    assert f"elapsed=55.0min  remaining={remaining:.1f}min  budget=60min  closing_phase={closing}" in prompt.turn
    assert ("< 5 min remaining" in prompt.turn) is warned


async def test_the_inbox_shows_one_budgeted_page_and_reports_the_last_message_shown(monkeypatch) -> None:
    monkeypatch.setattr(orchestrate_prompt, "INBOX_PROMPT_CHARS", 30)
    port = _Port()
    inbox = [{"seq": 4, "line": "a" * 10}, {"seq": 7, "line": "b" * 10}, {"seq": 9, "line": "c" * 10}]
    prompt = await compose_prompt(_view(inbox=inbox), port)
    assert prompt.turn.endswith(
        f"=== Inbox for orchestration (newest last) ===\n  {'a' * 10}\n  {'b' * 10}\n"
        "  (+1 newer messages after seq=7 held for the next turn)"
    )
    assert port.rendered == [7]


async def test_the_model_is_asked_the_turn_with_the_context_as_its_cacheable_prefix() -> None:
    model = _Model(LlmReply(intents=_INTENTS, raw_text="r"))
    port = _Port(model)
    turn = await create_brain({}).on_llm_turn(_batch(request="answer this"), port)
    expected = await compose_prompt(_view(), _Port())
    assert model.calls == [
        {
            "prompt": f"{expected.turn}\n\nanswer this",
            "system_prompt": "sys",
            "prefix": expected.context,
            "tools": ["t1"],
            "max_turns": 0,
        }
    ]
    assert [(c.name, dict(c.args)) for c in turn.tool_calls] == [("intent.route", i) for i in _INTENTS]
    assert turn.brain_state == b"kept"


async def test_a_turn_without_intents_routes_nothing() -> None:
    turn = await create_brain({}).on_llm_turn(_batch(), _Port(_Model(LlmReply(raw_text="nothing"))))
    assert turn == BrainTurn(brain_state=b"kept")


async def test_a_failed_model_call_reaches_the_body_as_it_was_raised() -> None:
    failure = LlmCallError("backend down")
    with pytest.raises(LlmCallError) as caught:
        await create_brain({}).on_llm_turn(_batch(), _Port(_Model(failure)))
    assert caught.value is failure


async def test_a_turn_the_body_did_not_report_fails() -> None:
    batch = TickBatch(call_site=CallSite.LLM_TURN, tick=0, now_unix=0.0, events=(), snapshot=StateSnapshot(""))
    with pytest.raises(TurnFailed, match=ORCHESTRATION_KEY):
        await create_brain({}).on_llm_turn(batch, _Port(_Model(LlmReply())))

# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Critic turn of Brain v0, against a fake body: what it reads, what it asks the model, what it records."""

from __future__ import annotations

import json

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain.critic.constraints import LEVER_ORIENTATION, PHASE_ORIENTATION
from hyperloom.brain.critic.review import REVIEW_OUTPUT_INSTRUCTIONS
from hyperloom.brain_api import CallSite, LlmCallError, LlmReply, StateSnapshot, TickBatch, TurnFailed

_RULES = {
    "scope_domains": "domains",
    "cross_domain_rules": [
        {"rule_id": "x", "description": "d", "failure_verdict": "advise", "failure_reason_code": "c"}
    ],
    "quantitative_claim_rule": {"rule_id": "no_self_reported_gain", "forbidden_proposal_fields": ["gain"]},
    "advisory_rule_actions": ["explore", "specialist"],
}
_RULING = '```json\n{"review_verdicts": [{"target_proposal_msg_id": "p1", "verdict": "approve"}]}\n```'


class _Model:
    role = "critic"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def run(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _Port:
    def __init__(self, model, bundle, levers=("",)):
        self.llm = {"critic": model}
        self.bundle = bundle
        self.levers = list(levers)
        self.reads = 0

    async def call(self, call):
        assert call.name == "kb.critic.read"
        self.reads += 1
        return {"judge_bundle": json.loads(json.dumps(self.bundle)), "proposal_levers": self.levers}


def _batch(phase="FRAMEWORK_AGENT", **critic):
    inputs = {
        "system_prompt": "critic system",
        "skill_preamble": "PREAMBLE",
        "max_completion_tokens": 100,
        "action_verdict_policy": {},
        "rules": _RULES,
    }
    inputs.update(critic)
    return TickBatch(
        call_site=CallSite.LLM_TURN,
        tick=3,
        now_unix=0.0,
        events=(),
        snapshot=StateSnapshot(phase=phase, data={"critic": inputs}),
        brain_state=b"s",
    )


def _bundle(*proposals):
    return {"session_id": "s1", "proposals": list(proposals), "review_constraints": {}}


async def _turn(port, **critic):
    return await create_brain({}).on_llm_turn(_batch(**critic), port)


async def test_no_proposals_rules_on_nothing_without_asking_the_model() -> None:
    model = _Model()
    turn = await _turn(_Port(model, _bundle()))
    (call,) = turn.tool_calls
    assert call.name == "verdict.record"
    assert call.args["review"] == {"review_verdicts": []}
    assert (call.args["raw_text"], call.args["finish"]) == ("(skipped — no proposals)", "skipped")
    assert model.calls == []
    assert turn.brain_state == b"s"


async def test_the_model_is_asked_with_the_constrained_bundle_and_the_cached_preamble() -> None:
    model = _Model(LlmReply(raw_text=_RULING, finish_reason="stop"))
    proposal = {"msg_id": "p1", "action_name": "explore", "scope": "domains"}
    turn = await _turn(_Port(model, _bundle(proposal), levers=("config",)), action_verdict_policy={"explore": "x"})
    (asked,) = model.calls
    assert (asked["prefix"], asked["system_prompt"], asked["max_tokens"]) == ("PREAMBLE", "critic system", 100)
    bundle = turn.tool_calls[0].args["judge_bundle"]
    rc = bundle["review_constraints"]
    assert list(rc) == [
        "action_verdict_policy",
        "phase",
        "phase_orientation",
        "lever_kind",
        "lever_orientation",
        "cross_domain",
        "cross_domain_rules",
        "quantitative_claim_rule",
    ]
    assert rc["phase_orientation"] == PHASE_ORIENTATION["FRAMEWORK_AGENT"]
    assert rc["lever_orientation"] == LEVER_ORIENTATION["config"]
    view = {k: bundle.get(k) for k in ("session_id", "decision_id", "phase", "merged_context", "missing_context")}
    view |= {k: bundle.get(k) for k in ("required_context", "proposals", "kb_priors_by_proposal")}
    view |= {k: bundle.get(k) for k in ("kb_read_skipped_reason", "review_constraints", "notes")}
    assert asked["prompt"] == (
        f"==== JUDGE BUNDLE ====\n{json.dumps(view, ensure_ascii=False, separators=(',', ':'))}"
        f"\n==== END JUDGE BUNDLE ====\n\n{REVIEW_OUTPUT_INSTRUCTIONS}"
    )
    assert turn.tool_calls[0].args["review"]["review_verdicts"][0]["verdict"] == "approve"


async def test_the_kb_keeps_a_lesson_only_for_a_persisted_verdict_worth_keeping() -> None:
    verdicts = [
        {"target_proposal_msg_id": "p1", "verdict": "reject", "persist_to_kb": True},
        {"target_proposal_msg_id": "p2", "verdict": "advise", "persist_to_kb": True},
        {"target_proposal_msg_id": "p3", "verdict": "approve"},
        {"target_proposal_msg_id": "p4", "verdict": "redirect", "persist_to_kb": True},
        "not a verdict",
    ]
    reply = LlmReply(raw_text=f"```json\n{json.dumps({'review_verdicts': verdicts})}\n```")
    turn = await _turn(_Port(_Model(reply), _bundle({"msg_id": "p1", "action_name": "baseline"})))
    assert turn.tool_calls[0].args["kb_writes"] == ["p1", "p4"]
    skipped = await _turn(_Port(_Model(), _bundle()))
    assert skipped.tool_calls[0].args["kb_writes"] == []


async def test_a_proposal_no_rule_governs_gets_no_rule_constraints() -> None:
    model = _Model(LlmReply(raw_text=_RULING))
    turn = await _turn(_Port(model, _bundle({"msg_id": "p1", "action_name": "baseline"})), phase="")
    assert turn.tool_calls[0].args["judge_bundle"]["review_constraints"] == {}


async def test_a_truncated_reply_is_retried_once_with_twice_the_cap() -> None:
    model = _Model(LlmReply(raw_text='{"review_verdicts": [', finish_reason="length"), LlmReply(raw_text=_RULING))
    turn = await _turn(_Port(model, _bundle({"msg_id": "p1"})))
    assert [c["max_tokens"] for c in model.calls] == [100, 200]
    assert turn.tool_calls[0].args["raw_text"] == _RULING


async def test_a_rejected_retry_fails_the_turn_naming_both_caps() -> None:
    model = _Model(LlmReply(raw_text="", finish_reason="max_tokens"), LlmCallError("limit is 150"))
    with pytest.raises(TurnFailed, match="truncated at 100 tokens and the retry at 200 was rejected: limit is 150"):
        await _turn(_Port(model, _bundle({"msg_id": "p1"})))


async def test_a_failed_first_call_reaches_the_body_unchanged() -> None:
    failure = LlmCallError("provider down")
    with pytest.raises(LlmCallError) as caught:
        await _turn(_Port(_Model(failure), _bundle({"msg_id": "p1"})))
    assert caught.value is failure


async def test_a_reply_without_a_ruling_fails_the_turn() -> None:
    model = _Model(LlmReply(raw_text="just prose", finish_reason="stop"))
    with pytest.raises(TurnFailed, match="no parseable review_verdicts"):
        await _turn(_Port(model, _bundle({"msg_id": "p1"})))

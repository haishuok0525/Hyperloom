# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Critic rules on a constrained bundle and its ruling reads back as one verdict per candidate."""

from __future__ import annotations

import json

import pytest

from hyperloom.brain.critic import judge_bundle, review_candidates
from hyperloom.brain.critic.constraints import LEVER_ORIENTATION, PHASE_ORIENTATION
from hyperloom.brain_api import (
    Candidate,
    ConfigDelta,
    Decision,
    LlmReply,
    PatchFile,
    PortError,
    ReviewAsk,
    SearchState,
    Workload,
)

_STATE = SearchState(Workload("m", "vllm"), baseline_tput=100.0, current_tput=110.0, keep_threshold_pct=1.0)
_PATCH = Candidate("p", patches=(PatchFile("vllm/x.py", "-a\n+b\n" * 4000),), reason="bridge")
_ASK = ReviewAsk((_PATCH, Candidate("c", ConfigDelta("--c")), Candidate("d", ConfigDelta("--d"))), ("no fp8",))


class _Llm:
    def __init__(self, *replies: LlmReply) -> None:
        self.replies, self.requests = list(replies), []

    async def llm(self, request):
        self.requests.append(request)
        return self.replies.pop(0)

    async def agent(self, request):
        raise AssertionError("the Critic runs no agent")


def _ruling(*verdicts: tuple[str, str]) -> str:
    rows = [
        {"target_proposal_msg_id": n, "verdict": v, "reasoning": f"why {n}", "predicted_gain_pct": 2}
        for n, v in verdicts
    ]
    return "```json\n" + json.dumps({"review_verdicts": rows}) + "\n```"


def test_the_bundle_holds_each_candidate_with_its_lever_and_the_asks_rules() -> None:
    bundle = judge_bundle(_STATE, _ASK)
    rc = bundle["review_constraints"]
    assert rc["rules"] == ["no fp8"]
    assert rc["phase_orientation"] == PHASE_ORIENTATION["FRAMEWORK_AGENT"]
    assert rc["lever_orientation"] == LEVER_ORIENTATION["source_patch"]
    first, second, _ = bundle["proposals"]
    assert (first["msg_id"], first["action_name"], first["lever_kind"]) == ("p", "integrate_patch", "source_patch")
    assert first["params"]["patch_files"] == ["vllm/x.py"] and "more chars cut" in first["params"]["diff"]
    assert (second["lever_kind"], second["params"]["extra_args"]) == ("config", "--c")


async def test_the_ruling_maps_to_decisions_and_an_uncovered_candidate_is_deferred() -> None:
    ports = _Llm(LlmReply(_ruling(("p", "advise"), ("c", "reject"))))
    review = await review_candidates(_STATE, _ASK, ports)
    got = {v.candidate: (v.decision, v.predicted_gain_pct) for v in review.verdicts}
    assert got == {"p": (Decision.APPROVE, 2.0), "c": (Decision.REJECT, 2.0), "d": (Decision.DEFER, None)}
    (request,) = ports.requests
    assert "==== JUDGE BUNDLE ====" in request.prompt and "You are the Critic" in request.system_prompt


async def test_redirect_and_needs_review_hold_a_candidate_back() -> None:
    ports = _Llm(LlmReply(_ruling(("p", "redirect"), ("c", "needs_review"), ("d", "approve"))))
    review = await review_candidates(_STATE, _ASK, ports)
    assert [v.decision for v in review.verdicts] == [Decision.DEFER, Decision.DEFER, Decision.APPROVE]


async def test_a_truncated_reply_is_retried_once_with_twice_the_cap() -> None:
    ports = _Llm(LlmReply('{"review_verdicts": [', finish_reason="max_tokens"), LlmReply(_ruling(("p", "approve"))))
    review = await review_candidates(_STATE, _ASK, ports)
    assert [r.max_tokens for r in ports.requests] == [8192, 16384]
    assert review.verdicts[0].decision is Decision.APPROVE


async def test_a_reply_without_a_ruling_fails_retryably() -> None:
    with pytest.raises(PortError, match="review_verdicts") as err:
        await review_candidates(_STATE, _ASK, _Llm(LlmReply("I think they are fine.")))
    assert err.value.retryable

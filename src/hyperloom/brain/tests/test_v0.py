# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0 honours the contract: a proposal is the specialist's candidates the Critic approves."""

from __future__ import annotations

import json

from hyperloom.brain import create_brain
from hyperloom.brain_api import (
    AgentReply,
    Candidate,
    ConfigDelta,
    Decision,
    Gap,
    LlmReply,
    PatchFile,
    PortError,
    ProposeAsk,
    ReviewAsk,
    SearchState,
    Workload,
    check_proposal,
    check_review,
    points_of,
    validate_brain,
)

_STATE = SearchState(
    Workload("m", "vllm"),
    baseline_tput=100.0,
    current_tput=100.0,
    keep_threshold_pct=1.0,
    gaps=(Gap("attention.decode", domain="attention", severity="high", layer="kernel"),),
)
_ASK = ProposeAsk(task="Optimise m serving with vllm.", max_candidates=6, patches=True)
_DONE = {
    "proposal_set": [{"name": "a", "extra_args": "--a"}, {"name": "b", "extra_envs": {"B": "1"}}],
    "patch_reason": "faster decode attention",
    "summary": "two flags and a patch",
    "confidence": 0.7,
    "new_findings": [{"what": "decode attention takes the naive path"}],
    "residual_questions": ["is fp8 kv allowed?"],
}


class _Ports:
    def __init__(self, ruling: list[tuple[str, str]] | None = None, *, critic_fails: bool = False) -> None:
        self.ruling, self.critic_fails = ruling or [], critic_fails
        self.agent_requests, self.llm_requests = [], []

    async def agent(self, request):
        self.agent_requests.append(request)
        text = "```json\n" + json.dumps({"intent_type": "specialist_done", "payload": _DONE}) + "\n```"
        return AgentReply(text, patches=(PatchFile("vllm/attn.py", "-a\n+b\n"),), finish_reason="end_turn")

    async def llm(self, request):
        self.llm_requests.append(request)
        if self.critic_fails:
            raise PortError("gateway down", retryable=True)
        rows = [{"target_proposal_msg_id": n, "verdict": v, "reasoning": n} for n, v in self.ruling]
        return LlmReply(json.dumps({"review_verdicts": rows}))


def test_v0_honours_the_contract_and_marks_its_points() -> None:
    brain = validate_brain(create_brain({}))
    assert brain.name == "v0"
    assert sorted(points_of(brain)) == ["critic.review", "main.propose", "main.review", "specialist.propose"]


async def test_a_proposal_is_what_the_critic_approves_of_the_specialists_candidates() -> None:
    ports = _Ports([("freeform-patch", "approve"), ("a", "reject"), ("b", "advise")])
    proposal = check_proposal(_ASK, await create_brain({}).propose(_STATE, _ASK, ports))
    assert [c.name for c in proposal.candidates] == ["freeform-patch", "b"]
    assert proposal.candidates[0].patches[0].path == "vllm/attn.py"
    assert (proposal.summary, proposal.confidence) == ("two flags and a patch", 0.7)
    assert proposal.questions == ("is fp8 kv allowed?",)
    assert "decode attention takes the naive path" in proposal.findings
    assert any(f.startswith("Critic held back a: reject") for f in proposal.findings)
    (request,) = ports.agent_requests
    assert request.writes_source and request.max_turns == 30
    assert "### Domain focus — kernel_switch_specialist" in request.system_prompt


async def test_without_the_critic_or_when_it_fails_every_candidate_is_proposed() -> None:
    names = ["freeform-patch", "a", "b"]
    unreviewed = await create_brain({"critic": False}).propose(_STATE, _ASK, _Ports())
    assert [c.name for c in unreviewed.candidates] == names
    failed = await create_brain({}).propose(_STATE, _ASK, _Ports(critic_fails=True))
    assert [c.name for c in failed.candidates] == names
    assert any("the Critic failed" in f for f in failed.findings)


async def test_an_ask_without_patches_drops_the_edit_and_runs_read_only() -> None:
    ask = ProposeAsk(task="t", patches=False)
    ports = _Ports([("a", "approve"), ("b", "approve")])
    proposal = check_proposal(ask, await create_brain({}).propose(_STATE, ask, ports))
    assert [c.name for c in proposal.candidates] == ["a", "b"]
    assert not ports.agent_requests[0].writes_source


async def test_a_domain_that_authors_no_candidates_proposes_nothing_without_running() -> None:
    ports = _Ports()
    ask = ProposeAsk(domain="research_scout_specialist")
    proposal = await create_brain({}).propose(_STATE, ask, ports)
    assert proposal.candidates == () and "authors no candidates" in proposal.summary
    assert ports.agent_requests == []


async def test_a_review_is_the_critics_ruling() -> None:
    ask = ReviewAsk((Candidate("a", ConfigDelta("--a")), Candidate("b", ConfigDelta("--b"))))
    review = check_review(ask, await create_brain({}).review(_STATE, ask, _Ports([("a", "approve")])))
    assert [v.decision for v in review.verdicts] == [Decision.APPROVE, Decision.DEFER]

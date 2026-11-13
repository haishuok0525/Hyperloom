# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0: today's specialist and Critic decisions behind the contract.

A proposal is what today's round brings to the bench: the specialist the ask names (or the freeform specialist,
anchored at the worst gap) runs on the agent port, and the Critic reviews its candidates before any is measured;
what the Critic does not approve is not proposed. A review is the Critic's ruling alone.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from hyperloom.brain.critic import review_candidates
from hyperloom.brain.specialist import run_specialist
from hyperloom.brain_api import (
    CONTRACT_VERSION,
    Decision,
    PortError,
    Ports,
    Proposal,
    ProposeAsk,
    Review,
    ReviewAsk,
    SearchState,
    eval_point,
)


class BrainV0:
    """The brain that reproduces the current behaviour.

    ``config["critic"]`` (default true) says whether a proposal goes through the Critic.
    """

    name = "v0"
    contract_version = CONTRACT_VERSION

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.critic = bool(self.config.get("critic", True))

    def eval_points(self) -> dict[str, Callable[..., Any]]:
        """The units a proposal is made of, each scorable on its own."""
        return {"specialist.propose": self.specialist, "critic.review": self.critique}

    @eval_point("main.propose")
    async def propose(self, state: SearchState, ask: ProposeAsk, ports: Ports) -> Proposal:
        """The specialist's candidates the Critic approves, in the specialist's order."""
        proposal = await self.specialist(state, ask, ports)
        if not (self.critic and proposal.candidates):
            return proposal
        try:
            review = await self.critique(state, ReviewAsk(proposal.candidates), ports)
        except PortError as exc:
            return Proposal(
                proposal.candidates,
                summary=proposal.summary,
                confidence=proposal.confidence,
                findings=(*proposal.findings, f"proposed unreviewed: the Critic failed ({exc})"),
                questions=proposal.questions,
            )
        approved = {v.candidate for v in review.verdicts if v.decision is Decision.APPROVE}
        kept = tuple(c for c in proposal.candidates if c.name in approved)
        held = [f"{v.candidate}: {v.reason}" for v in review.verdicts if v.decision is not Decision.APPROVE]
        return Proposal(
            kept,
            summary=proposal.summary,
            confidence=proposal.confidence,
            findings=(*proposal.findings, *(f"Critic held back {h}" for h in held)),
            questions=proposal.questions,
        )

    @eval_point("main.review")
    async def review(self, state: SearchState, ask: ReviewAsk, ports: Ports) -> Review:
        """The Critic's ruling."""
        return await self.critique(state, ask, ports)

    @eval_point("specialist.propose")
    async def specialist(self, state: SearchState, ask: ProposeAsk, ports: Ports) -> Proposal:
        """What the specialist that answers ``ask`` delivers, unreviewed."""
        return await run_specialist(state, ask, ports)

    @eval_point("critic.review")
    async def critique(self, state: SearchState, ask: ReviewAsk, ports: Ports) -> Review:
        """The Critic's verdict on each candidate."""
        return await review_candidates(state, ask, ports)

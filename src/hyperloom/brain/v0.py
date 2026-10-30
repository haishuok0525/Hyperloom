# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0: today's decisions, moved behind the contract one call site at a time.

A call site that has not moved yet decides nothing here; the body still runs its own code for it.
"""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain.critic import CRITIC_ROLE, PATCH_APPROVAL_VERDICTS, SETTLEMENT_KEY, review_turn, settle_verdict
from hyperloom.brain.orchestrate import ORCHESTRATION_ROLE, orchestration_turn
from hyperloom.brain.pump import (
    AUTO_BENCH_KEY,
    CANDIDATE_DISCOVERY_KEY,
    FRAMEWORK_AGENT_KEY,
    FRAMEWORK_PUMP_KEY,
    PREDICTOR_ASK_KEY,
    PREDICTOR_ENTRY_KEY,
    RESEARCH_SCOUT_KEY,
    STATIC_RECON_KEY,
    TRAJECTORY_REVIEW_KEY,
    auto_bench,
    candidate_discovery,
    framework_agent_pump,
    framework_pump,
    predictor_ask,
    predictor_entry,
    research_scout,
    static_recon,
    trajectory_review,
)
from hyperloom.brain.roles import ROLE_INTENTS
from hyperloom.brain.settle import (
    CONFIG_DELIVERY_KEY,
    PATCH_DELIVERY_KEY,
    REAUTHOR_KEY,
    settle_config_delivery,
    settle_patch_delivery,
    settle_reauthor,
)
from hyperloom.brain.specialist import brief_turn
from hyperloom.brain_api import CONTRACT_VERSION, BrainTurn, Port, TickBatch

_PORTED_PUMP_STRATEGIES = (
    (FRAMEWORK_PUMP_KEY, framework_pump),
    (FRAMEWORK_AGENT_KEY, framework_agent_pump),
    (CANDIDATE_DISCOVERY_KEY, candidate_discovery),
    (AUTO_BENCH_KEY, auto_bench),
    (PREDICTOR_ASK_KEY, predictor_ask),
    (STATIC_RECON_KEY, static_recon),
    (TRAJECTORY_REVIEW_KEY, trajectory_review),
)


class BrainV0:
    """The brain that reproduces the current behaviour."""

    name = "v0"
    contract_version = CONTRACT_VERSION
    role_intents = ROLE_INTENTS
    patch_approval_verdicts = PATCH_APPROVAL_VERDICTS

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    async def on_llm_turn(self, batch: TickBatch, port: Port) -> BrainTurn:
        """The Critic turn or the orchestration turn, whichever role's model the body lent."""
        if CRITIC_ROLE in port.llm:
            return await review_turn(batch, port)
        if ORCHESTRATION_ROLE in port.llm:
            return await orchestration_turn(batch, port)
        return BrainTurn(brain_state=batch.brain_state)

    async def on_settlement(self, batch: TickBatch, port: Port) -> BrainTurn:
        """The follow-up the body asks about: a Critic verdict, a specialist's deliverable or a re-author."""
        data = batch.snapshot.data
        if SETTLEMENT_KEY in data:
            return settle_verdict(batch)
        if PATCH_DELIVERY_KEY in data:
            return settle_patch_delivery(batch)
        if CONFIG_DELIVERY_KEY in data:
            return await settle_config_delivery(batch, port)
        if REAUTHOR_KEY in data:
            return await settle_reauthor(batch, port)
        return BrainTurn(brain_state=batch.brain_state)

    async def on_phase_pump(self, batch: TickBatch, port: Port) -> BrainTurn:
        """The FRAMEWORK pump's steps, its candidate track, bench and predictor, or whether an internal specialist is
        dispatched."""
        data = batch.snapshot.data
        for key, strategy in _PORTED_PUMP_STRATEGIES:
            if key in data:
                return await strategy(batch, port)
        if RESEARCH_SCOUT_KEY in data:
            return research_scout(batch)
        if PREDICTOR_ENTRY_KEY in data:
            return predictor_entry(batch)
        return BrainTurn(brain_state=batch.brain_state)

    async def on_brief(self, batch: TickBatch, port: Port) -> BrainTurn:
        """The specialist brief."""
        return brief_turn(batch)

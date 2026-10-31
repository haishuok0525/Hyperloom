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
    CONC_SWEEP_KEY,
    CUSTOM_SPECIALIST_KEY,
    CYCLE_REPROFILE_KEY,
    DISPATCH_ORDER_KEY,
    EXPLORE_COVERAGE_KEY,
    FRAMEWORK_AGENT_KEY,
    FRAMEWORK_PUMP_KEY,
    KERNEL_AGENT_KEY,
    KERNEL_ENTER_KEY,
    MACRO_FOCUS_KEY,
    PREDICTOR_ASK_KEY,
    PREDICTOR_ENTRY_KEY,
    RESEARCH_SCOUT_KEY,
    STALLED_DOMAIN_KEY,
    STATIC_RECON_KEY,
    SWEEP_DIRECTIVE_KEY,
    TRAJECTORY_REVIEW_KEY,
    WAVE_SPLIT_KEY,
    auto_bench,
    candidate_discovery,
    conc_sweep,
    custom_specialist,
    cycle_reprofile,
    dispatch_order,
    explore_coverage,
    framework_agent_pump,
    framework_pump,
    kernel_agent,
    kernel_enter,
    macro_focus,
    predictor_ask,
    predictor_entry,
    research_scout,
    specialist_wave,
    stalled_domain,
    static_recon,
    sweep_directive,
    trajectory_review,
)
from hyperloom.brain.roles import ROLE_INTENTS
from hyperloom.brain.score import PROPOSAL_SCORE_KEY, score_proposals
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

#: Body commit on fork main that Brain v0 was released against (P4e tip; P5 body wiring ships with this release).
BODY_BASELINE_COMMIT = "0a93533882f3d8bffa52c092a2a7c1976975f22f"

_PORTED_PUMP_STRATEGIES = (
    (FRAMEWORK_PUMP_KEY, framework_pump),
    (FRAMEWORK_AGENT_KEY, framework_agent_pump),
    (CANDIDATE_DISCOVERY_KEY, candidate_discovery),
    (AUTO_BENCH_KEY, auto_bench),
    (PREDICTOR_ASK_KEY, predictor_ask),
    (STATIC_RECON_KEY, static_recon),
    (TRAJECTORY_REVIEW_KEY, trajectory_review),
    (STALLED_DOMAIN_KEY, stalled_domain),
    (CUSTOM_SPECIALIST_KEY, custom_specialist),
    (WAVE_SPLIT_KEY, specialist_wave),
    (MACRO_FOCUS_KEY, macro_focus),
    (CYCLE_REPROFILE_KEY, cycle_reprofile),
    (KERNEL_ENTER_KEY, kernel_enter),
    (KERNEL_AGENT_KEY, kernel_agent),
    (SWEEP_DIRECTIVE_KEY, sweep_directive),
    (CONC_SWEEP_KEY, conc_sweep),
    (DISPATCH_ORDER_KEY, dispatch_order),
    (EXPLORE_COVERAGE_KEY, explore_coverage),
)


class BrainV0:
    """The brain that reproduces the current behaviour."""

    name = "v0"
    contract_version = CONTRACT_VERSION
    role_intents = ROLE_INTENTS
    patch_approval_verdicts = PATCH_APPROVAL_VERDICTS
    body_baseline_commit = BODY_BASELINE_COMMIT

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
        """Phase strategy: FRAMEWORK pieces, dispatch, macro-cycle, KERNEL, SWEEP, or an internal specialist."""
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

    async def on_score(self, batch: TickBatch, port: Port) -> BrainTurn:
        """Advisory multi-model scoring of a specialist proposal set."""
        if PROPOSAL_SCORE_KEY in batch.snapshot.data:
            return await score_proposals(batch, port)
        return BrainTurn(brain_state=batch.brain_state)

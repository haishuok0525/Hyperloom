# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The body steps the brain runs the FRAMEWORK pump from, each still deciding its own work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..brain_link import QueryAnswer, step_answer

if TYPE_CHECKING:
    from .framework import FrameworkPhase

# Snapshot key the brain runs the FRAMEWORK pump under.
FRAMEWORK_PUMP_KEY = "framework_pump"


def framework_pump_steps(phase: "FrameworkPhase", stage: list[str]) -> dict[str, QueryAnswer]:
    """The ``pump.*`` steps ``phase`` answers; ``stage[0]`` names the one a fault is filed under."""
    c = phase._coord

    async def _advisory_plateau() -> None:
        phase._record_advisory_plateau()

    async def _predictor() -> None:
        stage[0] = "framework_predictor"
        await phase._predictor_pump().step(phase.shared_state)

    steps = {
        "pump.framework_agent": lambda: phase._pump_framework_agent_phase(),
        "pump.bench_untested": lambda: phase._maybe_bench_untested_proposals(),
        "pump.research_scout": lambda: c.phase_internal.maybe_enqueue_explore_research_scout(),
        "pump.config_plateau_recon": lambda: c.phase_internal.maybe_enqueue_config_plateau_static_recon(),
        "pump.stalled_domain": lambda: c.specialist_dispatch.maybe_force_stalled_domain_specialist(),
        "pump.custom_specialist": lambda: c.specialist_dispatch.ensure_custom_specialist(),
        "pump.advisory_plateau": _advisory_plateau,
        "pump.predictor": _predictor,
        "pump.finish_custom_cycle": lambda: phase._maybe_finish_custom_only_cycle(),
    }
    return {name: step_answer(run) for name, run in steps.items()}

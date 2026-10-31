# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phase strategy: FRAMEWORK pump pieces, dispatch, macro-cycle, KERNEL, SWEEP, and internal specialists."""

from __future__ import annotations

from hyperloom.brain.pump.bench import AUTO_BENCH_KEY, auto_bench
from hyperloom.brain.pump.candidates import (
    CANDIDATE_DISCOVERY_KEY,
    FRAMEWORK_AGENT_KEY,
    candidate_discovery,
    framework_agent_pump,
)
from hyperloom.brain.pump.dispatch import (
    CUSTOM_SPECIALIST_KEY,
    STALLED_DOMAIN_KEY,
    WAVE_SPLIT_KEY,
    custom_specialist,
    specialist_wave,
    stalled_domain,
)
from hyperloom.brain.pump.dispatch_order import DISPATCH_ORDER_KEY, dispatch_order
from hyperloom.brain.pump.explore_coverage import EXPLORE_COVERAGE_KEY, explore_coverage
from hyperloom.brain.pump.framework import FRAMEWORK_PUMP_KEY, framework_pump
from hyperloom.brain.pump.internal import (
    RESEARCH_SCOUT_KEY,
    STATIC_RECON_KEY,
    TRAJECTORY_REVIEW_KEY,
    research_scout,
    static_recon,
    trajectory_review,
)
from hyperloom.brain.pump.kernel import KERNEL_AGENT_KEY, KERNEL_ENTER_KEY, kernel_agent, kernel_enter
from hyperloom.brain.pump.macro import CYCLE_REPROFILE_KEY, MACRO_FOCUS_KEY, cycle_reprofile, macro_focus
from hyperloom.brain.pump.predictor import PREDICTOR_ASK_KEY, PREDICTOR_ENTRY_KEY, predictor_ask, predictor_entry
from hyperloom.brain.pump.sweep import CONC_SWEEP_KEY, SWEEP_DIRECTIVE_KEY, conc_sweep, sweep_directive

__all__ = [
    "AUTO_BENCH_KEY",
    "CANDIDATE_DISCOVERY_KEY",
    "CONC_SWEEP_KEY",
    "CUSTOM_SPECIALIST_KEY",
    "CYCLE_REPROFILE_KEY",
    "DISPATCH_ORDER_KEY",
    "EXPLORE_COVERAGE_KEY",
    "FRAMEWORK_AGENT_KEY",
    "FRAMEWORK_PUMP_KEY",
    "KERNEL_AGENT_KEY",
    "KERNEL_ENTER_KEY",
    "MACRO_FOCUS_KEY",
    "PREDICTOR_ASK_KEY",
    "PREDICTOR_ENTRY_KEY",
    "RESEARCH_SCOUT_KEY",
    "STALLED_DOMAIN_KEY",
    "STATIC_RECON_KEY",
    "SWEEP_DIRECTIVE_KEY",
    "TRAJECTORY_REVIEW_KEY",
    "WAVE_SPLIT_KEY",
    "auto_bench",
    "candidate_discovery",
    "conc_sweep",
    "custom_specialist",
    "cycle_reprofile",
    "dispatch_order",
    "explore_coverage",
    "framework_agent_pump",
    "framework_pump",
    "kernel_agent",
    "kernel_enter",
    "macro_focus",
    "predictor_ask",
    "predictor_entry",
    "research_scout",
    "specialist_wave",
    "stalled_domain",
    "static_recon",
    "sweep_directive",
    "trajectory_review",
]

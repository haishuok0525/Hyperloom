# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phase strategy: the FRAMEWORK pump's order, its candidate track and bench, the predictor, and the internal
specialists the phases dispatch."""

from __future__ import annotations

from hyperloom.brain.pump.bench import AUTO_BENCH_KEY, auto_bench
from hyperloom.brain.pump.candidates import (
    CANDIDATE_DISCOVERY_KEY,
    FRAMEWORK_AGENT_KEY,
    candidate_discovery,
    framework_agent_pump,
)
from hyperloom.brain.pump.framework import FRAMEWORK_PUMP_KEY, framework_pump
from hyperloom.brain.pump.internal import (
    RESEARCH_SCOUT_KEY,
    STATIC_RECON_KEY,
    TRAJECTORY_REVIEW_KEY,
    research_scout,
    static_recon,
    trajectory_review,
)
from hyperloom.brain.pump.predictor import PREDICTOR_ASK_KEY, PREDICTOR_ENTRY_KEY, predictor_ask, predictor_entry

__all__ = [
    "AUTO_BENCH_KEY",
    "CANDIDATE_DISCOVERY_KEY",
    "FRAMEWORK_AGENT_KEY",
    "FRAMEWORK_PUMP_KEY",
    "PREDICTOR_ASK_KEY",
    "PREDICTOR_ENTRY_KEY",
    "RESEARCH_SCOUT_KEY",
    "STATIC_RECON_KEY",
    "TRAJECTORY_REVIEW_KEY",
    "auto_bench",
    "candidate_discovery",
    "framework_agent_pump",
    "framework_pump",
    "predictor_ask",
    "predictor_entry",
    "research_scout",
    "static_recon",
    "trajectory_review",
]

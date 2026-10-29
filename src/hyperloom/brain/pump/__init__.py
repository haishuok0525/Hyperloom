# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The phase strategy: the FRAMEWORK pump's order and the internal specialists the phases dispatch."""

from __future__ import annotations

from hyperloom.brain.pump.framework import FRAMEWORK_PUMP_KEY, framework_pump
from hyperloom.brain.pump.internal import (
    RESEARCH_SCOUT_KEY,
    STATIC_RECON_KEY,
    TRAJECTORY_REVIEW_KEY,
    research_scout,
    static_recon,
    trajectory_review,
)

__all__ = [
    "FRAMEWORK_PUMP_KEY",
    "RESEARCH_SCOUT_KEY",
    "STATIC_RECON_KEY",
    "TRAJECTORY_REVIEW_KEY",
    "framework_pump",
    "research_scout",
    "static_recon",
    "trajectory_review",
]

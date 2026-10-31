# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared thresholds for Coordinator-owned specialist dispatch."""

from __future__ import annotations

#: Optimisation rounds a domain may go without a specialist dispatch before a force.
FORCE_STALLED_SPECIALIST_ROUNDS: int = 8
#: Optimisation rounds a domain may go without a KEEP before a force.
FORCE_STALLED_KEEP_ROUNDS: int = 12
#: Gap id for the custom dispatch when the ledger has no gap yet.
CUSTOM_SESSION_GAP_ID: str = "gap.custom.session"

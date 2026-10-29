# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Critic ruling as the brain settles it, for tests that hand the router a ruling directly."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from hyperloom.brain.critic import verdict_outcome
from hyperloom.orchestrator.loop.verdicts import CriticRuling


def settled(pending: Any, ruling: CriticRuling) -> CriticRuling:
    """``ruling`` with the outcome the brain settles its verdict on ``pending`` to."""
    return replace(ruling, outcome=verdict_outcome(ruling.verdict, pending.action_name, pending.payload or {}))

# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestration turn: what the orchestration model is asked, and what becomes of the intents it emits."""

from __future__ import annotations

from hyperloom.brain.orchestrate.prompt import OrchestrationPrompt, compose_prompt
from hyperloom.brain.orchestrate.turn import ORCHESTRATION_KEY, ORCHESTRATION_ROLE, orchestration_turn

__all__ = ["ORCHESTRATION_KEY", "ORCHESTRATION_ROLE", "OrchestrationPrompt", "compose_prompt", "orchestration_turn"]

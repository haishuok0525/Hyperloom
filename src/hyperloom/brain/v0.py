# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0: today's decisions, moved behind the contract one call site at a time.

A call site that has not moved yet decides nothing here; the body still runs its own code for it.
"""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import CONTRACT_VERSION, BrainTurn, LlmHandle, TickBatch


class BrainV0:
    """The brain that reproduces the current behaviour."""

    name = "v0"
    contract_version = CONTRACT_VERSION

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    async def on_llm_turn(self, batch: TickBatch, llm: Mapping[str, LlmHandle]) -> BrainTurn:
        """No model turn has moved yet."""
        return BrainTurn(brain_state=batch.brain_state)

    async def on_settlement(self, batch: TickBatch, llm: Mapping[str, LlmHandle]) -> BrainTurn:
        """No settlement follow-up has moved yet."""
        return BrainTurn(brain_state=batch.brain_state)

    async def on_phase_pump(self, batch: TickBatch, llm: Mapping[str, LlmHandle]) -> BrainTurn:
        """No phase strategy has moved yet."""
        return BrainTurn(brain_state=batch.brain_state)

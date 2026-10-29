# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0: today's decisions, moved behind the contract one call site at a time.

A call site that has not moved yet decides nothing here; the body still runs its own code for it.
"""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain.critic import CRITIC_ROLE, review_turn
from hyperloom.brain.orchestrate import ORCHESTRATION_ROLE, orchestration_turn
from hyperloom.brain.roles import ROLE_INTENTS
from hyperloom.brain.specialist import brief_turn
from hyperloom.brain_api import CONTRACT_VERSION, BrainTurn, Port, TickBatch


class BrainV0:
    """The brain that reproduces the current behaviour."""

    name = "v0"
    contract_version = CONTRACT_VERSION
    role_intents = ROLE_INTENTS

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
        """No settlement follow-up has moved yet."""
        return BrainTurn(brain_state=batch.brain_state)

    async def on_phase_pump(self, batch: TickBatch, port: Port) -> BrainTurn:
        """No phase strategy has moved yet."""
        return BrainTurn(brain_state=batch.brain_state)

    async def on_brief(self, batch: TickBatch, port: Port) -> BrainTurn:
        """The specialist brief."""
        return brief_turn(batch)

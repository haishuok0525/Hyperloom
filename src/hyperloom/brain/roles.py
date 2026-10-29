# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which intents each model role may emit: orchestration drives the run, the Critic only judges it."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from hyperloom.brain_api import INTENT_TOOLS

_BASE_INTENTS = frozenset({"send_message", "alert"})
# Orchestration is the only role with REQUEST authority.
_ORCHESTRATION_INTENTS = _BASE_INTENTS | {
    "propose_action",
    "delegate",
    "request",
    "extend_lease",
    "prune_branch",
    "escalate_strategy_change",
}
_CRITIC_INTENTS = _BASE_INTENTS | {"review_verdict"}


def _in_contract_order(intents: frozenset[str]) -> tuple[str, ...]:
    return tuple(intent for intent in INTENT_TOOLS if intent in intents)


#: Role -> the intent types its model may emit, in contract order.
ROLE_INTENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "orchestration": _in_contract_order(_ORCHESTRATION_INTENTS),
        "critic": _in_contract_order(_CRITIC_INTENTS),
    }
)


__all__ = ["ROLE_INTENTS"]

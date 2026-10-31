# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The order the dispatcher visits queued tasks in (R-06)."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

DISPATCH_ORDER_KEY = "dispatch_order"

# Phase → task kinds the phase wants ahead of ordinary work (same as today's body map).
_PHASE_FIRST: dict[str, frozenset[str]] = {
    "ENABLEMENT": frozenset({"roofline"}),
    "FRAMEWORK_AGENT": frozenset({"integrate_patch"}),
}


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


async def dispatch_order(batch: TickBatch, _port: Port) -> BrainTurn:
    """Order queued tasks: operator work, then the phase's dispatch-first kinds, then the rest."""
    request: Mapping[str, Any] = batch.snapshot.data[DISPATCH_ORDER_KEY]
    phase = str(request["phase"] or "").strip().upper()
    first = _PHASE_FIRST.get(phase, frozenset())
    tasks = list(request["tasks"] or [])

    def tier(row: Mapping[str, Any]) -> int:
        if row.get("operator_work"):
            return 0
        return 1 if str(row.get("kind") or "") in first else 2

    ordered = sorted(tasks, key=tier)
    return _turn(batch, ToolCall("dispatch.order", {"ids": [str(row["id"]) for row in ordered]}))

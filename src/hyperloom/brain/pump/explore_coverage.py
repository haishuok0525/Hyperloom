# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a delegated explore grid adds nothing new on the current stack (R-01)."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

EXPLORE_COVERAGE_KEY = "explore_coverage"


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


async def explore_coverage(batch: TickBatch, _port: Port) -> BrainTurn:
    """Refuse an explore grid the body already found fully covered, unless a framework lever rewrite is pending."""
    request: Mapping[str, Any] = batch.snapshot.data[EXPLORE_COVERAGE_KEY]
    if not request["is_explore"] or request["lever_grid"] or not request["covered"]:
        return _turn(batch)
    return _turn(
        batch,
        ToolCall(
            "explore.refuse_covered",
            {
                "message": request["message"],
                "hint": request["hint"],
            },
        ),
    )

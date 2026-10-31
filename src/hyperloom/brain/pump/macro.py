# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Macro-cycle focus emission and whether a reopened cycle forces a reprofile."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

MACRO_FOCUS_KEY = "macro_focus"
CYCLE_REPROFILE_KEY = "cycle_reprofile"

_SWEEP_PHASE = "SWEEP"


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


async def macro_focus(batch: TickBatch, _port: Port) -> BrainTurn:
    """Emit the advisory focus plan the body scored for this cycle (M-01).

    Scoring lives in the body's ``focus_plan`` so sync call sites keep working; the brain names the plan that is
    recorded. A later brain can replace the scorer without changing the tool.
    """
    plan = dict(batch.snapshot.data[MACRO_FOCUS_KEY]["plan"])
    return _turn(batch, ToolCall("macro.set_focus", plan))


async def cycle_reprofile(batch: TickBatch, _port: Port) -> BrainTurn:
    """Force a fresh analysis when a SWEEP origin reopens a macro-cycle (M-02)."""
    request: Mapping[str, Any] = batch.snapshot.data[CYCLE_REPROFILE_KEY]
    if str(request["from_phase"] or "").strip().upper() != _SWEEP_PHASE:
        return _turn(batch)
    if int(request["macro_cycle"] or 0) <= 0:
        return _turn(batch)
    return _turn(batch, ToolCall("prelude.reprofile", {"reason": "cycle_start"}))

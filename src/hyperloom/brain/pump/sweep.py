# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SWEEP: the cycle-directive turn and whether auto conc_sweep runs."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

SWEEP_DIRECTIVE_KEY = "sweep_directive"
_SKIP_CONC = "sweep.skip_conc"
CONC_SWEEP_KEY = "conc_sweep"

#: Validated gain must clear the last watermark by this much before another auto sweep.
_GAIN_EPS = 1e-6


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


async def sweep_directive(batch: TickBatch, _port: Port) -> BrainTurn:
    """Ask Orchestration once per macro-cycle how the next cycle should open (S-01)."""
    request: Mapping[str, Any] = batch.snapshot.data[SWEEP_DIRECTIVE_KEY]
    if not request["has_orchestration"] or request["user_supplied_prompt"]:
        return _turn(batch)
    if request["memory_for_cycle"] == request["macro_cycle"]:
        return _turn(batch)
    if not request["reloop_feasible"]:
        return _turn(batch)
    return _turn(batch, ToolCall("sweep.ask_directive", {}))


async def conc_sweep(batch: TickBatch, _port: Port) -> BrainTurn:
    """Enqueue auto conc_sweep on SWEEP entry, or name why not (S-02).

    Stack recovery and KEEP drains run in the body before this ask.
    """
    request: Mapping[str, Any] = batch.snapshot.data[CONC_SWEEP_KEY]
    if not request["enabled"]:
        return _turn(batch, ToolCall(_SKIP_CONC, {"reason": "disabled"}))
    prev = request.get("watermark") or {}
    prev_validated = prev.get("cumulative_gain_validated_at_record") if isinstance(prev, dict) else None
    cur = float(request["validated_gain"] or 0.0)
    if prev and isinstance(prev_validated, (int, float)) and cur <= float(prev_validated) + _GAIN_EPS:
        return _turn(
            batch,
            ToolCall(
                _SKIP_CONC,
                {"reason": "no_validated_gain_since_last_conc_sweep", "validated_gain": cur},
            ),
        )
    if request.get("budget_denied"):
        return _turn(batch, ToolCall(_SKIP_CONC, {"reason": "budget", "denied": request["budget_denied"]}))
    return _turn(batch, ToolCall("sweep.enqueue_conc", {"reason": "phase_entry"}))

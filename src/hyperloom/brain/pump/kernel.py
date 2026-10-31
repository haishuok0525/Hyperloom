# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KERNEL entry route, the gemm gate, and the agent run's order (K-01..K-03)."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

KERNEL_ENTER_KEY = "kernel_enter"
KERNEL_AGENT_KEY = "kernel_agent"

_GEMM_ELIGIBLE = frozenset({"sglang", "vllm", "vllm-aiter"})
_GEMM_TERMINAL = frozenset({"ok", "succeeded", "success", "complete", "completed", "skipped", "failed"})
_ROUTE_GEAK = "geak"
_ROUTE_FORGE = "forge"
_ROUTE_FORGE_NO_GEMM = "forge_no_gemm"
_ROUTE_FORGE_GEMM = "forge_gemm"


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


def _gemm_still_needed(framework: str, last_status: str) -> bool:
    if str(framework or "").strip().lower() not in _GEMM_ELIGIBLE:
        return False
    return str(last_status or "").strip().lower() not in _GEMM_TERMINAL


async def kernel_enter(batch: TickBatch, _port: Port) -> BrainTurn:
    """Open the KERNEL timeline and enqueue the carrier task when the phase is enabled (K-01)."""
    request: Mapping[str, Any] = batch.snapshot.data[KERNEL_ENTER_KEY]
    if not request["enabled"]:
        return _turn(batch)
    route = _ROUTE_GEAK if request["geak"] else _ROUTE_FORGE
    return _turn(
        batch,
        ToolCall(
            "kernel.enter",
            {
                "route": route,
                "route_reason": request["route_reason"],
                "from_phase": request["from_phase"],
                "macro_cycle": int(request["macro_cycle"] or 0),
            },
        ),
    )


async def kernel_agent(batch: TickBatch, _port: Port) -> BrainTurn:
    """Choose geak, forge without gemm, or forge after gemm for one KERNEL agent run (K-02/K-03)."""
    request: Mapping[str, Any] = batch.snapshot.data[KERNEL_AGENT_KEY]
    if request["geak"]:
        route = _ROUTE_GEAK
    elif request["skip_gemm"] or not _gemm_still_needed(
        str(request["framework"] or ""), str(request["last_gemm_status"] or "")
    ):
        route = _ROUTE_FORGE_NO_GEMM
    else:
        route = _ROUTE_FORGE_GEMM
    return _turn(batch, ToolCall("kernel.run", {"route": route, "from_phase": request["from_phase"]}))

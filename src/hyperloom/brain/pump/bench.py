# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which untested proposals the FRAMEWORK phase benches on its own, and when."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

# Snapshot key the body asks under.
AUTO_BENCH_KEY = "auto_bench"

# Specialist proposal rows benched per automatic explore grid.
AUTO_BENCH_GRID_CAP = 4
_FRAMEWORK_PHASE = "FRAMEWORK_AGENT"


async def auto_bench(batch: TickBatch, port: Port) -> BrainTurn:
    """Bench the operator-defined specialist's proposals ahead of every gate; the rest one capped grid at a time,
    once no explore is queued or running and the framework candidate track is open."""
    request: Mapping[str, Any] = batch.snapshot.data[AUTO_BENCH_KEY]
    done = BrainTurn(brain_state=batch.brain_state)
    if str(request["phase"] or "").strip().upper() != _FRAMEWORK_PHASE:
        return done
    admission = await port.call(ToolCall("query.dispatch_admission", {}))
    if admission["frozen"] or admission["paused"]:
        return done
    queue = await port.call(ToolCall("query.bench_queue", {}))
    if queue["custom_rows"] and not queue["operator_work"]:
        if (await _bench(port, "custom", None))["created"]:
            return done
    if request["custom_specialist_only"] or request["phase_done"] or queue["explores"]:
        return done
    if (await port.call(ToolCall("query.untested_rows", {})))["count"]:
        await _bench(port, "untested", AUTO_BENCH_GRID_CAP)
    return done


async def _bench(port: Port, rows: str, limit: int | None) -> Mapping[str, Any]:
    return await port.call(ToolCall("bench.enqueue", {"rows": rows, "limit": limit}))

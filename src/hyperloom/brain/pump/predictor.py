# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""When the predictor is asked at a decision point, and how long FRAMEWORK entry waits for its answer."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

# Snapshot keys the body asks under.
PREDICTOR_ASK_KEY = "predictor_ask"
PREDICTOR_ENTRY_KEY = "predictor_entry"

#: One explore variant on a large model, server restart plus benchmark. An
#: answer is worth asking for only while the phase can still bench one variant.
MIN_VARIANT_SEC = 600.0
#: Longest a decision point is held for a re-profile in flight; past it the predictor answers on the profile there is.
MAX_PROFILE_WAIT_SEC = 900.0
#: Longest FRAMEWORK entry waits for the answer before orchestration's first turn.
MAX_ENTRY_WAIT_SEC = 300.0


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


def _skip(batch: TickBatch, reason: str) -> BrainTurn:
    return _turn(batch, ToolCall("predictor.skip", {"reason": reason}))


def _skip_reason(request: Mapping[str, Any]) -> str | None:
    """Why not to ask, before the phase clock is read; ``None`` to go on."""
    if not request["enabled"]:
        return "disabled"
    if not request["framework_supported"]:
        return f"no flag catalogue for framework {request['framework']!r}"
    if request["asked"]:
        return "decision point already asked"
    return None


async def predictor_ask(batch: TickBatch, port: Port) -> BrainTurn:
    """Ask once per decision point while the phase can still bench one variant.

    A KEEP moves the decision point at once, and the re-profile it triggers moves it again minutes later; an answer
    in between reads the profile of the stack before the KEEP, so the point is held for the re-profile a while.
    """
    request: Mapping[str, Any] = batch.snapshot.data[PREDICTOR_ASK_KEY]
    reason = _skip_reason(request)
    if reason is not None:
        return _skip(batch, reason)
    headroom = (await port.call(ToolCall("query.phase_headroom", {})))["headroom_sec"]
    if headroom is not None and headroom < MIN_VARIANT_SEC:
        return _skip(batch, f"phase headroom {headroom:.0f}s is below one variant")
    hold = await port.call(ToolCall("query.profile_hold", {}))
    if hold["pending"] and hold["held_sec"] < MAX_PROFILE_WAIT_SEC:
        return _turn(batch)
    return _turn(batch, ToolCall("predictor.ask", {}))


def predictor_entry(batch: TickBatch) -> BrainTurn:
    """Hold orchestration's first turn for an answer that will be queued at the current decision point."""
    request: Mapping[str, Any] = batch.snapshot.data[PREDICTOR_ENTRY_KEY]
    if not (request["inflight"] and request["enqueues"] and request["current"]):
        return _turn(batch)
    return _turn(batch, ToolCall("predictor.wait", {"timeout_sec": MAX_ENTRY_WAIT_SEC}))

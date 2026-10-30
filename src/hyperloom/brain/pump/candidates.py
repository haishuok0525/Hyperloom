# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The FRAMEWORK candidate track: how many candidates are in review at once, which goes next, and what the phase
falls back to once none is left."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

# Snapshot keys the body asks under.
FRAMEWORK_AGENT_KEY = "framework_agent_pump"
CANDIDATE_DISCOVERY_KEY = "candidate_discovery"

# Empty or failed discovery rounds a phase tolerates before it stops asking.
DISCOVERY_RETRY_LIMIT = 3
_FRAMEWORK_PHASE = "FRAMEWORK_AGENT"


async def _ask(port: Port, name: str, args: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return await port.call(ToolCall(name, dict(args or {})))


async def framework_agent_pump(batch: TickBatch, port: Port) -> BrainTurn:
    """Keep up to the research lane's capacity of candidates in review; landing still serialises on the workspace.

    Once no candidate is left and none is in flight, ask for discovery, then a local source exploration while the
    source arm still pays, and close the candidate track when neither is dispatched.
    """
    request: Mapping[str, Any] = batch.snapshot.data[FRAMEWORK_AGENT_KEY]
    done = BrainTurn(brain_state=batch.brain_state)
    if str(request["phase"] or "").strip().upper() != _FRAMEWORK_PHASE or request["phase_done"]:
        return done
    in_flight = list((await _ask(port, "query.framework_inflight_candidates"))["ids"])
    submitted = False
    while len(in_flight) < max(1, request["research_lane_capacity"]):
        candidate = await _ask(port, "query.framework_next_candidate", {"exclude": in_flight})
        if not candidate["found"]:
            break
        key = (await _ask(port, "framework.submit_candidate", {"id": candidate["id"]}))["id"]
        if key not in in_flight:
            in_flight.append(key)
        submitted = True
    if submitted or in_flight or (await _ask(port, "query.framework_authoring_inflight"))["inflight"]:
        return done
    if (await _ask(port, "framework.discover", {"reason": "candidate_pool_empty"}))["dispatched"]:
        return done
    source_dry = (await _ask(port, "query.source_arm_plateau"))["plateaued"]
    if request["local_explore_enabled"] and not source_dry:
        if (await _ask(port, "framework.local_explore", {"reason": "no_new_candidates"}))["dispatched"]:
            return done
    await _ask(port, "framework.phase_done", {"reason": "no_candidates_and_discovery_exhausted"})
    return done


async def candidate_discovery(batch: TickBatch, port: Port) -> BrainTurn:
    """Ask for one discovery round at a time until the retry limit of empty or failed rounds is reached."""
    request: Mapping[str, Any] = batch.snapshot.data[CANDIDATE_DISCOVERY_KEY]
    if request["empty_discoveries"] >= DISCOVERY_RETRY_LIMIT or request["discover_failures"] >= DISCOVERY_RETRY_LIMIT:
        return BrainTurn(brain_state=batch.brain_state)
    name = (
        "discovery.wait"
        if (await _ask(port, "query.candidate_discovery_inflight"))["inflight"]
        else "discovery.dispatch"
    )
    return BrainTurn(tool_calls=(ToolCall(name, {}),), brain_state=batch.brain_state)

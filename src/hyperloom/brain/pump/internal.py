# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""When the phase strategy dispatches its Coordinator-owned read-only specialists: the research scout, static recon
and the plateau trajectory review."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

# Snapshot keys the body asks under.
RESEARCH_SCOUT_KEY = "research_scout"
STATIC_RECON_KEY = "static_recon"
TRAJECTORY_REVIEW_KEY = "trajectory_review"

# Config-arm rounds between research scouts when the session names no interval.
_SCOUT_INTERVAL = 3
# The phase a config-arm plateau re-runs static recon in.
_FRAMEWORK_PHASE = "FRAMEWORK_AGENT"
# Who reviews the trajectory when no roofline bottleneck points at a domain.
_DEFAULT_REVIEW_DOMAIN = "serving_specialist"


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


def research_scout(batch: TickBatch) -> BrainTurn:
    """Scout once in PRELUDE, then again every few config-arm rounds, once per round."""
    request: Mapping[str, Any] = batch.snapshot.data[RESEARCH_SCOUT_KEY]
    if not request["enabled"]:
        return _turn(batch)
    if request["prelude"]:
        return _turn(batch, ToolCall("specialist.research_scout", {"reason": "prelude_initial", "round_id": 0}))
    interval = max(1, int(request["interval"] or _SCOUT_INTERVAL))
    round_id = int(request["cursor"] or 0)
    if round_id <= 0 or (round_id % interval) != 0 or int(request["last_round"]) == round_id:
        return _turn(batch)
    return _turn(batch, ToolCall("specialist.research_scout", {"reason": "explore_periodic", "round_id": round_id}))


async def static_recon(batch: TickBatch, port: Port) -> BrainTurn:
    """Recon the source in PRELUDE, before any measurement exists, and again once per macro-cycle when the config
    arm goes dry: the profile then says which subsystem's predicates matter."""
    request: Mapping[str, Any] = batch.snapshot.data[STATIC_RECON_KEY]
    if not request["enabled"]:
        return _turn(batch)
    if request["prelude"]:
        return _turn(batch, ToolCall("specialist.static_recon", {"reason": "prelude_initial", "slug": "prelude"}))
    if str(request["phase"] or "").strip().upper() != _FRAMEWORK_PHASE:
        return _turn(batch)
    if not (await port.call(ToolCall("query.config_arm_plateau", {})))["plateaued"]:
        return _turn(batch)
    slug = f"cycle{int(request['macro_cycle'] or 0)}"
    return _turn(batch, ToolCall("specialist.static_recon", {"reason": "config_arm_plateau", "slug": slug}))


async def trajectory_review(batch: TickBatch, port: Port) -> BrainTurn:
    """On a plateau, have the specialist the dominant bottleneck points at propose fresh directions."""
    if not batch.snapshot.data[TRAJECTORY_REVIEW_KEY]["enabled"]:
        return _turn(batch)
    if not (await port.call(ToolCall("query.plateau_advisory", {})))["active"]:
        return _turn(batch)
    hinted = (await port.call(ToolCall("query.trajectory_direction", {})))["hinted_domain"]
    return _turn(batch, ToolCall("specialist.trajectory_review", {"domain": hinted or _DEFAULT_REVIEW_DOMAIN}))

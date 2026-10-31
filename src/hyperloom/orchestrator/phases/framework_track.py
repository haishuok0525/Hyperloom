# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The body steps the brain runs the FRAMEWORK candidate track, its discovery and the untested-proposal bench from."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from hyperloom.brain_api import CallSite

from . import machine_state as _phase_state
from ..brain_link import QueryAnswer, one_decision
from .proposal_bench import CUSTOM_PROPOSAL_DOMAIN

if TYPE_CHECKING:
    from .framework import FrameworkPhase

# Snapshot keys the brain answers these under.
FRAMEWORK_AGENT_KEY = "framework_agent_pump"
CANDIDATE_DISCOVERY_KEY = "candidate_discovery"
AUTO_BENCH_KEY = "auto_bench"
DISCOVERY_DECISIONS = ("discovery.dispatch", "discovery.wait")

# Whether a bench grid of the named rows takes the operator's priority.
_OPERATOR_PRIORITY = {"custom": True, "untested": False}


async def run_framework_agent_track(phase: "FrameworkPhase") -> None:
    """Let the brain drive one pass of the candidate track through :func:`framework_agent_steps`."""
    state = phase.shared_state
    request = {
        "phase": state.phase or "",
        "phase_done": bool(state.framework_agent_phase_done),
        "research_lane_capacity": int(state.research_lane_capacity),
        "local_explore_enabled": bool(state.framework_local_explore_enabled),
    }
    turn = await phase._ask_brain(
        FRAMEWORK_AGENT_KEY, request, framework_agent_steps(phase), call_site=CallSite.PHASE_PUMP
    )
    one_decision(turn, (), what="the FRAMEWORK candidate track")


def framework_agent_steps(phase: "FrameworkPhase") -> dict[str, QueryAnswer]:
    """The candidate track's queries and steps; a candidate's ``id`` is the handle the selection handed out."""
    state = phase.shared_state
    candidates: dict[str, dict[str, Any]] = {}

    async def _inflight(_args: Mapping[str, Any]) -> dict[str, Any]:
        ids = phase._authoring_inflight_candidate_ids(await phase.tasks.queued(), await phase.tasks.running())
        return {"ids": sorted(ids)}

    async def _next(args: Mapping[str, Any]) -> dict[str, Any]:
        candidate = phase._select_next_framework_agent_candidate(exclude_ids=set(args["exclude"]))
        if candidate is None:
            return {"found": False, "id": ""}
        handle = str(len(candidates))
        candidates[handle] = candidate
        return {"found": True, "id": handle}

    async def _submit(args: Mapping[str, Any]) -> dict[str, Any]:
        candidate = candidates[str(args["id"])]
        await phase._submit_framework_agent_candidate_for_review(
            candidate,
            audit=dict(candidate.get("audit") or {}),
            audit_step=str(candidate.get("route") or "author_via_specialist"),
        )
        return {"id": phase._framework_candidate_key(candidate)}

    async def _authoring(_args: Mapping[str, Any]) -> dict[str, Any]:
        return {"inflight": bool(await phase._framework_agent_authoring_inflight())}

    async def _discover(args: Mapping[str, Any]) -> dict[str, Any]:
        dispatched = bool(await phase._maybe_enqueue_candidate_discovery(reason=str(args["reason"])))
        if dispatched:
            state.save(phase.session_dir)
        return {"dispatched": dispatched}

    async def _source_dry(_args: Mapping[str, Any]) -> dict[str, Any]:
        return {"plateaued": bool(_phase_state.per_lever_dryness(state)[1]["source_arm_plateaued"])}

    async def _local_explore(args: Mapping[str, Any]) -> dict[str, Any]:
        dispatched = bool(await _dispatch_local_explore(phase, reason=str(args["reason"])))
        if dispatched:
            state.save(phase.session_dir)
        return {"dispatched": dispatched}

    async def _phase_done(args: Mapping[str, Any]) -> dict[str, Any]:
        phase._record_framework_agent_phase_done(
            reason=str(args["reason"]),
            failure_count=int(state.framework_agent_discover_failures or 0),
        )
        state.framework_agent_phase_done = True
        state.save(phase.session_dir)
        return {}

    return {
        "query.framework_inflight_candidates": _inflight,
        "query.framework_next_candidate": _next,
        "framework.submit_candidate": _submit,
        "query.framework_authoring_inflight": _authoring,
        "framework.discover": _discover,
        "query.source_arm_plateau": _source_dry,
        "framework.local_explore": _local_explore,
        "framework.phase_done": _phase_done,
    }


async def _dispatch_local_explore(phase: "FrameworkPhase", *, reason: str) -> bool:
    gap, keywords = phase._compose_framework_local_explore_gap()
    title = (
        f"local source exploration ({gap})"
        if gap
        else "local source exploration (author a throughput patch from live source + profile)"
    )
    return await phase._enqueue_framework_agent_local_explore_specialist(
        {
            "title": title,
            "repo": "(local source)",
            "framework": str(phase.shared_state.framework or "").strip().lower(),
            "gap_description": gap,
            "gap_keywords": keywords,
        },
        reason=reason,
    )


async def discovery_decision(phase: "FrameworkPhase", *, empties: int, failures: int) -> str | None:
    """``"discovery.dispatch"`` or ``"discovery.wait"`` as the brain decides; ``None`` once it declines."""

    async def _inflight(_args: Mapping[str, Any]) -> dict[str, Any]:
        return {"inflight": bool(await phase._candidate_discovery_inflight())}

    request = {"empty_discoveries": empties, "discover_failures": failures}
    queries = {"query.candidate_discovery_inflight": _inflight}
    turn = await phase._ask_brain(CANDIDATE_DISCOVERY_KEY, request, queries, call_site=CallSite.PHASE_PUMP)
    decision = one_decision(turn, DISCOVERY_DECISIONS, what="candidate discovery")
    return None if decision is None else decision.name


async def run_auto_bench(phase: "FrameworkPhase") -> None:
    """Let the brain decide which untested proposals to bench through :func:`auto_bench_steps`."""
    state = phase.shared_state
    request = {
        "phase": state.phase or "",
        "custom_specialist_only": bool(state.custom_specialist_only),
        "phase_done": bool(state.framework_agent_phase_done),
    }
    turn = await phase._ask_brain(AUTO_BENCH_KEY, request, auto_bench_steps(phase), call_site=CallSite.PHASE_PUMP)
    one_decision(turn, (), what="the untested-proposal bench")


def auto_bench_steps(phase: "FrameworkPhase") -> dict[str, QueryAnswer]:
    """The bench's queries and its enqueue; ``rows`` names the row set the last query of its kind read."""
    state = phase.shared_state
    rows: dict[str, list[dict[str, Any]]] = {}

    async def _admission(_args: Mapping[str, Any]) -> dict[str, Any]:
        dispatcher = phase._coord.dispatcher
        frozen = bool(dispatcher.admission_frozen)
        return {"frozen": frozen, "paused": not frozen and bool(dispatcher.dispatch_paused_for_phase_budget())}

    async def _queue(_args: Mapping[str, Any]) -> dict[str, Any]:
        explores = [t for t in (*await phase.tasks.queued(), *await phase.tasks.running()) if t.kind == "explore"]
        rows["custom"] = custom = phase._custom_proposal_rows()
        operator_work = bool(custom) and bool(phase._has_operator_work(explores))
        return {"explores": len(explores), "custom_rows": len(custom), "operator_work": operator_work}

    async def _untested(_args: Mapping[str, Any]) -> dict[str, Any]:
        rows["untested"] = [r for r in state.untested_proposal_rows() if r["domain"] != CUSTOM_PROPOSAL_DOMAIN]
        return {"count": len(rows["untested"])}

    async def _enqueue(args: Mapping[str, Any]) -> dict[str, Any]:
        name, limit = str(args["rows"]), args.get("limit")
        chosen = rows[name] if limit is None else rows[name][: int(limit)]
        return {"created": bool(await phase._enqueue_bench_grid(chosen, operator_priority=_OPERATOR_PRIORITY[name]))}

    return {
        "query.dispatch_admission": _admission,
        "query.bench_queue": _queue,
        "query.untested_rows": _untested,
        "bench.enqueue": _enqueue,
    }

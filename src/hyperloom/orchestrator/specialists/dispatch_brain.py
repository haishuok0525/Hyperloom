# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Body side of forced stalled-domain, custom specialist, and wave-split brain asks."""

from __future__ import annotations

import logging as _logging

from typing import TYPE_CHECKING, Any, Mapping

from hyperloom.brain_api import CallSite
from hyperloom.orchestrator.brain_link import DecisionPort, QueryAnswer, ask_brain, one_decision
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType

log = _logging.getLogger(__name__)

if TYPE_CHECKING:
    from .dispatch import SpecialistDispatchCollaborator

STALLED_DOMAIN_KEY = "stalled_domain"
CUSTOM_SPECIALIST_KEY = "custom_specialist"
WAVE_SPLIT_KEY = "specialist_wave"
_SCOPE_DOMAIN = "domain"
_SOURCE_INTERNAL = "coordinator_internal"


async def _ask(state: Any, key: str, request: Mapping[str, Any], queries: Mapping[str, QueryAnswer] | None = None):
    port = DecisionPort(f"a {key} decision", queries)
    return await ask_brain(
        CallSite.PHASE_PUMP,
        {key: request},
        port=port,
        tick=int(state.tick or 0),
        phase=state.phase or "",
    )


def _idempotency_query(dispatch: "SpecialistDispatchCollaborator") -> QueryAnswer:
    async def _exists(args: Mapping[str, Any]) -> dict[str, Any]:
        return {"exists": await dispatch.tasks.find_by_idempotency_key(str(args["key"])) is not None}

    return _exists


async def run_stalled_domain(dispatch: "SpecialistDispatchCollaborator") -> None:
    """Ask the brain whether to force one stalled-domain specialist this tick."""
    from .domains import domain_for_tag
    from .in_flight import in_flight_domains
    from .dispatch_policy import FORCE_STALLED_KEEP_ROUNDS, FORCE_STALLED_SPECIALIST_ROUNDS

    state = dispatch.shared_state
    stalled = []
    if str(getattr(state, "phase", "") or "").upper() == "FRAMEWORK_AGENT":
        anchors = state.stalled_domains(
            specialist_threshold=FORCE_STALLED_SPECIALIST_ROUNDS,
            keep_threshold=FORCE_STALLED_KEEP_ROUNDS,
        )
        busy = in_flight_domains([*await dispatch.tasks.queued(), *await dispatch.tasks.running()])
        round_id = int((state.explore_search or {}).get("cursor") or 0)
        suffix = dispatch._coord.dispatcher.cycle_idem_suffix()
        for anchor in anchors:
            gap_cid = state.best_gap_for_anchor(anchor)
            dom = domain_for_tag(anchor)
            stalled.append(
                {
                    "anchor": anchor,
                    "domain": None if dom is None else dom.key,
                    "tags": [anchor],
                    "gap_canonical_id": gap_cid or "",
                    "busy": dom is not None and dom.key in busy,
                    "idempotency_key": f"forced-stalled-{anchor}-round{round_id}{suffix}",
                    "reason": f"stalled_domain_force:{anchor}",
                }
            )
    request = {"phase": state.phase or "", "candidates": stalled}

    async def _preflight(args: Mapping[str, Any]) -> dict[str, Any]:
        params = {
            "domain": args["domain"],
            "tags": list(args["tags"]),
            "gap_canonical_id": args["gap_canonical_id"],
            "scope": _SCOPE_DOMAIN,
            "source": _SOURCE_INTERNAL,
            "reason": f"stalled_domain_force:{args['tags'][0]}" if args.get("tags") else "stalled_domain_force",
        }
        await dispatch.warm_specialist_params(params)
        refuse = dispatch.refuse_patch_without_source(params, task_id=str(args["idempotency_key"]))
        # Stash warmed params for the decision that follows this query.
        _preflight.last = params  # type: ignore[attr-defined]
        return {"refuse": bool(refuse)}

    _preflight.last = {}  # type: ignore[attr-defined]
    turn = await _ask(
        state,
        STALLED_DOMAIN_KEY,
        request,
        {"query.task_idempotency": _idempotency_query(dispatch), "query.patch_preflight": _preflight},
    )
    decision = one_decision(turn, ("specialist.force_stalled",), what="a stalled-domain force")
    if decision is not None:
        await _execute_forced_stalled(dispatch, state, decision.args, dict(_preflight.last))  # type: ignore[attr-defined]


async def run_custom_specialist(dispatch: "SpecialistDispatchCollaborator") -> None:
    """Ask the brain whether to dispatch the operator-defined specialist this cycle."""
    from .domains import CUSTOM_SPECIALIST_KEY as DOMAIN_KEY, custom_specialist_key
    from ..state._shared_state.phase_state import gap_actionability_key
    from .dispatch_policy import CUSTOM_SESSION_GAP_ID
    from .runner import specialist_patch_preflight_error

    state = dispatch.shared_state
    prompt = str(state.custom_specialist_prompt or "").strip()
    key = custom_specialist_key(int(state.macro_cycle or 0), prompt) if prompt else ""
    request = {
        "phase": state.phase or "",
        "prompt": prompt,
        "key": key,
        "refused_key": dispatch.refused_custom_key,
        "custom_only": bool(state.custom_specialist_only),
    }

    async def _gap(_args: Mapping[str, Any]) -> dict[str, Any]:
        gaps = [g for g in state.gaps if isinstance(g, dict) and str(g.get("canonical_id") or "").strip()]
        out: dict[str, Any] = {}
        if gaps:
            out["gap_canonical_id"] = str(min(gaps, key=gap_actionability_key)["canonical_id"]).strip()
        else:
            out["gap_canonical_id"] = CUSTOM_SESSION_GAP_ID
            out["gap_symptom"] = "No profiled gap; follow the operator-defined focus."
            out["gap_layer"] = "operator-defined"
        params = {
            "domain": DOMAIN_KEY,
            "scope": _SCOPE_DOMAIN,
            "source": _SOURCE_INTERNAL,
            "reason": "custom_specialist",
            "gap_canonical_id": out["gap_canonical_id"],
            **({k: out[k] for k in ("gap_symptom", "gap_layer") if k in out}),
        }
        await dispatch.warm_specialist_params(params)
        research = specialist_patch_preflight_error(params, framework_repo_path=state.framework_repo_path)
        if research:
            out["research_reason"] = research
        _gap.last = params  # type: ignore[attr-defined]
        return out

    _gap.last = {}  # type: ignore[attr-defined]
    turn = await _ask(
        state,
        CUSTOM_SPECIALIST_KEY,
        request,
        {"query.task_idempotency": _idempotency_query(dispatch), "query.custom_specialist_gap": _gap},
    )
    decision = one_decision(turn, ("specialist.ensure_custom",), what="a custom specialist")
    if decision is not None:
        await _execute_custom(dispatch, decision.args, dict(_gap.last))  # type: ignore[attr-defined]


async def wave_split_decision(
    dispatch: "SpecialistDispatchCollaborator",
    intent: Intent,
    params: dict[str, Any],
) -> Mapping[str, Any] | None:
    """Ask the brain how to fan a wave; returns the ``specialist.wave_split`` args, or ``None``."""
    shared = {k: v for k, v in params.items() if k != "tasks"}
    request = {
        "base_key": str(intent.payload.get("idempotency_key") or "").strip(),
        "shared": shared,
        "tasks": list(params.get("tasks") or []),
    }
    turn = await _ask(dispatch.shared_state, WAVE_SPLIT_KEY, request)
    decision = one_decision(turn, ("specialist.wave_split",), what="a specialist wave")
    return None if decision is None else dict(decision.args)


async def _execute_forced_stalled(
    dispatch: "SpecialistDispatchCollaborator", state: Any, args: Mapping[str, Any], warmed: dict[str, Any]
) -> None:
    params = dict(warmed)
    if not params:
        params = {
            "domain": args["domain"],
            "tags": list(args["tags"]),
            "gap_canonical_id": args["gap_canonical_id"],
            "scope": _SCOPE_DOMAIN,
            "source": _SOURCE_INTERNAL,
            "reason": args["reason"],
        }
        await dispatch.warm_specialist_params(params)
    else:
        params["reason"] = args["reason"]
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "specialist", "params": params, "idempotency_key": args["idempotency_key"]},
    )
    await dispatch._coord.router.handle_intent("orchestration", intent)
    try:
        state.save(dispatch.session_dir)
    except OSError:
        log.exception("stalled-domain force: state save failed")
    log.info(
        "stalled-domain force: dispatched domain=%s anchor=%s gap=%s round=%d",
        args["domain"],
        (args["tags"] or [""])[0],
        args["gap_canonical_id"],
        int((state.explore_search or {}).get("cursor") or 0),
    )


async def _execute_custom(
    dispatch: "SpecialistDispatchCollaborator", args: Mapping[str, Any], warmed: dict[str, Any]
) -> None:
    from .profile import MODE_RESEARCH, NO_LEASE_EXTENSION

    params = dict(warmed)
    if args["no_lease_extension"]:
        params[NO_LEASE_EXTENSION] = True
    if args["mode_research"]:
        params["mode"] = MODE_RESEARCH
        await dispatch.bus.record_observation(
            "coordinator",
            "observation",
            {"kind": "custom_specialist_research_only", "reason": args["research_reason"]},
        )
    for field in ("gap_symptom", "gap_layer"):
        if args.get(field) and field not in params:
            params[field] = args[field]
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={"action_name": "specialist", "params": params, "idempotency_key": args["key"]},
    )
    await dispatch._coord.router.handle_delegate("orchestration", intent)
    if await dispatch.tasks.find_by_idempotency_key(args["key"]) is None:
        dispatch.refused_custom_key = args["key"]
        log.warning(
            "custom specialist: dispatch %s refused (see the policy denial); not retried this cycle",
            args["key"],
        )
        return
    log.info(
        "custom specialist: requested key=%s gap=%s mode=%s",
        args["key"],
        params["gap_canonical_id"],
        params.get("mode") or "default",
    )

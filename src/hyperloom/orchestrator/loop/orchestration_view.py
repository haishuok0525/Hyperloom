# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the body reports for the brain to compose the orchestration prompt from.

Each section's facts are read here, in the order the prompt has always read them: the state summaries the body renders
for its own tools too, the raw facts behind the advisories the brain renders, the Experience KB read the queue triggers,
the trajectory digest a plateau pays for, and the inbox tail as the lines the inbox tools render.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..phases import machine_state as _phase_state
from ..policy.projection import resource_pools_summary

if TYPE_CHECKING:
    from ..state.shared_state import SharedState
    from .conversation import ConversationCollaborator

ORCHESTRATION = "orchestration"
# Variant-failure rows each delegated result shows: only Orchestration acts on them.
_INBOX_VARIANT_ROWS = 3


def _time_budget(conv: "ConversationCollaborator") -> dict[str, Any] | None:
    deadline = conv._coord.run_deadline
    started = conv._coord.run_started_monotonic
    if deadline is None or started is None:
        return None
    state = conv.shared_state
    return {
        "elapsed_min": (time.monotonic() - started) / 60.0,
        "remaining_min": max(0.0, deadline.remaining() / 60.0),
        "budget_min": state.max_minutes or 0,
        "closing_phase": state.closing_phase,
    }


def _specialist_rounds(state: "SharedState") -> list[dict[str, Any]]:
    return [
        {
            "domain": row.get("domain"),
            "new_findings": row.get("new_findings") or [],
            "residual_questions": row.get("residual_questions") or [],
        }
        for row in state.specialist_rounds or []
        if isinstance(row, dict)
    ]


def _bottleneck(conv: "ConversationCollaborator") -> dict[str, Any]:
    """The roofline and bottleneck facts behind the redirect advisory, which only the optimisation phase carries."""
    from hyperloom.inference_optimizer.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

    state = conv.shared_state
    direction, pct = conv.dominant_roofline_direction()
    shift = state.bottleneck_shift
    return {
        "pending_switch": bool(state.pending_bottleneck_switch),
        "saturated": {
            domain: {key: row.get(key) for key in ("saturated", "direction", "within_pct", "threshold_pct")}
            for domain, row in state.saturated_directions.items()
        },
        "last_cycle_bottleneck": state.last_cycle_bottleneck,
        "current_top": state.current_top_bottleneck(),
        "shift": {key: shift.get(key) for key in ("from", "to", "within_delta", "gap_delta")},
        "dominant": {"direction": direction, "pct": pct, "hint": BOTTLENECK_DOMAIN_HINTS.get(direction)},
        "macro_cycle": state.macro_cycle,
    }


def _acceptance(state: "SharedState") -> dict[str, Any]:
    search = state.explore_search
    entries = [*(search.get("tested") or {}).values(), *(search.get("rejected") or [])]
    return {
        "keep_threshold": _phase_state.resolve_keep_threshold(state),
        "macro_cycle": state.macro_cycle,
        "entries": [{key: e.get(key) for key in ("name", "fingerprint", "gain_pct")} for e in entries],
    }


def _advisories(conv: "ConversationCollaborator", framework: bool) -> dict[str, Any]:
    state = conv.shared_state
    return {
        "advisories": {
            "warm_start": state.to_warm_start_summary(),
            "gaps": state.to_gaps_summary(),
            "target_gap": conv.target_gap_advisory_block(),
            "proposal_scores": state.to_proposal_scores_summary(),
            "priors_match": conv._priors_match_advisory_block(),
            "intervention_mix": state.to_intervention_mix_summary(),
        },
        "specialist_rounds": _specialist_rounds(state),
        "bottleneck": _bottleneck(conv) if framework else None,
        "acceptance": _acceptance(state),
        "discarded_escalate_hint": {
            "hint": str(state.last_discarded_escalate_hint or ""),
            "ts": str(state.last_discarded_escalate_hint_ts or ""),
        },
    }


async def _proposal_queue(conv: "ConversationCollaborator") -> dict[str, Any]:
    """The predictor's queue and the untested proposals, and the Experience KB evidence read against them."""
    from ..predictor.prompt import queue_block
    from ..predictor.rows import PROVENANCE

    predictor = queue_block(conv.shared_state, in_grids=await conv._explore_grid_fingerprints())
    untested = conv.shared_state.to_untested_proposals_summary(skip_provenance=PROVENANCE)
    kb_evidence = await conv._kb_prompt_block("\n\n".join(block for block in (predictor, untested) if block))
    return {"predictor": predictor, "untested": untested, "kb_evidence": kb_evidence}


async def _inbox(conv: "ConversationCollaborator") -> list[dict[str, Any]]:
    from .inbox_lines import _format_inbox_event

    cursor = await conv.cursors.load(ORCHESTRATION)
    unread = await conv.bus.replay_for(ORCHESTRATION, after_seq=cursor.last_processed_seq)
    return [{"seq": int(m.seq), "line": _format_inbox_event(m, max_variant_rows=_INBOX_VARIANT_ROWS)} for m in unread]


async def orchestration_view(conv: "ConversationCollaborator") -> dict[str, Any]:
    """Everything the orchestration prompt is composed from, read now."""
    state = conv.shared_state
    framework = (state.phase or "").strip().upper() == _phase_state.PHASE_FRAMEWORK_AGENT
    view: dict[str, Any] = {
        "session_dir": str(conv.session_dir),
        "phase": _phase_state.phase_status_summary(state),
        "mission": state.to_mission_summary(),
        "time_budget": _time_budget(conv),
        "tick": state.to_tick_summary(),
        "shared_state": state.to_prompt_summary(),
        "target_gap_pct": f"{conv._coord.target_gap_pct():.2f}",
        "constraints": list(state.to_constraint_sections()),
        "resource_pools": resource_pools_summary(state),
    }
    view.update(_advisories(conv, framework))
    view["denial_summary"] = state.to_policy_denial_summary(top_k=6)
    view["proposal_queue"] = await _proposal_queue(conv) if framework else None
    view["plateau"] = conv.plateau_advisory_block()
    # Candidate directions are worth the digest's cost only once the session has stopped gaining.
    view["trajectory_digest"] = conv.trajectory_digest() if view["plateau"] else ""
    view["inbox"] = await _inbox(conv)
    return view

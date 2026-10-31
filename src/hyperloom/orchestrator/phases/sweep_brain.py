# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Body side of SWEEP directive and conc_sweep brain asks."""

from __future__ import annotations

import logging as _logging
import sqlite3
from typing import TYPE_CHECKING, Any

from hyperloom.brain_api import CallSite

from ..brain_link import DecisionPort, ask_brain, one_decision
from ..kernel import kernel_decisions
from ..state.orchestration_memory import CYCLE_DIRECTIVE_REQUEST, build_cycle_memory
from . import machine_state as _phase_state

if TYPE_CHECKING:
    from .machine import Transition
    from .sweep import SweepPhase

log = _logging.getLogger(__name__)

SWEEP_DIRECTIVE_KEY = "sweep_directive"
CONC_SWEEP_KEY = "conc_sweep"
_SKIP_DISABLED = "disabled"
_SKIP_NO_GAIN = "no_validated_gain_since_last_conc_sweep"


async def run_sweep_directive(phase: "SweepPhase") -> None:
    """Ask the brain whether to request a cycle directive this macro-cycle."""
    state = phase.shared_state
    feasible, _ = _phase_state.cycle_reloop_decision(state)
    request = {
        "has_orchestration": "orchestration" in phase.backends,
        "user_supplied_prompt": bool(phase.orch_prompt.is_user_supplied),
        "memory_for_cycle": state.orchestration_memory.get("for_cycle"),
        "macro_cycle": state.macro_cycle,
        "reloop_feasible": bool(feasible),
    }
    turn = await ask_brain(
        CallSite.PHASE_PUMP,
        {SWEEP_DIRECTIVE_KEY: request},
        port=DecisionPort("a SWEEP cycle directive"),
        tick=int(state.tick or 0),
        phase=state.phase or "",
    )
    decision = one_decision(turn, ("sweep.ask_directive",), what="a SWEEP cycle directive")
    if decision is None:
        return
    replies: list[Any] = []

    async def _directive_turn() -> None:
        replies.append(await phase._coord.reactor_pass("orchestration", request=CYCLE_DIRECTIVE_REQUEST))

    await phase._coord.await_within_session_bound(_directive_turn, stage="reactor:orchestration")
    state.orchestration_memory = build_cycle_memory(replies[0] if replies else None, cycle=state.macro_cycle)
    log.info(
        "cycle %d handoff: directive=%r parse_error=%r",
        state.macro_cycle,
        state.orchestration_memory["next_cycle_directive"][:80],
        state.orchestration_memory["parse_error"],
    )


async def run_conc_sweep(phase: "SweepPhase", tr: "Transition") -> None:
    """Recover the stack, then ask the brain whether to auto-enqueue conc_sweep."""
    from_phase = tr.from_phase
    state = phase.shared_state
    await phase._coord.phase_kernel_stack.recover_interrupted_stack_validation()
    if kernel_decisions.has_keep_pending_integrate(state):
        await phase._coord.phase_kernel_stack.drain_pending_keep_integrates()
    await phase._coord.phase_kernel_stack.maybe_validate_positive_needs_review_stack()
    denied = phase._coord.dispatcher.time_budget_denial_for_action("conc_sweep")
    request = {
        "enabled": bool(state.conc_sweep_enabled),
        "watermark": state.last_conc_sweep_watermark if isinstance(state.last_conc_sweep_watermark, dict) else {},
        "validated_gain": float(state.cumulative_gain_validated or 0.0),
        "budget_denied": None if denied is None else str(denied),
    }
    turn = await ask_brain(
        CallSite.PHASE_PUMP,
        {CONC_SWEEP_KEY: request},
        port=DecisionPort("a conc_sweep entry"),
        tick=int(state.tick or 0),
        phase=state.phase or "",
    )
    decision = one_decision(turn, ("sweep.skip_conc", "sweep.enqueue_conc"), what="a conc_sweep entry")
    if decision is None:
        return
    if decision.name == "sweep.skip_conc":
        _apply_conc_skip(phase, from_phase, decision.args, request.get("watermark") or {}, denied)
        return
    await _enqueue_conc(phase, from_phase, str(decision.args["reason"]))


def _apply_conc_skip(
    phase: "SweepPhase",
    from_phase: str | None,
    args: dict[str, Any],
    watermark: Any,
    denied: Any,
) -> None:
    reason = str(args["reason"])
    if reason == _SKIP_DISABLED:
        log.info("SWEEP entry (from=%s): conc_sweep disabled; recording terminal skip.", from_phase or "<unknown>")
        phase._record_terminal_conc_sweep_skip(skip_reason=_SKIP_DISABLED, auto_conc_sweep_skipped=_SKIP_DISABLED)
        return
    if reason == _SKIP_NO_GAIN:
        cur = float(args.get("validated_gain") or 0.0)
        log.info(
            "SWEEP entry (from=%s): skipping auto-conc-sweep — no validated gain since last "
            "conc_sweep (validated=%.4f%% unchanged since %s).",
            from_phase or "<unknown>",
            cur,
            (watermark.get("ts") if isinstance(watermark, dict) else None) or "(unknown)",
        )
        phase._record_terminal_conc_sweep_skip(
            skip_reason=_SKIP_NO_GAIN,
            auto_conc_sweep_skipped=_SKIP_NO_GAIN,
            auto_conc_sweep_skipped_validated_gain=cur,
        )
        return
    log.info(
        "SWEEP entry (from=%s): conc_sweep cannot fit the session budget (%s); recording terminal skip.",
        from_phase or "<unknown>",
        args.get("denied"),
    )
    phase.record_session_budget_conc_sweep_skip(denied=args.get("denied") or denied)


async def _enqueue_conc(phase: "SweepPhase", from_phase: str | None, reason: str) -> None:
    try:
        task = await phase._enqueue_internal_conc_sweep_task(reason=reason)
    except (sqlite3.Error, TypeError, ValueError) as exc:
        log.exception("SWEEP entry hook: failed to enqueue auto-conc-sweep: %r", exc)
        phase._record_terminal_conc_sweep("failed", "enqueue_failed", auto_conc_sweep_error=repr(exc)[:240])
        return
    if task is None:
        return
    log.info(
        "SWEEP entry (from=%s): auto-enqueued conc_sweep task=%s (concs=%s total_budget_sec=%s)",
        from_phase or "<unknown>",
        task.task_id,
        task.params.get("concs"),
        task.params.get("total_budget_sec"),
    )
    phase._coord.phase_machine.record_phase_entry_evidence(
        auto_conc_sweep_enqueued=True,
        auto_conc_sweep_task_id=task.task_id,
        auto_conc_sweep_concs=task.params.get("concs"),
    )

# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Body side of KERNEL enter and agent-route brain asks."""

from __future__ import annotations

import logging as _logging
from typing import TYPE_CHECKING, Any

from hyperloom.brain_api import CallSite
from hyperloom.common.env import env_bool

from ..brain_link import DecisionPort, ask_brain, one_decision
from ..state.task_registry import TERMINAL_STATES, TaskSpec

if TYPE_CHECKING:
    from .kernel import KernelPhase
    from .machine import Transition

log = _logging.getLogger(__name__)

KERNEL_ENTER_KEY = "kernel_enter"
_KIND_KERNEL_AGENT = "kernel_agent"
_ROUTE_GEAK = "geak"
KERNEL_AGENT_KEY = "kernel_agent"


async def run_kernel_enter(phase: "KernelPhase", tr: "Transition") -> None:
    """Ask the brain whether to open the KERNEL timeline and enqueue the carrier task."""
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import ROUTE_FORGE, ROUTE_GEAK

    state = phase.shared_state
    request = {
        "enabled": bool(phase._coord.phase_machine.kernel_enabled()),
        "geak": bool(phase.geak_enabled()),
        "route_reason": f"kernel_optimizer={str(state.kernel_optimizer or '')}",
        "from_phase": tr.from_phase or "",
        "macro_cycle": int(state.macro_cycle or 0),
    }
    turn = await ask_brain(
        CallSite.PHASE_PUMP,
        {KERNEL_ENTER_KEY: request},
        port=DecisionPort("a KERNEL enter"),
        tick=int(state.tick or 0),
        phase=state.phase or "",
    )
    decision = one_decision(turn, ("kernel.enter",), what="a KERNEL enter")
    if decision is None:
        log.info("KERNEL entry hook fired with kernel_enabled=False (from=%s)", tr.from_phase or "<unknown>")
        return
    route = ROUTE_GEAK if decision.args["route"] == _ROUTE_GEAK else ROUTE_FORGE
    phase._open_kernel_timeline(
        route=route,
        route_reason=str(decision.args["route_reason"]),
        from_phase=tr.from_phase,
    )
    lanes, lease_ttl = phase._coord.dispatcher.registry_lanes_ttl(_KIND_KERNEL_AGENT)
    from . import machine_state as _phase_state

    remaining = _phase_state.phase_budget_remaining_seconds(state)
    ttl = int(remaining) if remaining is not None and remaining > 0 else lease_ttl
    base_key = f"{_KIND_KERNEL_AGENT}_c{int(decision.args['macro_cycle'] or 0)}"
    attempt = 0
    while True:
        task, was_existing = await phase.tasks.create_or_return_existing(
            TaskSpec(
                kind=_KIND_KERNEL_AGENT,
                params={"from_phase": str(tr.from_phase or "")},
                idempotency_key=base_key if attempt == 0 else f"{base_key}-r{attempt}",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        if not (was_existing and task.state in TERMINAL_STATES):
            break
        attempt += 1
    log.info("KERNEL entry: kernel_agent task=%s (%s)", task.task_id, task.state)


async def run_kernel_agent(phase: "KernelPhase", ctx: Any) -> dict[str, Any]:
    """Ask the brain which KERNEL route to run, then execute it."""
    state = phase.shared_state
    from_phase = str((ctx.task.params or {}).get("from_phase") or "")
    last = state.last_gemm_tuning or {}
    request = {
        "geak": bool(phase.geak_enabled()),
        "skip_gemm": bool(env_bool("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING")),
        "framework": str(state.framework or ""),
        "last_gemm_status": str(last.get("status") or ""),
        "from_phase": from_phase,
    }
    turn = await ask_brain(
        CallSite.PHASE_PUMP,
        {KERNEL_AGENT_KEY: request},
        port=DecisionPort("a KERNEL agent run"),
        tick=int(state.tick or 0),
        phase=state.phase or "",
    )
    decision = one_decision(turn, ("kernel.run",), what="a KERNEL agent run")
    assert decision is not None
    route = str(decision.args["route"])
    if route == _ROUTE_GEAK:
        await phase._run_geak_kernel_phase(from_phase=from_phase)
        return {"status": "ok", "route": _ROUTE_GEAK}
    if route == "forge_no_gemm":
        await phase._finish_kernel_entry()
        return {"status": "ok", "route": "forge_no_gemm"}
    await _run_forge_gemm(phase, state)
    return {"status": "ok", "route": "forge_gemm"}


async def _run_forge_gemm(phase: "KernelPhase", state: Any) -> None:
    """Reprofile, run GEMM tuning, file the result, then finish the KERNEL entry."""
    from ..bus.message_bus import Message
    from ..kernel.request_handlers import run_gemm_tuning_handler

    await phase._maybe_reprofile_for_kernel()
    log.info("KERNEL entry: running GEMM tuning before source-level kernel_opt")
    phase._coord.phase_machine.record_phase_entry_evidence(
        gemm_tuning={"status": "running", "source": "kernel_entry_auto"},
    )
    try:
        result = await run_gemm_tuning_handler(
            {
                "task_id": "kernel_entry_gemm_tuning",
                "reason": "kernel_entry_auto",
                "macro_cycle": int(state.macro_cycle or 0),
            },
            session_dir=phase.session_dir,
        )
    except Exception as exc:  # isolation boundary: the kernel-entry GEMM tuning stage, recorded as failed
        log.exception("KERNEL entry GEMM tuning failed")
        result = {
            "status": "failed",
            "decision": "REVERT",
            "error_class": exc.__class__.__name__,
            "error": repr(exc),
        }
    await phase._handle_gemm_tuning_result(result)
    status = str(result.get("status") or "unknown")
    await phase.bus.append_and_seq(
        Message.new(
            _KIND_KERNEL_AGENT,
            "orchestration",
            "response",
            {
                "in_reply_to": "",
                "kind": "run_gemm_tuning_done",
                "status": status,
                "result": result,
                "source": "kernel_entry_auto",
            },
        )
    )
    phase._coord.phase_machine.record_phase_entry_evidence(
        gemm_tuning={
            "status": "done" if status in {"ok", "complete", "succeeded"} else status,
            "source": "kernel_entry_auto",
            "best_speedup": result.get("best_speedup"),
            "tuned_file": result.get("tuned_file"),
        },
    )
    await phase._finish_kernel_entry()


_GEMM_ELIGIBLE = frozenset({"sglang", "vllm", "vllm-aiter"})
_GEMM_TERMINAL = frozenset({"ok", "succeeded", "success", "complete", "completed", "skipped", "failed"})


def gemm_required_for_phase(phase: "KernelPhase") -> bool:
    """Whether GEMM tuning must run before kernel_opt (K-02; keeps the method name for tests)."""
    if env_bool("INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING"):
        return False
    framework = str(phase.shared_state.framework or "").strip().lower()
    if framework not in _GEMM_ELIGIBLE:
        return False
    last = phase.shared_state.last_gemm_tuning or {}
    return str(last.get("status") or "").strip().lower() not in _GEMM_TERMINAL

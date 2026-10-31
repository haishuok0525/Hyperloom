# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The order the dispatcher visits queued tasks in: operator work, then the kinds the phase dispatches first, then the rest."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hyperloom.brain_api import CallSite

from ..brain_link import DecisionPort, ask_brain, one_decision
from ..specialists.domains import CUSTOM_SPECIALIST_KEY
from .machine_state import PHASE_ENABLEMENT, PHASE_FRAMEWORK_AGENT

if TYPE_CHECKING:
    from ..state.task_registry import Task

__all__ = ["PHASE_DISPATCH_FIRST", "in_dispatch_order", "in_dispatch_order_async", "is_operator_work"]

# In FRAMEWORK_AGENT a reviewed patch and a configuration grid hold the same serving lanes for a whole round: a grid
# can be proposed again, but a patch still queued when the phase exits is cancelled unmeasured.
# ENABLEMENT holds its exit for the initial roofline its promoted baseline enqueued, so a roofline queued behind
# integrates would keep the phase open for as long as they take.
# The brain applies the same map; this copy remains for prompt/backlog readers that only need the kind set.
PHASE_DISPATCH_FIRST: dict[str, frozenset[str]] = {
    PHASE_ENABLEMENT: frozenset({"roofline"}),
    PHASE_FRAMEWORK_AGENT: frozenset({"integrate_patch"}),
}

_DISPATCH_ORDER_KEY = "dispatch_order"


def is_operator_work(kind: str, params: dict[str, Any] | None) -> bool:
    """Whether a task is the operator-defined specialist or the bench of its proposals."""
    params = params or {}
    if kind == "specialist":
        return str(params.get("domain") or "") == CUSTOM_SPECIALIST_KEY
    return kind == "explore" and bool(params.get("operator_priority"))


def in_dispatch_order(queued: list[Task], phase: str) -> list[Task]:
    """Return ``queued`` grouped as operator work, then ``phase``'s dispatch-first kinds, then the rest, each in order.

    Sync helper for tests and prompt lines; the live dispatcher asks the brain via :func:`in_dispatch_order_async`.
    """
    first = PHASE_DISPATCH_FIRST.get(phase, frozenset())

    def tier(task: Task) -> int:
        if is_operator_work(task.kind, task.params):
            return 0
        return 1 if task.kind in first else 2

    return sorted(queued, key=tier)


async def in_dispatch_order_async(queued: list[Task], phase: str, tick: int = 0) -> list[Task]:
    """Ask the brain for the dispatch order of ``queued`` in ``phase``."""
    by_id = {str(task.task_id): task for task in queued}
    request = {
        "phase": phase or "",
        "tasks": [
            {
                "id": str(task.task_id),
                "kind": task.kind,
                "operator_work": is_operator_work(task.kind, task.params),
            }
            for task in queued
        ],
    }
    turn = await ask_brain(
        CallSite.PHASE_PUMP,
        {_DISPATCH_ORDER_KEY: request},
        port=DecisionPort("a dispatch order"),
        tick=tick,
        phase=phase or "",
    )
    decision = one_decision(turn, ("dispatch.order",), what="a dispatch order")
    if decision is None:
        return list(queued)
    ordered = [by_id[i] for i in decision.args["ids"] if i in by_id]
    seen = {t.task_id for t in ordered}
    ordered.extend(task for task in queued if task.task_id not in seen)
    return ordered

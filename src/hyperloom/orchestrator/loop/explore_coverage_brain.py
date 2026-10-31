# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Body side of the explore-grid coverage brain ask (R-01)."""

from __future__ import annotations

from typing import Any

from hyperloom.brain_api import CallSite
from hyperloom.orchestrator.brain_link import DecisionPort, ask_brain, one_decision
from hyperloom.orchestrator.policy.gate import PolicyDenied

EXPLORE_COVERAGE_KEY = "explore_coverage"


async def refuse_if_explore_covered(
    *,
    action_name: str,
    params: dict[str, Any],
    idempotency_key: str,
    tasks: Any,
    shared_state: Any,
) -> PolicyDenied | None:
    """Return a denial when the brain refuses a covered explore grid; ``None`` to proceed."""
    from ..actions.executors.explore import framework_lever_grid
    from ..actions.executors.measured_variants import explore_grid_already_covered

    retry_prefix = f"{idempotency_key}-retry"
    active = [
        task
        for task in (*await tasks.queued(), *await tasks.running())
        if task.idempotency_key != idempotency_key and not task.idempotency_key.startswith(retry_prefix)
    ]
    covered = (
        None
        if action_name != "explore"
        else explore_grid_already_covered(params, active, shared_state.explore_search)
    )
    request = {
        "is_explore": action_name == "explore",
        "lever_grid": bool(framework_lever_grid(shared_state)),
        "covered": covered is not None,
        "message": "" if covered is None else str(covered),
        "hint": "" if covered is None else str(getattr(covered, "hint", "") or ""),
    }
    turn = await ask_brain(
        CallSite.PHASE_PUMP,
        {EXPLORE_COVERAGE_KEY: request},
        port=DecisionPort("an explore coverage check"),
        tick=int(shared_state.tick or 0),
        phase=shared_state.phase or "",
    )
    decision = one_decision(turn, ("explore.refuse_covered",), what="an explore coverage check")
    if decision is None:
        return None
    return covered or PolicyDenied(
        str(decision.args["message"]),
        rule="explore_grid_already_covered",
        hint=str(decision.args.get("hint") or ""),
    )

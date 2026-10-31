# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Macro-cycle engine: cycle-focus planning, reloop, soft-restart, and the cycle-start reprofile."""

from __future__ import annotations

import logging as _logging
from typing import Any

from hyperloom.brain_api import CallSite
from hyperloom.common.env import env_bool
from .focus_plan import plan_cycle_focus as score_cycle_focus

from ..actions.executors.explore import TS_SKIPPED_DEDUP
from ..brain_link import DecisionPort, ask_brain, one_decision
from ..collaborator import CoordinatorCollaborator
from ..loop.maintenance import run_lease_and_db_reclaim

log = _logging.getLogger(__name__)

__all__ = ["MacroCycleCollaborator"]

# Snapshot key the brain decides the cycle-start reprofile under (must match brain.pump.macro).
_CYCLE_REPROFILE_KEY = "cycle_reprofile"


class MacroCycleCollaborator(CoordinatorCollaborator):
    """Macro-cycle planning, focus scoring, soft-restart, reprofile, and orchestration-memory helpers."""

    def __init__(self, coordinator) -> None:
        super().__init__(coordinator)
        # Medium-intensity soft restart at each macro-cycle boundary.
        self._soft_restart_enabled = not env_bool("INFERENCE_OPTIMIZER_DISABLE_CYCLE_SOFT_RESTART")

    def _negative_ledger_domain_counts(self, *, recent_cycles: int = 3) -> dict[str, int]:
        """Summarise recent negative explore-ledger pressure by specialist domain."""
        state = self.shared_state
        cur_cycle = int(state.macro_cycle or 0)
        search = state.explore_search or {}
        rows: list[Any] = []
        if isinstance(search, dict):
            tested = search.get("tested") or {}
            if isinstance(tested, dict):
                rows.extend(tested.values())
            rejected = search.get("rejected") or []
            if isinstance(rejected, list):
                rows.extend(rejected)
        counts: dict[str, int] = {}
        for row in rows:
            # A row the round answered without measuring rejects nothing.
            if not isinstance(row, dict) or row.get("outcome") == TS_SKIPPED_DEDUP:
                continue
            try:
                cycle = int(row.get("cycle", cur_cycle) or 0)
            except (TypeError, ValueError):
                cycle = cur_cycle
            if cycle < max(0, cur_cycle - recent_cycles + 1):
                continue
            domain = str(
                row.get("domain")
                or row.get("specialist_domain")
                or row.get("source_domain")
                or row.get("provenance")
                or ""
            ).strip()
            if not domain:
                continue
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    def plan_cycle_focus(self) -> dict[str, Any]:
        """Pick an advisory specialist-domain focus for the current macro-cycle."""
        from hyperloom.inference_optimizer.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

        state = self.shared_state
        return score_cycle_focus(
            {
                "domains": sorted({v[0] for v in BOTTLENECK_DOMAIN_HINTS.values()} | {"freeform_specialist"}),
                "shift": state.bottleneck_shift or {},
                "saturated": state.saturated_directions or {},
                "log_rows": list(state.cycle_strategy_log or []),
                "negative_counts": self._negative_ledger_domain_counts(),
                "macro_cycle": int(state.macro_cycle or 0),
                "bottleneck_fallback": self.shared_state.current_top_bottleneck() or "",
                "gain_at_start": float(state.gain_at_cycle_start or 0.0),
            }
        )

    def record_cycle_strategy_for_current_cycle(self) -> None:
        """Append/update the advisory cycle-strategy row for the current cycle."""
        state = self.shared_state
        planned = self.plan_cycle_focus()
        log_rows = [r for r in (state.cycle_strategy_log or []) if isinstance(r, dict)]
        cycle = int(planned.get("cycle", 0) or 0)
        replaced = False
        for idx, row in enumerate(log_rows):
            if int(row.get("cycle", -1) or -1) == cycle:
                merged = dict(row)
                merged.update(planned)
                log_rows[idx] = merged
                replaced = True
                break
        if not replaced:
            log_rows.append(planned)
        state.cycle_strategy_log = log_rows[-50:]

    async def run_cycle_soft_restart(
        self,
        *,
        prior_cycle: int,
        new_cycle: int,
    ) -> dict[str, Any] | None:
        """Medium-intensity soft restart at a macro-cycle boundary.

        Recycles transient/per-cycle resources (fresh leases, pruned DB, cleared
        caches, re-scoped system prompt) without losing accumulated optimization state;
        ``current_best`` / ``optimization_stack`` / ``explore_search`` are
        preserved. Idempotent.

        Args:
            prior_cycle: The macro-cycle number that just finished.
            new_cycle: The macro-cycle number being entered.

        Returns:
            A summary dict of the restart steps performed, or ``None`` when the
            soft restart is disabled.
        """
        if not self._soft_restart_enabled:
            return None
        summary: dict[str, Any] = {
            "prior_cycle": int(prior_cycle),
            "new_cycle": int(new_cycle),
        }
        # Reap leases, reclaim orphaned running tasks, prune DB.
        await run_lease_and_db_reclaim(self._coord, summary, reason="cycle_soft_restart")
        log.info(
            "cycle soft-restart %d → %d: %s",
            int(prior_cycle),
            int(new_cycle),
            summary,
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {"kind": "cycle_soft_restart", **summary},
        )
        return summary

    async def on_cycle_start_reprofile(self, *, from_phase: str) -> None:
        """Force a fresh analysis at the start of a reopened macro-cycle.

        Reached on every cycle start, whichever phase the reloop targets, so a
        new cycle never re-targets the bottleneck the *previous* cycle measured.

        Args:
            from_phase: The phase being left; only a SWEEP origin starts a cycle.
        """
        state = self.shared_state
        request = {"from_phase": from_phase or "", "macro_cycle": int(state.macro_cycle or 0)}
        port = DecisionPort("a cycle-start reprofile")
        turn = await ask_brain(
            CallSite.PHASE_PUMP,
            {_CYCLE_REPROFILE_KEY: request},
            port=port,
            tick=int(state.tick or 0),
            phase=state.phase or "",
        )
        decision = one_decision(turn, ("prelude.reprofile",), what="a cycle-start reprofile")
        if decision is None:
            return
        task = await self._coord.phase_prelude.enqueue_internal_analysis_task(reason=str(decision.args["reason"]))
        if task is None:
            return
        state.auto_roofline_pending_task_id = task.task_id
        log.info("cycle %d start: forced reprofile task=%s", int(state.macro_cycle or 0), task.task_id)

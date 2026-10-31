# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ask the predictor once per decision point and queue its answer.

FRAMEWORK_AGENT steps this at entry and on every tick, each time after its own
work at that site, so a predictor fault reaches the guard the site already has
without keeping the phase's work from running. The request runs in a worker
thread so the tick loop never waits on it; the step that finds it finished files
the answer into ``specialist_rounds``, where it joins the untested-proposal
queue the Coordinator benches from, beside the specialists'.

A decision point is ``c{macro_cycle}-s{stack_depth}-r{roofline_count}``. A KEEP,
a new macro-cycle and a new roofline each change what the answer is conditioned
on. A key counts as asked once its request goes out, so it is never asked twice
(``SessionView.predictor_asked_keys``), not even after a failure. An answer that
arrives after the macro-cycle moved on is filed under the cycle it was asked in,
where the queue no longer looks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from hyperloom.brain_api import BrainTurn, CallSite
from hyperloom.common.timeutil import now_iso
from hyperloom.common import predictor_config
from hyperloom.orchestrator.brain_link import DecisionPort, ask_brain, one_decision
from hyperloom.orchestrator.predictor.client import Prediction, predict
from hyperloom.orchestrator.predictor.mandate import patch_mandate
from hyperloom.orchestrator.predictor.payload import PhaseBudget, build_request
from hyperloom.orchestrator.predictor.rows import PROVENANCE, QUEUE_DOMAIN, QUEUE_PRIORITY, proposal_rows
from hyperloom.orchestrator.predictor.sidecars import load_sidecars
from hyperloom.orchestrator.predictor.source_sites import load_source_sites
from hyperloom.orchestrator.predictor.view import SessionView


log = logging.getLogger(__name__)

# Snapshot keys the brain decides the asks and the entry wait under.
PREDICTOR_ASK_KEY = "predictor_ask"
PREDICTOR_ENTRY_KEY = "predictor_entry"
_ASK_DECISIONS = ("predictor.ask", "predictor.skip")


def decision_point_key(state: SessionView) -> str:
    """``c{macro_cycle}-s{stack_depth}-r{roofline_snapshot_count}``."""
    return f"c{int(state.macro_cycle or 0)}-s{len(state.optimization_stack)}-r{len(state.roofline_snapshots)}"


async def _ask_brain(state: SessionView, key: str, request: Mapping[str, Any], port: DecisionPort) -> BrainTurn:
    tick, phase = int(getattr(state, "tick", 0) or 0), str(getattr(state, "phase", "") or "")
    return await ask_brain(CallSite.PHASE_PUMP, {key: request}, port=port, tick=tick, phase=phase)


class PredictorPump:
    """The request one session has in flight, if any.

    ``budget`` reads the phase clock when a request goes out; ``on_filed``
    receives each round as it lands in ``specialist_rounds``.
    """

    def __init__(
        self,
        *,
        budget: Callable[[SessionView], PhaseBudget],
        on_filed: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._budget = budget
        self._on_filed = on_filed
        self._inflight: asyncio.Task[Prediction] | None = None
        self._key = ""
        self._cycle = 0
        self._conf = predictor_config.PredictorConfig()
        self._started = 0.0
        self._asked_at = ""
        self._held_since: dict[str, float] = {}

    def _profile_hold(self, state: SessionView, key: str) -> dict[str, Any]:
        """The re-profile ``key`` waits for, if one is in flight, and how long it has been held for it."""
        pending = str(state.auto_roofline_pending_task_id or "").strip()
        if not pending:
            return {"pending": False, "held_sec": 0.0}
        if key not in self._held_since:
            self._held_since[key] = time.monotonic()
            log.info("predictor: holding decision point %s for re-profile task %s", key, pending)
        return {"pending": True, "held_sec": time.monotonic() - self._held_since[key]}

    async def settle(self, state: SessionView) -> None:
        """Ask when this decision point warrants it, then wait as long as the brain holds entry to file the answer.

        FRAMEWORK entry calls this before orchestration's first turn. The tick
        files an answer only after that turn, by which time the first grid is
        composed. Only active mode with an endpoint waits; off, shadow and an
        unset endpoint return at once. The wait yields to the event loop, and a
        request still running at the deadline is left for a later step to file.
        """
        await self.step(state)
        request = {
            "inflight": self._inflight is not None,
            "enqueues": bool(self._conf.enqueues),
            "current": self._key == decision_point_key(state),
        }
        turn = await _ask_brain(state, PREDICTOR_ENTRY_KEY, request, DecisionPort("the predictor's entry wait"))
        decision = one_decision(turn, ("predictor.wait",), what="the predictor's entry wait")
        if decision is None or self._inflight is None:
            return
        timeout = float(decision.args["timeout_sec"])
        log.info("predictor: holding orchestration's first turn up to %.0fs for %s", timeout, self._key)
        await asyncio.wait({self._inflight}, timeout=timeout)
        if self._inflight.done():
            await self.step(state)
        else:
            log.info("predictor: no answer at %s within %.0fs; orchestration starts without it", self._key, timeout)

    async def step(self, state: SessionView) -> None:
        """File an answer that has arrived, then ask when the current decision point warrants it."""
        if self._inflight is not None:
            if not self._inflight.done():
                return
            task, self._inflight = self._inflight, None
            self._file(state, task.result())
        await self._ask(state)

    async def _ask(self, state: SessionView) -> None:
        conf = predictor_config.load()
        key = decision_point_key(state)
        budgets: list[PhaseBudget] = []

        async def _headroom(_args: Mapping[str, Any]) -> dict[str, Any]:
            budgets.append(self._budget(state))
            return {"headroom_sec": budgets[-1].headroom_sec}

        async def _hold(_args: Mapping[str, Any]) -> dict[str, Any]:
            return self._profile_hold(state, key)

        request = {
            "enabled": bool(conf.enabled),
            "framework": state.framework,
            "asked": key in state.predictor_asked_keys,
        }
        port = DecisionPort("the predictor ask", {"query.phase_headroom": _headroom, "query.profile_hold": _hold})
        decision = one_decision(
            await _ask_brain(state, PREDICTOR_ASK_KEY, request, port), _ASK_DECISIONS, what="the predictor ask"
        )
        if decision is None:
            return
        if decision.name == "predictor.skip":
            log.debug("predictor: not asking (%s)", decision.args["reason"])
            return
        budget = budgets[-1] if budgets else self._budget(state)
        self._key, self._conf, self._started, self._asked_at = key, conf, time.monotonic(), now_iso()
        self._cycle = int(state.macro_cycle or 0)
        # Before anything can raise: a request that fails is not retried at the same decision point.
        state.predictor_asked_keys.append(key)
        report_path = state.last_trace_analyze.get("analysis_md_path")
        sites = await asyncio.to_thread(load_source_sites, report_path)
        sidecars = await asyncio.to_thread(load_sidecars, report_path)
        request = build_request(state, sites=sites, sidecars=sidecars, budget=budget)
        self._inflight = asyncio.create_task(
            asyncio.to_thread(predict, request, endpoint=conf.endpoint, timeout_sec=conf.timeout_sec)
        )
        log.info("predictor: asked %s at decision point %s (mode %s)", conf.endpoint, self._key, conf.mode)

    def _file(self, state: SessionView, answer: Prediction) -> None:
        key, latency_ms = self._key, int((time.monotonic() - self._started) * 1000)
        if not self._conf.enqueues:
            log.info(
                "predictor (shadow): key=%s parsed=%s latency=%dms actions=%r",
                key,
                answer.parsed,
                latency_ms,
                [(a.server_args, a.envs, a.source_change[:200]) for a in answer.actions],
            )
            return
        if not answer.parsed:
            log.info("predictor: no answer at %s after %dms (%s)", key, latency_ms, answer.error)
            return
        rows = proposal_rows(answer, key=key, state=state)
        mandate = patch_mandate(answer, key=key)
        if not rows and mandate is None:
            log.info("predictor: nothing new to queue at %s (%d proposals)", key, len(answer.config_actions))
            return
        # No task_id: the round has no task, and task_id names a registry task.
        filed = state.record_specialist_round(
            {
                "round_id": key,
                "cycle": self._cycle,
                "domain": QUEUE_DOMAIN,
                "provenance": PROVENANCE,
                "priority": QUEUE_PRIORITY,
                "asked_at": self._asked_at,
                "proposal_set": rows,
                "predict_meta": {
                    "latency_ms": latency_ms,
                    "samples": answer.meta.get("samples"),
                    "prompt_chars": answer.meta.get("prompt_chars"),
                    "actions_returned": len(answer.actions),
                },
                **(mandate or {}),
            }
        )
        log.info(
            "predictor: queued %d row(s) at %s after %dms%s: %r",
            len(rows),
            key,
            latency_ms,
            " with a source-change mandate" if mandate else "",
            [(row["extra_args"], row["extra_envs"], row["votes"]) for row in rows],
        )
        if filed is not None and self._on_filed is not None:
            self._on_filed(filed)

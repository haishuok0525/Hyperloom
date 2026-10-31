# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forced stalled-domain, custom specialist, wave split, macro, KERNEL and SWEEP strategies."""

from __future__ import annotations

from hyperloom.brain import create_brain
from hyperloom.brain.pump import (
    CONC_SWEEP_KEY,
    CUSTOM_SPECIALIST_KEY,
    CYCLE_REPROFILE_KEY,
    DISPATCH_ORDER_KEY,
    EXPLORE_COVERAGE_KEY,
    KERNEL_AGENT_KEY,
    KERNEL_ENTER_KEY,
    MACRO_FOCUS_KEY,
    STALLED_DOMAIN_KEY,
    SWEEP_DIRECTIVE_KEY,
    WAVE_SPLIT_KEY,
)
from hyperloom.brain_api import CallSite, StateSnapshot, TickBatch
from hyperloom.orchestrator.phases.focus_plan import plan_cycle_focus


def _port(**mapping):
    class P:
        llm: dict = {}

        async def call(self, call):
            answer = mapping.get(call.name, {})
            return answer(call.args) if callable(answer) else answer

    return P()


async def _pump(key: str, request: dict, port=None):
    batch = TickBatch(CallSite.PHASE_PUMP, 1, 0.0, (), StateSnapshot("FRAMEWORK_AGENT", {key: request}), b"kept")
    turn = await create_brain({}).on_phase_pump(batch, port or _port())
    assert turn.brain_state == b"kept"
    return [(c.name, c.args) for c in turn.tool_calls]


async def test_stalled_domain_forces_the_first_eligible_anchor() -> None:
    port = _port(**{"query.task_idempotency": {"exists": False}, "query.patch_preflight": {"refuse": False}})
    cand = {
        "anchor": "attn",
        "domain": "attention_specialist",
        "tags": ["attn"],
        "gap_canonical_id": "gap.a",
        "busy": False,
        "idempotency_key": "k1",
        "reason": "stalled_domain_force:attn",
    }
    calls = await _pump(STALLED_DOMAIN_KEY, {"phase": "FRAMEWORK_AGENT", "candidates": [cand]}, port)
    assert calls == [
        (
            "specialist.force_stalled",
            {
                "domain": "attention_specialist",
                "tags": ["attn"],
                "gap_canonical_id": "gap.a",
                "idempotency_key": "k1",
                "reason": "stalled_domain_force:attn",
            },
        )
    ]


async def test_stalled_domain_skips_busy_existing_and_refused() -> None:
    port = _port(
        **{
            "query.task_idempotency": lambda args: {"exists": args["key"] == "exist"},
            "query.patch_preflight": lambda args: {"refuse": args["idempotency_key"] == "refuse"},
        }
    )
    base = {"domain": "d", "tags": ["a"], "gap_canonical_id": "g", "reason": "r"}
    calls = await _pump(
        STALLED_DOMAIN_KEY,
        {
            "phase": "FRAMEWORK_AGENT",
            "candidates": [
                {**base, "anchor": "a", "tags": ["a"], "busy": True, "idempotency_key": "busy"},
                {**base, "anchor": "b", "tags": ["b"], "busy": False, "idempotency_key": "exist"},
                {**base, "anchor": "c", "tags": ["c"], "busy": False, "idempotency_key": "refuse"},
            ],
        },
        port,
    )
    assert calls == []


async def test_custom_specialist_dispatches_once() -> None:
    port = _port(
        **{
            "query.task_idempotency": {"exists": False},
            "query.custom_specialist_gap": {
                "gap_canonical_id": "gap.custom.session",
                "gap_symptom": "x",
                "gap_layer": "op",
            },
        }
    )
    calls = await _pump(
        CUSTOM_SPECIALIST_KEY,
        {"phase": "FRAMEWORK_AGENT", "prompt": "focus", "key": "c1", "refused_key": "", "custom_only": True},
        port,
    )
    assert calls[0][0] == "specialist.ensure_custom"
    assert calls[0][1]["no_lease_extension"] is True


async def test_wave_split_defaults_mode_to_research() -> None:
    calls = await _pump(
        WAVE_SPLIT_KEY,
        {"base_key": "w", "shared": {"domain": "freeform"}, "tasks": [{"task_description": "look"}, {"mode": "bench"}]},
    )
    assert calls[0][0] == "specialist.wave_split"
    assert [t["mode"] for t in calls[0][1]["tasks"]] == ["research", "bench"]


def test_macro_focus_prefers_bottleneck_shift() -> None:
    plan = plan_cycle_focus(
        {
            "domains": ["attention_specialist", "freeform_specialist"],
            "shift": {"to_domain": "attention_specialist", "to": "attention"},
            "saturated": {},
            "log_rows": [],
            "negative_counts": {},
            "macro_cycle": 2,
            "bottleneck_fallback": "",
            "gain_at_start": 1.0,
        }
    )
    assert plan["focus"] == "attention_specialist"
    assert plan["cycle"] == 2


async def test_macro_focus_emits_the_body_scored_plan() -> None:
    plan = {"cycle": 1, "focus": "freeform_specialist", "score": 0.0, "rationale": "x"}
    assert await _pump(MACRO_FOCUS_KEY, {"plan": plan}) == [("macro.set_focus", plan)]


async def test_cycle_reprofile_only_from_sweep() -> None:
    assert await _pump(CYCLE_REPROFILE_KEY, {"from_phase": "SWEEP", "macro_cycle": 1}) == [
        ("prelude.reprofile", {"reason": "cycle_start"})
    ]
    assert await _pump(CYCLE_REPROFILE_KEY, {"from_phase": "FRAMEWORK_AGENT", "macro_cycle": 1}) == []


async def test_kernel_enter_and_agent_routes() -> None:
    assert await _pump(
        KERNEL_ENTER_KEY,
        {"enabled": True, "geak": False, "route_reason": "r", "from_phase": "FRAMEWORK", "macro_cycle": 3},
    ) == [("kernel.enter", {"route": "forge", "route_reason": "r", "from_phase": "FRAMEWORK", "macro_cycle": 3})]
    assert await _pump(
        KERNEL_AGENT_KEY,
        {"geak": False, "skip_gemm": False, "framework": "sglang", "last_gemm_status": "", "from_phase": "x"},
    ) == [("kernel.run", {"route": "forge_gemm", "from_phase": "x"})]
    assert await _pump(
        KERNEL_AGENT_KEY,
        {"geak": True, "skip_gemm": False, "framework": "sglang", "last_gemm_status": "", "from_phase": "x"},
    ) == [("kernel.run", {"route": "geak", "from_phase": "x"})]


async def test_sweep_directive_and_conc_sweep() -> None:
    assert await _pump(
        SWEEP_DIRECTIVE_KEY,
        {
            "has_orchestration": True,
            "user_supplied_prompt": False,
            "memory_for_cycle": 1,
            "macro_cycle": 2,
            "reloop_feasible": True,
        },
    ) == [("sweep.ask_directive", {})]
    assert await _pump(
        CONC_SWEEP_KEY, {"enabled": False, "watermark": {}, "validated_gain": 0.0, "budget_denied": None}
    ) == [("sweep.skip_conc", {"reason": "disabled"})]
    assert await _pump(
        CONC_SWEEP_KEY, {"enabled": True, "watermark": {}, "validated_gain": 1.0, "budget_denied": None}
    ) == [("sweep.enqueue_conc", {"reason": "phase_entry"})]


async def test_dispatch_order_puts_operator_and_phase_first_kinds_ahead() -> None:
    calls = await _pump(
        DISPATCH_ORDER_KEY,
        {
            "phase": "FRAMEWORK_AGENT",
            "tasks": [
                {"id": "a", "kind": "explore", "operator_work": False},
                {"id": "b", "kind": "integrate_patch", "operator_work": False},
                {"id": "c", "kind": "specialist", "operator_work": True},
            ],
        },
    )
    assert calls == [("dispatch.order", {"ids": ["c", "b", "a"]})]


async def test_explore_coverage_refuses_only_a_covered_explore_without_lever_grid() -> None:
    assert await _pump(
        EXPLORE_COVERAGE_KEY,
        {"is_explore": True, "lever_grid": False, "covered": True, "message": "done", "hint": "h"},
    ) == [("explore.refuse_covered", {"message": "done", "hint": "h"})]
    assert await _pump(
        EXPLORE_COVERAGE_KEY,
        {"is_explore": True, "lever_grid": True, "covered": True, "message": "done", "hint": "h"},
    ) == []

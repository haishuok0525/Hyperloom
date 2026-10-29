# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The FRAMEWORK pump's order, and when the internal specialists are dispatched."""

from __future__ import annotations

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain.pump import FRAMEWORK_PUMP_KEY, RESEARCH_SCOUT_KEY, STATIC_RECON_KEY, TRAJECTORY_REVIEW_KEY
from hyperloom.brain_api import CallSite, StateSnapshot, TickBatch


class _Port:
    llm: dict = {}

    def __init__(self, **answers: dict) -> None:
        self.answers = answers
        self.asked: list[str] = []

    async def call(self, call):
        self.asked.append(call.name)
        return self.answers.get(call.name.split(".", 1)[1], {})


async def _pump(key: str, request: dict, port: _Port | None = None):
    batch = TickBatch(CallSite.PHASE_PUMP, 1, 0.0, (), StateSnapshot("FRAMEWORK_AGENT", {key: request}), b"kept")
    turn = await create_brain({}).on_phase_pump(batch, port or _Port())
    assert turn.brain_state == b"kept"
    return [(c.name, c.args) for c in turn.tool_calls]


@pytest.mark.parametrize(
    ("only", "steps"),
    [
        (True, ["pump.custom_specialist", "pump.bench_untested", "pump.finish_custom_cycle"]),
        (
            False,
            [
                "pump.framework_agent",
                "pump.bench_untested",
                "pump.research_scout",
                "pump.config_plateau_recon",
                "pump.stalled_domain",
                "pump.custom_specialist",
                "pump.advisory_plateau",
                "pump.predictor",
            ],
        ),
    ],
)
async def test_the_framework_pump_runs_its_steps_in_order(only, steps) -> None:
    port = _Port()
    assert await _pump(FRAMEWORK_PUMP_KEY, {"custom_specialist_only": only}, port) == []
    assert port.asked == steps


def _scout(**overrides) -> dict:
    return {"prelude": False, "enabled": True, "interval": None, "cursor": 3, "last_round": 0, **overrides}


@pytest.mark.parametrize(
    ("request_", "dispatched"),
    [
        ({"prelude": True, "enabled": True}, {"reason": "prelude_initial", "round_id": 0}),
        ({"prelude": True, "enabled": False}, None),
        (_scout(), {"reason": "explore_periodic", "round_id": 3}),
        (_scout(cursor=4), None),
        (_scout(cursor=4, interval=2), {"reason": "explore_periodic", "round_id": 4}),
        (_scout(cursor=0), None),
        (_scout(last_round=3), None),
        (_scout(enabled=False), None),
    ],
)
async def test_the_scout_runs_in_prelude_and_every_few_config_rounds(request_, dispatched) -> None:
    calls = await _pump(RESEARCH_SCOUT_KEY, request_)
    assert calls == ([("specialist.research_scout", dispatched)] if dispatched else [])


def _recon(**overrides) -> dict:
    return {"prelude": False, "enabled": True, "phase": "framework_agent", "macro_cycle": 2, **overrides}


@pytest.mark.parametrize(
    ("request_", "plateaued", "dispatched", "asked"),
    [
        ({"prelude": True, "enabled": True}, False, {"reason": "prelude_initial", "slug": "prelude"}, []),
        (_recon(), True, {"reason": "config_arm_plateau", "slug": "cycle2"}, ["query.config_arm_plateau"]),
        (_recon(), False, None, ["query.config_arm_plateau"]),
        (_recon(phase="SWEEP"), True, None, []),
        (_recon(enabled=False), True, None, []),
    ],
)
async def test_static_recon_reruns_once_the_config_arm_goes_dry(request_, plateaued, dispatched, asked) -> None:
    port = _Port(config_arm_plateau={"plateaued": plateaued})
    calls = await _pump(STATIC_RECON_KEY, request_, port)
    assert calls == ([("specialist.static_recon", dispatched)] if dispatched else [])
    assert port.asked == asked


@pytest.mark.parametrize(
    ("enabled", "active", "hinted", "dispatched", "asked"),
    [
        (
            True,
            True,
            "kernel_specialist",
            "kernel_specialist",
            ["query.plateau_advisory", "query.trajectory_direction"],
        ),
        (True, True, "", "serving_specialist", ["query.plateau_advisory", "query.trajectory_direction"]),
        (True, False, "", None, ["query.plateau_advisory"]),
        (False, True, "", None, []),
    ],
)
async def test_a_plateau_is_reviewed_by_the_domain_its_bottleneck_points_at(
    enabled, active, hinted, dispatched, asked
) -> None:
    port = _Port(plateau_advisory={"active": active}, trajectory_direction={"hinted_domain": hinted})
    calls = await _pump(TRAJECTORY_REVIEW_KEY, {"enabled": enabled}, port)
    assert calls == ([("specialist.trajectory_review", {"domain": dispatched})] if dispatched else [])
    assert port.asked == asked

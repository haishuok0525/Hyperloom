# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The FRAMEWORK candidate track, its discovery, the untested-proposal bench and the predictor."""

from __future__ import annotations

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain.pump import (
    AUTO_BENCH_KEY,
    CANDIDATE_DISCOVERY_KEY,
    FRAMEWORK_AGENT_KEY,
    PREDICTOR_ASK_KEY,
    PREDICTOR_ENTRY_KEY,
)
from hyperloom.brain_api import CallSite, StateSnapshot, TickBatch


class _Port:
    llm: dict = {}

    def __init__(self, **answers) -> None:
        self.answers = answers
        self.asked: list[tuple[str, dict]] = []

    async def call(self, call):
        self.asked.append((call.name, call.args))
        answer = self.answers.get(call.name, {})
        return answer(call.args) if callable(answer) else answer


async def _pump(key: str, request: dict, port: _Port | None = None):
    batch = TickBatch(CallSite.PHASE_PUMP, 1, 0.0, (), StateSnapshot("FRAMEWORK_AGENT", {key: request}), b"kept")
    turn = await create_brain({}).on_phase_pump(batch, port or _Port())
    assert turn.brain_state == b"kept"
    return [(c.name, c.args) for c in turn.tool_calls]


def _track(**overrides) -> dict:
    return {
        "phase": "FRAMEWORK_AGENT",
        "phase_done": False,
        "research_lane_capacity": 2,
        "local_explore_enabled": True,
        **overrides,
    }


def _pool(*ids: str):
    left = list(ids)
    return lambda args: {"found": bool(left), "id": left.pop(0) if left else ""}


async def test_the_track_fills_the_research_lanes_with_candidates() -> None:
    port = _Port(
        **{
            "query.framework_inflight_candidates": {"ids": []},
            "query.framework_next_candidate": _pool("a", "b", "c"),
            "framework.submit_candidate": lambda args: {"id": f"key-{args['id']}"},
        }
    )
    assert await _pump(FRAMEWORK_AGENT_KEY, _track(), port) == []
    assert [args for name, args in port.asked if name == "framework.submit_candidate"] == [{"id": "a"}, {"id": "b"}]
    assert ("query.framework_next_candidate", {"exclude": ["key-a"]}) in port.asked


@pytest.mark.parametrize(
    ("request_", "answers", "last"),
    [
        (_track(phase="CONFIG"), {}, None),
        (_track(phase_done=True), {}, None),
        (_track(), {"query.framework_inflight_candidates": {"ids": ["x", "y"]}}, "query.framework_inflight_candidates"),
        (_track(), {"query.framework_authoring_inflight": {"inflight": True}}, "query.framework_authoring_inflight"),
        (_track(), {"framework.discover": {"dispatched": True}}, "framework.discover"),
        (_track(), {"framework.local_explore": {"dispatched": True}}, "framework.local_explore"),
        (_track(), {"query.source_arm_plateau": {"plateaued": True}}, "framework.phase_done"),
        (_track(local_explore_enabled=False), {}, "framework.phase_done"),
    ],
)
async def test_an_empty_pool_falls_back_to_discovery_then_local_explore_then_phase_done(
    request_, answers, last
) -> None:
    port = _Port(
        **{
            "query.framework_inflight_candidates": {"ids": []},
            "query.framework_next_candidate": {"found": False, "id": ""},
            "query.framework_authoring_inflight": {"inflight": False},
            "framework.discover": {"dispatched": False},
            "query.source_arm_plateau": {"plateaued": False},
            "framework.local_explore": {"dispatched": False},
            **answers,
        }
    )
    assert await _pump(FRAMEWORK_AGENT_KEY, request_, port) == []
    assert (port.asked[-1][0] if port.asked else None) == last
    if last == "framework.phase_done":
        assert port.asked[-1][1] == {"reason": "no_candidates_and_discovery_exhausted"}
        assert "framework.local_explore" not in [name for name, _ in port.asked]


@pytest.mark.parametrize(
    ("empties", "failures", "inflight", "decision"),
    [
        (0, 0, False, "discovery.dispatch"),
        (2, 2, True, "discovery.wait"),
        (3, 0, False, None),
        (0, 3, True, None),
    ],
)
async def test_discovery_is_asked_for_until_three_rounds_came_back_empty_or_failed(
    empties, failures, inflight, decision
) -> None:
    port = _Port(**{"query.candidate_discovery_inflight": {"inflight": inflight}})
    calls = await _pump(CANDIDATE_DISCOVERY_KEY, {"empty_discoveries": empties, "discover_failures": failures}, port)
    assert calls == ([(decision, {})] if decision else [])


def _bench(**overrides) -> dict:
    return {"phase": "FRAMEWORK_AGENT", "custom_specialist_only": False, "phase_done": False, **overrides}


@pytest.mark.parametrize(
    ("request_", "answers", "benched"),
    [
        (_bench(), {}, [{"rows": "custom", "limit": None}]),
        (
            _bench(),
            {"bench.enqueue": {"created": False}},
            [{"rows": "custom", "limit": None}, {"rows": "untested", "limit": 4}],
        ),
        (
            _bench(),
            {"query.bench_queue": {"explores": 0, "custom_rows": 1, "operator_work": True}},
            [{"rows": "untested", "limit": 4}],
        ),
        (_bench(phase_done=True), {"query.bench_queue": {"explores": 0, "custom_rows": 0, "operator_work": False}}, []),
        (_bench(), {"query.bench_queue": {"explores": 1, "custom_rows": 0, "operator_work": False}}, []),
        (
            _bench(),
            {
                "query.bench_queue": {"explores": 0, "custom_rows": 0, "operator_work": False},
                "query.untested_rows": {"count": 0},
            },
            [],
        ),
        (_bench(), {"query.dispatch_admission": {"frozen": True, "paused": False}}, []),
        (_bench(phase="CONFIG"), {}, []),
    ],
)
async def test_the_operator_rows_are_benched_first_and_the_rest_four_at_a_time(request_, answers, benched) -> None:
    port = _Port(
        **{
            "query.dispatch_admission": {"frozen": False, "paused": False},
            "query.bench_queue": {"explores": 0, "custom_rows": 2, "operator_work": False},
            "query.untested_rows": {"count": 6},
            "bench.enqueue": lambda args: {"created": args["rows"] == "custom"},
            **answers,
        }
    )
    assert await _pump(AUTO_BENCH_KEY, request_, port) == []
    assert [args for name, args in port.asked if name == "bench.enqueue"] == benched


def _ask(**overrides) -> dict:
    return {"enabled": True, "framework": "sglang", "asked": False, **overrides}


@pytest.mark.parametrize(
    ("request_", "answers", "decision"),
    [
        (_ask(), {}, ("predictor.ask", {})),
        (_ask(enabled=False), {}, ("predictor.skip", {"reason": "disabled"})),
        (_ask(framework="trt"), {}, ("predictor.skip", {"reason": "no flag catalogue for framework 'trt'"})),
        (_ask(asked=True), {}, ("predictor.skip", {"reason": "decision point already asked"})),
        (
            _ask(),
            {"query.phase_headroom": {"headroom_sec": 599.6}},
            ("predictor.skip", {"reason": "phase headroom 600s is below one variant"}),
        ),
        (_ask(), {"query.profile_hold": {"pending": True, "held_sec": 899.0}}, None),
        (_ask(), {"query.profile_hold": {"pending": True, "held_sec": 900.0}}, ("predictor.ask", {})),
    ],
)
async def test_the_predictor_is_asked_once_per_point_while_a_variant_still_fits(request_, answers, decision) -> None:
    port = _Port(
        **{
            "query.phase_headroom": {"headroom_sec": None},
            "query.profile_hold": {"pending": False, "held_sec": 0.0},
            **answers,
        }
    )
    assert await _pump(PREDICTOR_ASK_KEY, request_, port) == ([decision] if decision else [])


@pytest.mark.parametrize(
    ("request_", "waits"),
    [
        ({"inflight": True, "enqueues": True, "current": True}, True),
        ({"inflight": False, "enqueues": True, "current": True}, False),
        ({"inflight": True, "enqueues": False, "current": True}, False),
        ({"inflight": True, "enqueues": True, "current": False}, False),
    ],
)
async def test_framework_entry_waits_five_minutes_for_an_answer_queued_at_this_point(request_, waits) -> None:
    calls = await _pump(PREDICTOR_ENTRY_KEY, request_)
    assert calls == ([("predictor.wait", {"timeout_sec": 300.0})] if waits else [])

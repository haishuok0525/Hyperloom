# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every brain contract entity survives a JSON round trip, and nothing that would not survive it gets in."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hyperloom.brain_api import (
    BrainTurn,
    CallSite,
    Event,
    LlmReply,
    Result,
    StateSnapshot,
    TickBatch,
    ToolCall,
    WireError,
)


def _event(seq: int = 3, **payload) -> Event:
    return Event(
        seq=seq,
        msg_id=f"m{seq}",
        topic="delegated_result",
        from_agent="coordinator",
        to_agent="*",
        payload=payload or {"task_id": "t1", "status": "succeeded", "gain_pct": 2.5},
        in_reply_to="p1",
        ts="2026-10-28T00:00:00Z",
    )


def _batch(**overrides) -> TickBatch:
    fields = dict(
        call_site=CallSite.PHASE_PUMP,
        tick=7,
        now_unix=1_790_000_000.25,
        events=(_event(3), _event(5)),
        snapshot=StateSnapshot(phase="FRAMEWORK_AGENT", data={"current_best": {"tput": 1234.5}, "stack": ["a", "b"]}),
        brain_state=b"\x00\xffstate",
    )
    fields.update(overrides)
    return TickBatch(**fields)


def _through_json(obj):
    return type(obj).from_wire(json.loads(json.dumps(obj.to_wire())))


@pytest.mark.parametrize(
    "obj",
    [
        ToolCall("action.explore", {"params": {"grid": [[1, 2], [3]]}}, idempotency_key="k1"),
        _event(),
        Result("c1", "KEEP", 1.5, "beat current best"),
        Result("c2", "NEEDS_REVIEW", None, "noisy"),
        StateSnapshot(phase="PRELUDE"),
        _batch(),
        _batch(brain_state=None, events=()),
        BrainTurn(tool_calls=(ToolCall("task.prune", {"family": "attn", "reason": "flat"}),), brain_state=b"s"),
        BrainTurn(),
    ],
)
def test_entity_round_trips_through_json_unchanged(obj) -> None:
    assert _through_json(obj) == obj


def test_tuples_are_stored_as_they_read_back_from_json() -> None:
    call = ToolCall("message.send", {"topic": "advice", "rows": (1, (2, 3))})
    assert call.args == {"topic": "advice", "rows": [1, [2, 3]]}
    assert _through_json(call) == call


def test_args_are_a_copy_of_what_the_caller_passed() -> None:
    args = {"rows": [1]}
    call = ToolCall("message.send", args)
    args["rows"].append(2)
    assert call.args == {"rows": [1]}


@pytest.mark.parametrize(
    "payload",
    [
        {"path": Path("/tmp")},
        {"rows": {1, 2}},
        {"gain": math.nan},
        {"gain": math.inf},
        {1: "int key"},
        {"nested": {"obj": object()}},
    ],
)
def test_a_value_that_would_not_survive_json_is_refused(payload) -> None:
    with pytest.raises(WireError):
        ToolCall("message.send", payload)
    with pytest.raises(WireError):
        _event(x=payload)


def test_an_unknown_tool_topic_or_verdict_is_refused() -> None:
    with pytest.raises(WireError, match="unknown tool"):
        ToolCall("shell.exec", {})
    with pytest.raises(WireError, match="unknown topic"):
        Event(seq=1, msg_id="m", topic="gossip", from_agent="a", to_agent="b", payload={})
    with pytest.raises(WireError, match="unknown verdict"):
        Result("c", "MAYBE", None, "")  # type: ignore[arg-type]


def test_a_batch_refuses_events_out_of_seq_order_and_a_nan_clock() -> None:
    with pytest.raises(WireError, match="seq order"):
        _batch(events=(_event(5), _event(3)))
    with pytest.raises(WireError):
        _batch(now_unix=math.nan)


def test_brain_state_travels_as_base64_and_bad_base64_is_refused() -> None:
    wire = _batch().to_wire()
    assert isinstance(wire["brain_state"], str)
    wire["brain_state"] = "not base64!"
    with pytest.raises(WireError, match="base64"):
        TickBatch.from_wire(wire)


def test_llm_reply_holds_json_intents_only() -> None:
    reply = LlmReply(intents=({"type": "alert", "payload": {"severity": "low", "summary": "s"}},), raw_text="t")
    assert reply.intents[0]["payload"]["severity"] == "low"
    with pytest.raises(WireError):
        LlmReply(intents=({"type": "alert", "payload": object()},))

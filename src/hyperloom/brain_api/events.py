# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the body hands the brain at a call site, and what the brain hands back."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping

from hyperloom.brain_api.tools import ToolCall
from hyperloom.brain_api.wire import WireError, decode_state, encode_state, json_value

#: The message-bus topics an event can carry.
TOPICS: frozenset[str] = frozenset(
    {
        "proposal",
        "observation",
        "event",
        "decision",
        "alert",
        "delegated_result",
        "request",
        "response",
        "review_verdict",
        "advice",
        "strategy_change",
    }
)

Verdict = Literal["KEEP", "REVERT", "NEEDS_REVIEW"]
VERDICTS: frozenset[str] = frozenset({"KEEP", "REVERT", "NEEDS_REVIEW"})


class CallSite(str, Enum):
    """Where the body calls the brain: three points of one Coordinator tick, then the brief and advisory score."""

    #: The orchestration and critic model turns.
    LLM_TURN = "llm_turn"
    #: A settlement's follow-ups: a Critic verdict on a proposal, a specialist's deliverable, a re-author.
    SETTLEMENT = "settlement"
    #: The phase pump: the current phase's deterministic strategy and the internal specialists it dispatches.
    PHASE_PUMP = "phase_pump"
    #: Outside the tick order: a specialist job is about to run and needs its brief.
    BRIEF = "brief"
    #: Outside the tick order: advisory multi-model scoring of a specialist proposal set.
    SCORE = "score"


@dataclass(frozen=True)
class Event:
    """One bus message, as the brain sees it."""

    seq: int
    msg_id: str
    topic: str
    from_agent: str
    to_agent: str
    payload: Mapping[str, Any]
    in_reply_to: str | None = None
    ts: str = ""

    def __post_init__(self) -> None:
        if self.topic not in TOPICS:
            raise WireError(f"unknown topic {self.topic!r}")
        object.__setattr__(self, "payload", json_value(dict(self.payload), f"event {self.seq} payload"))

    def to_wire(self) -> dict[str, Any]:
        """This event as a JSON object."""
        return {
            "seq": self.seq,
            "msg_id": self.msg_id,
            "topic": self.topic,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "payload": self.payload,
            "in_reply_to": self.in_reply_to,
            "ts": self.ts,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "Event":
        """The inverse of :meth:`to_wire`."""
        return cls(
            seq=int(data["seq"]),
            msg_id=data["msg_id"],
            topic=data["topic"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            payload=data.get("payload") or {},
            in_reply_to=data.get("in_reply_to"),
            ts=data.get("ts") or "",
        )


@dataclass(frozen=True)
class Result:
    """A referee's ruling on one measured candidate, made inside a measurement tool."""

    candidate_id: str
    verdict: Verdict
    gain_pct: float | None
    reason: str

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise WireError(f"unknown verdict {self.verdict!r}")
        json_value(self.gain_pct, f"result {self.candidate_id} gain_pct")

    def to_wire(self) -> dict[str, Any]:
        """This result as a JSON object."""
        return {
            "candidate_id": self.candidate_id,
            "verdict": self.verdict,
            "gain_pct": self.gain_pct,
            "reason": self.reason,
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "Result":
        """The inverse of :meth:`to_wire`."""
        return cls(
            candidate_id=data["candidate_id"],
            verdict=data["verdict"],
            gain_pct=data.get("gain_pct"),
            reason=data.get("reason") or "",
        )


@dataclass(frozen=True)
class StateSnapshot:
    """The body state the brain may decide on: current best, stack, explore search, tasks, budgets, ..."""

    phase: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", json_value(dict(self.data), "snapshot data"))

    def to_wire(self) -> dict[str, Any]:
        """This snapshot as a JSON object."""
        return {"phase": self.phase, "data": self.data}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "StateSnapshot":
        """The inverse of :meth:`to_wire`."""
        return cls(phase=data["phase"], data=data.get("data") or {})


@dataclass(frozen=True)
class TickBatch:
    """Everything one brain call is given. The brain reads no clock and no file; this is its whole input."""

    call_site: CallSite
    tick: int
    #: The body's clock for this tick.
    now_unix: float
    #: Bus messages since the brain's cursor, in ``seq`` order; at a settlement, the one settled task.
    events: tuple[Event, ...]
    snapshot: StateSnapshot
    #: What the brain returned last time, handed back unchanged.
    brain_state: bytes | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_site", CallSite(self.call_site))
        object.__setattr__(self, "events", tuple(self.events))
        json_value(self.now_unix, "now_unix")
        seqs = [e.seq for e in self.events]
        if seqs != sorted(seqs):
            raise WireError("events must be in seq order")

    def to_wire(self) -> dict[str, Any]:
        """This batch as a JSON object."""
        return {
            "call_site": self.call_site.value,
            "tick": self.tick,
            "now_unix": self.now_unix,
            "events": [e.to_wire() for e in self.events],
            "snapshot": self.snapshot.to_wire(),
            "brain_state": encode_state(self.brain_state),
        }

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "TickBatch":
        """The inverse of :meth:`to_wire`."""
        return cls(
            call_site=CallSite(data["call_site"]),
            tick=int(data["tick"]),
            now_unix=float(data["now_unix"]),
            events=tuple(Event.from_wire(e) for e in data.get("events") or ()),
            snapshot=StateSnapshot.from_wire(data["snapshot"]),
            brain_state=decode_state(data.get("brain_state")),
        )


@dataclass(frozen=True)
class BrainTurn:
    """What one brain call decides: the tool calls to make, in order, and the state to keep."""

    tool_calls: tuple[ToolCall, ...] = ()
    brain_state: bytes | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    def to_wire(self) -> dict[str, Any]:
        """This turn as a JSON object."""
        return {"tool_calls": [c.to_wire() for c in self.tool_calls], "brain_state": encode_state(self.brain_state)}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "BrainTurn":
        """The inverse of :meth:`to_wire`."""
        return cls(
            tool_calls=tuple(ToolCall.from_wire(c) for c in data.get("tool_calls") or ()),
            brain_state=decode_state(data.get("brain_state")),
        )

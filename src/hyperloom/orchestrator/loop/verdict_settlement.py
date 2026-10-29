# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Critic verdict's settlement. The brain holds the verdict to the rule it cites, collapses a per-variant map and
says what the proposal comes to; the router records and carries out what it settled."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

from hyperloom.brain_api import BrainContractError, BrainTurn, CallSite, StateSnapshot, TickBatch, VerdictOutcome

from ..brain_link import DecisionPort, resolve_brain
from ..specialists.patch_safety import advisory_only_reason_codes, advisory_rules_govern

# Snapshot key the brain reads a verdict's settlement under.
_SETTLEMENT_KEY = "review_verdict_settlement"
_SETTLE = "verdict.settle"


@dataclass(frozen=True)
class VerdictHold:
    """One verdict the brain held to the advisory rule it cites; ``variant`` is empty for a single verdict."""

    variant: str
    to_verdict: str
    failure_reason_code: str


@dataclass(frozen=True)
class Settlement:
    """What one Critic verdict settled to."""

    verdict: str
    authored: str
    #: Each ``verdict_map`` variant's verdict after its hold; empty for a single verdict.
    variants: dict[str, str]
    #: The variants that may run; ``None`` keeps the full proposal.
    approved_variants: set[str] | None
    holds: tuple[VerdictHold, ...]
    outcome: VerdictOutcome


def settlement_request(review: Mapping[str, Any], action_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """What the brain settles a verdict from: the review, its proposal, and the advisory rules the review answers to."""
    return {
        "review_verdict": dict(review),
        "proposal": {"action_name": action_name, "payload": dict(payload)},
        "advisory_rules": {
            "reason_codes": sorted(advisory_only_reason_codes()),
            "govern": advisory_rules_govern(action_name),
        },
    }


async def settle_verdict(request: Mapping[str, Any], *, tick: int, phase: str) -> Settlement:
    """Ask the brain what the verdict in ``request`` settles to.

    Raises:
        BrainContractError: if the brain answers with anything but one well-formed ``verdict.settle``.
    """
    brain = resolve_brain()
    batch = TickBatch(
        call_site=CallSite.SETTLEMENT,
        tick=tick,
        now_unix=time.time(),
        events=(),
        snapshot=StateSnapshot(phase=phase, data={_SETTLEMENT_KEY: dict(request)}),
    )
    turn = await brain.on_settlement(batch, DecisionPort("a verdict settlement"))
    try:
        return _settlement(turn)
    except (KeyError, TypeError, ValueError) as exc:
        raise BrainContractError(f"brain {brain.name!r} settled a verdict malformed: {exc!r}") from exc


def _settlement(turn: BrainTurn) -> Settlement:
    calls = turn.tool_calls
    if len(calls) != 1 or calls[0].name != _SETTLE:
        raise ValueError(f"expected one {_SETTLE}, got {[c.name for c in calls]}")
    args = calls[0].args
    approved = args["approved_variants"]
    return Settlement(
        verdict=str(args["verdict"]),
        authored=str(args["authored"]),
        variants={str(name): str(verdict) for name, verdict in dict(args["variants"]).items()},
        approved_variants=None if approved is None else {str(name) for name in approved},
        holds=tuple(
            VerdictHold(str(h["variant"]), str(h["to_verdict"]), str(h["failure_reason_code"])) for h in args["holds"]
        ),
        outcome=VerdictOutcome(args["outcome"]),
    )

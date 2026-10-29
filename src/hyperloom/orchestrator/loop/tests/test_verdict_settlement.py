# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Critic verdict is settled by the brain, and the patch approval it requires is the brain's to say."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.brain_api import BrainContractError, BrainTurn, ToolCall, VerdictOutcome
from hyperloom.orchestrator.loop import verdict_settlement
from hyperloom.orchestrator.loop.verdict_settlement import VerdictHold, settle_verdict, settlement_request
from hyperloom.orchestrator.policy import gate


class _Brain:
    name = "scripted"

    def __init__(self, turn: BrainTurn):
        self.turn = turn
        self.batches: list = []

    async def on_settlement(self, batch, port):
        self.batches.append(batch)
        return self.turn


def _plug(monkeypatch, turn: BrainTurn) -> _Brain:
    brain = _Brain(turn)
    monkeypatch.setattr(verdict_settlement, "resolve_brain", lambda: brain)
    return brain


async def test_the_body_carries_out_the_ruling_the_brain_settled_a_verdict_to(monkeypatch) -> None:
    ruling = {
        "verdict": "approve",
        "authored": "reject",
        "variants": {"v_a": "advise", "v_b": "approve"},
        "approved_variants": ["v_a", "v_b"],
        "holds": [{"variant": "v_a", "to_verdict": "advise", "failure_reason_code": "c"}],
        "outcome": "materialize",
    }
    brain = _plug(monkeypatch, BrainTurn(tool_calls=(ToolCall("verdict.settle", ruling),)))
    request = settlement_request({"verdict": "reject"}, "specialist", {"params": {}})
    settled = await settle_verdict(request, tick=7, phase="FRAMEWORK")
    assert (settled.verdict, settled.authored, settled.outcome) == ("approve", "reject", VerdictOutcome.MATERIALIZE)
    assert settled.variants == {"v_a": "advise", "v_b": "approve"}
    assert settled.approved_variants == {"v_a", "v_b"}
    assert settled.holds == (VerdictHold("v_a", "advise", "c"),)
    (batch,) = brain.batches
    assert (batch.tick, batch.snapshot.phase) == (7, "FRAMEWORK")
    assert batch.snapshot.data["review_verdict_settlement"]["advisory_rules"]["govern"] is True


@pytest.mark.parametrize(
    "turn",
    [
        BrainTurn(),
        BrainTurn(tool_calls=(ToolCall("verdict.record", {}),)),
        BrainTurn(tool_calls=(ToolCall("verdict.settle", {"verdict": "approve"}),)),
    ],
)
async def test_a_brain_that_does_not_settle_the_verdict_breaks_the_contract(monkeypatch, turn) -> None:
    _plug(monkeypatch, turn)
    with pytest.raises(BrainContractError, match="settled a verdict malformed"):
        await settle_verdict(settlement_request({"verdict": "approve"}, "explore", {}), tick=0, phase="")


def test_integrate_patch_runs_on_the_verdicts_the_brain_approves_a_patch_on(monkeypatch) -> None:
    monkeypatch.setattr(gate, "resolve_brain", lambda: SimpleNamespace(patch_approval_verdicts=("approve",)))
    assert gate.integrate_patch_permissive_verdicts() == frozenset({"approve"})

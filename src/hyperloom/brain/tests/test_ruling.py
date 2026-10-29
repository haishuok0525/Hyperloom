# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a Critic verdict settles to: the hold to a cited rule, the per-variant collapse, and the proposal's outcome."""

from __future__ import annotations

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain.critic import SETTLEMENT_KEY, verdict_outcome
from hyperloom.brain_api import CallSite, StateSnapshot, TickBatch, VerdictOutcome

_CODE = "specialist_quantitative_claim_violation"


class _NoTools:
    llm: dict = {}

    async def call(self, call):
        raise AssertionError(f"unexpected tool call {call.name}")


async def _settle(review: dict, *, action_name: str = "specialist", payload: dict | None = None, govern: bool = True):
    request = {
        "review_verdict": review,
        "proposal": {"action_name": action_name, "payload": payload or {}},
        "advisory_rules": {"reason_codes": [_CODE], "govern": govern},
    }
    batch = TickBatch(CallSite.SETTLEMENT, 3, 0.0, (), StateSnapshot("FRAMEWORK", {SETTLEMENT_KEY: request}), b"kept")
    turn = await create_brain({}).on_settlement(batch, _NoTools())
    assert turn.brain_state == b"kept"
    (call,) = turn.tool_calls
    assert call.name == "verdict.settle"
    return call.args


async def test_a_reject_resting_on_an_advisory_rule_alone_is_held_to_advise_and_materialises() -> None:
    settled = await _settle({"verdict": "reject", "reasoning": f"{_CODE}: carries predicted_gain_pct."})
    assert settled == {
        "verdict": "advise",
        "authored": "reject",
        "variants": {},
        "approved_variants": None,
        "holds": [{"variant": "", "to_verdict": "advise", "failure_reason_code": _CODE}],
        "outcome": "materialize",
    }


async def test_a_reject_is_not_held_where_the_advisory_rules_do_not_speak() -> None:
    settled = await _settle({"verdict": "reject", "reasoning": f"{_CODE}: x."}, action_name="explore", govern=False)
    assert (settled["verdict"], settled["holds"], settled["outcome"]) == ("reject", [], "none")


async def test_a_verdict_map_is_held_per_variant_and_collapses_to_the_variants_that_may_run() -> None:
    review = {
        "verdict_map": {
            "v_a": {"verdict": "reject", "rationale": f"{_CODE}: self-reported gain."},
            "v_b": {"verdict": "reject", "rationale": "regresses accuracy"},
            "v_c": {"verdict": "approve", "rationale": "ok"},
        }
    }
    settled = await _settle(review, action_name="explore")
    assert settled["variants"] == {"v_a": "advise", "v_b": "reject", "v_c": "approve"}
    assert (settled["verdict"], settled["authored"]) == ("approve", "approve")
    assert settled["approved_variants"] == ["v_a", "v_c"]
    assert settled["holds"] == [{"variant": "v_a", "to_verdict": "advise", "failure_reason_code": _CODE}]


@pytest.mark.parametrize(
    ("verdict", "action_name", "payload", "outcome"),
    [
        ("approve", "explore", {}, VerdictOutcome.MATERIALIZE),
        ("advise", "integrate_patch", {}, VerdictOutcome.MATERIALIZE),
        ("reject", "integrate_patch", {"framework_agent_candidate_id": "c1"}, VerdictOutcome.RECORD_DENIAL),
        ("reject", "integrate_patch", {"params": {"enablement": True}}, VerdictOutcome.REARM_ENABLEMENT),
        ("reject", "explore", {"params": {"enablement": True}}, VerdictOutcome.NONE),
        ("needs_review", "integrate_patch", {}, VerdictOutcome.REAUTHOR),
        ("redirect", "specialist", {}, VerdictOutcome.NONE),
    ],
)
def test_each_verdict_comes_to_its_outcome(verdict, action_name, payload, outcome) -> None:
    assert verdict_outcome(verdict, action_name, payload) is outcome


def test_v0_approves_a_patch_on_approve_or_advise() -> None:
    assert create_brain({}).patch_approval_verdicts == ("advise", "approve")

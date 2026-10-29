# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a settled specialist's deliverable and a needs_review re-author come to."""

from __future__ import annotations

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain.settle import CONFIG_DELIVERY_KEY, PATCH_DELIVERY_KEY, REAUTHOR_KEY, REAUTHOR_MAX_ATTEMPTS
from hyperloom.brain_api import CallSite, StateSnapshot, TickBatch


class _Answers:
    llm: dict = {}

    def __init__(self, **answers: dict) -> None:
        self.answers = answers
        self.asked: list[str] = []

    async def call(self, call):
        self.asked.append(call.name)
        return self.answers[call.name.removeprefix("query.")]


async def _decide(key: str, request: dict, port: _Answers | None = None):
    batch = TickBatch(CallSite.SETTLEMENT, 1, 0.0, (), StateSnapshot("FRAMEWORK", {key: request}), b"kept")
    turn = await create_brain({}).on_settlement(batch, port or _Answers())
    assert turn.brain_state == b"kept"
    return [(c.name, c.args) for c in turn.tool_calls]


@pytest.mark.parametrize(
    ("on_disk", "claimed", "expected"),
    [
        ({"patches_on_disk": True, "artifacts_on_disk": False}, ["a.diff"], [("review.submit", {})]),
        ({"patches_on_disk": False, "artifacts_on_disk": True}, [], [("review.submit", {})]),
        (
            {"patches_on_disk": False, "artifacts_on_disk": False},
            ["a.diff"],
            [("review.skip", {"claimed": ["a.diff"]})],
        ),
        ({"patches_on_disk": False, "artifacts_on_disk": False}, [], []),
    ],
)
async def test_patches_on_disk_are_reviewed_and_a_claim_without_files_is_noted(on_disk, claimed, expected) -> None:
    assert await _decide(PATCH_DELIVERY_KEY, {"claimed": claimed, **on_disk}) == expected


def _config(**overrides) -> dict:
    return {
        "specialist_task_id": "spec-1",
        "authoring": True,
        "patches_written": False,
        "enablement": False,
        "build_requested": False,
        "setup_or_artifacts": False,
        **overrides,
    }


_LEVERS = {"extra_server_args": "--x", "extra_envs": {}}


@pytest.mark.parametrize(
    ("delivery", "levers", "known_bad", "submitted", "asked"),
    [
        (_config(), _LEVERS, False, True, ["query.config_levers", "query.config_lever_known_bad"]),
        (_config(), _LEVERS, True, False, ["query.config_levers", "query.config_lever_known_bad"]),
        (_config(), {}, False, False, ["query.config_levers"]),
        (_config(enablement=True), {}, False, True, ["query.config_levers"]),
        (_config(enablement=True, build_requested=True), {}, False, False, ["query.config_levers"]),
        (
            _config(enablement=True, build_requested=True, setup_or_artifacts=True),
            {},
            False,
            True,
            ["query.config_levers"],
        ),
        (_config(authoring=False), _LEVERS, False, False, []),
        (_config(patches_written=True), _LEVERS, False, False, []),
        (_config(specialist_task_id=""), _LEVERS, False, False, ["query.config_levers"]),
    ],
)
async def test_a_config_lever_is_reviewed_unless_the_kb_already_rejected_it(
    delivery, levers, known_bad, submitted, asked
) -> None:
    port = _Answers(config_levers={"levers": levers}, config_lever_known_bad={"known_bad": known_bad})
    calls = await _decide(CONFIG_DELIVERY_KEY, delivery, port)
    assert calls == ([("review.submit", {})] if submitted else [])
    assert port.asked == asked


_ADVISORY = {"required_evidence": [" profile ", ""], "advice_text": "show it", "risks": ["r"]}


async def test_a_reauthor_seeds_the_next_round_with_the_critics_feedback() -> None:
    port = _Answers(candidate_materializing={"materializing": False})
    calls = await _decide(REAUTHOR_KEY, {"advisory": _ADVISORY, "prior_attempts": 1}, port)
    feedback = {"required_evidence": ["profile"], "advice_text": "show it", "risks": ["r"]}
    assert calls == [("reauthor.dispatch", {"attempt": 2, "critic_feedback": feedback})]


async def test_a_reauthor_without_evidence_to_supply_is_declined_before_anything_is_asked() -> None:
    port = _Answers()
    calls = await _decide(REAUTHOR_KEY, {"advisory": {"advice_text": "x" * 600}, "prior_attempts": 0}, port)
    assert calls == [("reauthor.decline", {"rationale": "x" * 500})]
    assert port.asked == []


async def test_a_materializing_candidate_is_left_alone_and_a_spent_budget_closes_it() -> None:
    busy = _Answers(candidate_materializing={"materializing": True})
    assert await _decide(REAUTHOR_KEY, {"advisory": _ADVISORY, "prior_attempts": 9}, busy) == []
    idle = _Answers(candidate_materializing={"materializing": False})
    calls = await _decide(REAUTHOR_KEY, {"advisory": _ADVISORY, "prior_attempts": REAUTHOR_MAX_ATTEMPTS}, idle)
    rationale = f"reauthor attempts >= cap ({REAUTHOR_MAX_ATTEMPTS})"
    assert calls == [("reauthor.exhausted", {"attempts": REAUTHOR_MAX_ATTEMPTS, "rationale": rationale})]

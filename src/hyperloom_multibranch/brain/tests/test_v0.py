# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0: the judge round the scheduler made before the brain was cut out, behind the brain contract."""

from __future__ import annotations

import json

import pytest

from hyperloom_multibranch.brain import BODY_BASELINE_COMMIT, BrainV0
from hyperloom_multibranch.brain.judge import NO_ANSWER, REPLY_FORMAT, user_prompt
from hyperloom_multibranch.contracts.brain import (
    BRAIN_CONTRACT_VERSION,
    BRAIN_KEY,
    JUDGE_PROMPT_KEY,
    BrainSetup,
    CallSite,
    Exchange,
    JudgeAnswer,
    JudgeObservation,
    KOrder,
    StopOrder,
)
from hyperloom_multibranch.contracts.llm import CompletionFailed, CompletionReply
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.scheduling.brain_link import JUDGE_FACTS, specialist_facts

#: The judge prompt versions the scheduler recorded at :data:`BODY_BASELINE_COMMIT`, without and with a playbook.
BASELINE_PROMPT_VERSIONS = ("c0f88a655700", "16ef19bcdfa4")
STATE = {"metric": "output_throughput", "opening": False, "branches": [{"branch": "b01", "k": 2}]}


def _brain() -> BrainV0:
    return BrainV0(BrainSetup(judge=JUDGE_FACTS, specialist=specialist_facts()))


def _reply(text: str) -> CompletionReply:
    return CompletionReply(text, TokenCounts(), None)


def test_v0_answers_the_judge_round_and_the_branch_brief_under_the_current_contract() -> None:
    brain = _brain()

    assert (brain.name, brain.contract_version) == ("v0", BRAIN_CONTRACT_VERSION)
    assert brain.call_sites == frozenset({CallSite.JUDGE_ROUND, CallSite.BRANCH_BRIEF})
    assert len(BODY_BASELINE_COMMIT) == 40


def test_the_judge_prompts_are_the_ones_the_scheduler_used_at_the_baseline() -> None:
    brain = _brain()

    plain = brain.versions(plays_playbook=False)
    played = brain.versions(plays_playbook=True)

    assert (plain[JUDGE_PROMPT_KEY], played[JUDGE_PROMPT_KEY]) == BASELINE_PROMPT_VERSIONS
    assert plain[BRAIN_KEY] == played[BRAIN_KEY] == "v0"


def test_the_prompts_state_the_body_numbers_they_are_handed() -> None:
    prompts = _brain().prompts

    assert f"reaches {JUDGE_FACTS.stall_tests}," in prompts.plain
    assert f"{JUDGE_FACTS.grace_tests} of its explore tests" in prompts.plain
    assert f"for {JUDGE_FACTS.repair_protected_seconds / 60:g} minutes" in prompts.playbook
    assert "{" not in prompts.plain + prompts.playbook


def test_the_user_prompt_is_the_state_then_the_sections_then_the_reply_format() -> None:
    observation = JudgeObservation(STATE, history="h", open_areas="Open areas: x", playbook="Scheduling playbook: y")

    prompt = user_prompt(observation)

    assert prompt.startswith(f"Session state:\n{json.dumps(STATE, indent=2, sort_keys=True)}\n\n")
    assert prompt.index("How similar past sessions went:\nh") < prompt.index("Open areas") < prompt.index("playbook: y")
    assert prompt.endswith(REPLY_FORMAT)
    assert "How similar" not in user_prompt(JudgeObservation(STATE))


async def test_a_judge_round_asks_once_and_reads_the_answer() -> None:
    seen: list[tuple[str, str]] = []
    reply = json.dumps({"open": 1, "branches": [{"branch": "b01", "k": 3}], "stop": [{"branch": "b02", "reason": "r"}]})

    async def llm(system: str, user: str) -> CompletionReply:
        seen.append((system, user))
        return _reply(reply)

    brain = _brain()
    played = JudgeObservation(STATE, playbook="Scheduling playbook: y")
    outcome = await brain.judge_round(played, llm)

    assert outcome.answer == JudgeAnswer(open_count=1, k=(KOrder("b01", 3),), stop=(StopOrder("b02", "r"),))
    assert seen == [(brain.prompts.playbook, user_prompt(played))]
    assert outcome.exchanges == (Exchange(user_prompt(played), reply, ""),)


async def test_a_failed_call_gives_no_answer_and_says_why() -> None:
    async def llm(system: str, user: str) -> CompletionReply:
        raise CompletionFailed("status=529")

    outcome = await _brain().judge_round(JudgeObservation(STATE), llm)

    assert outcome.answer is None
    assert [(item.reply, item.error) for item in outcome.exchanges] == [("", "status=529")]


async def test_an_unreadable_reply_gives_no_answer() -> None:
    async def llm(system: str, user: str) -> CompletionReply:
        return _reply("I would open two branches.")

    outcome = await _brain().judge_round(JudgeObservation(STATE), llm)

    assert outcome.answer is None
    assert [item.error for item in outcome.exchanges] == [NO_ANSWER]


async def test_an_unexpected_error_is_not_swallowed() -> None:
    async def llm(system: str, user: str) -> CompletionReply:
        raise KeyError("bug")

    with pytest.raises(KeyError):
        await _brain().judge_round(JudgeObservation(STATE), llm)

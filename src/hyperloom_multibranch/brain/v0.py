# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0: the multi-branch scheduler's decisions as the body made them at :data:`BODY_BASELINE_COMMIT`.

It answers :attr:`~hyperloom_multibranch.contracts.brain.CallSite.JUDGE_ROUND` with the judge prompt and reply format
the scheduler used, and :attr:`~hyperloom_multibranch.contracts.brain.CallSite.BRANCH_BRIEF` with the specialist's role,
launch prompt and session messages it sent, so a session run on it is the session the body ran before the brain was
cut out.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from hyperloom_multibranch.brain.judge import JudgePrompts, judge_round
from hyperloom_multibranch.brain.specialist import SpecialistPrompts
from hyperloom_multibranch.contracts.brain import (
    BRAIN_CONTRACT_VERSION,
    BRAIN_KEY,
    JUDGE_PROMPT_KEY,
    BrainSetup,
    CallSite,
    JudgeObservation,
    JudgeRound,
)
from hyperloom_multibranch.contracts.llm import Completion

#: The multibranch-dev commit whose scheduler Brain v0 was cut from.
BODY_BASELINE_COMMIT = "034941d0aa0e624d8f5ea33b71f6d42197302f53"


@dataclass(frozen=True)
class BrainV0:
    """Brain v0 for the body's ``setup``."""

    setup: BrainSetup
    prompts: JudgePrompts = field(init=False)
    voice: SpecialistPrompts = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompts", JudgePrompts.for_facts(self.setup.judge))
        object.__setattr__(self, "voice", SpecialistPrompts(self.setup.specialist))

    @property
    def name(self) -> str:
        return "v0"

    @property
    def contract_version(self) -> str:
        return BRAIN_CONTRACT_VERSION

    @property
    def call_sites(self) -> frozenset[CallSite]:
        return frozenset({CallSite.JUDGE_ROUND, CallSite.BRANCH_BRIEF})

    @property
    def specialist(self) -> SpecialistPrompts:
        return self.voice

    def versions(self, *, plays_playbook: bool) -> Mapping[str, str]:
        return {BRAIN_KEY: self.name, JUDGE_PROMPT_KEY: self.prompts.version(plays_playbook=plays_playbook)}

    async def judge_round(self, observation: JudgeObservation, llm: Completion) -> JudgeRound:
        return await judge_round(observation, llm, self.prompts)


__all__ = ["BODY_BASELINE_COMMIT", "BrainV0"]

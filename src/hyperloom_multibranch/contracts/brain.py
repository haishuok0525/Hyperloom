# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brain: the search decisions the scheduler asks a swappable implementation for, and what each call carries.

The rest of the package is the body: it measures, judges candidates, enforces the hard caps and the safety envelope,
and applies what it is told. The brain decides how to search, and sees the session only as the observations named
here; it imports the contracts and nothing of the body, so a brain can be replaced, or rewritten by the RSI, without
touching what keeps the session honest. Every answer goes back through the body's caps before it is applied, and a
brain that cannot answer leaves the body to keep the status quo.

:class:`CallSite` names every decision point the contract plans; a brain lists the ones it answers in
:attr:`Brain.call_sites`. :data:`BRAIN_CONTRACT_VERSION` moves its major part on a change a brain written against the
old one cannot follow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from hyperloom_multibranch.contracts.brain_brief import SpecialistFacts, SpecialistVoice
from hyperloom_multibranch.contracts.llm import Completion

BRAIN_CONTRACT_VERSION = "0.2"
#: The ``policy_versions`` key naming the brain a session ran.
BRAIN_KEY = "brain"
#: The ``policy_versions`` key of the judge prompt's version.
JUDGE_PROMPT_KEY = "judge_prompt"


class CallSite(str, Enum):
    """Where the body asks the brain: at EXPLORE's start, every judge round, as a branch launches, on a fork or repair
    request, every integration round, as FINAL fixes its candidates and at CLOSE."""

    SESSION_PLAN = "session_plan"
    JUDGE_ROUND = "judge_round"
    BRANCH_BRIEF = "branch_brief"
    FORK_REQUEST = "fork_request"
    INTEGRATE = "integrate"
    FINAL_PICK = "final_pick"
    CLOSE_LEARN = "close_learn"


@dataclass(frozen=True)
class JudgeFacts:
    """The body's numbers a judge has to know to steer within them: the tests judged since a commit at which a
    specialist is told to reconsider (``stall_tests``), the judged explore tests a specialist branch's grace lasts
    without a playbook (``grace_tests``), how long a repair branch is protected and how long it may run at most, how
    many bars a merge's gain must reach to count as an event, and the share shift of the largest op that makes a
    bottleneck move, in percentage points."""

    stall_tests: int
    grace_tests: int
    repair_protected_seconds: float
    repair_branch_seconds: float
    big_gain_bars: int
    bottleneck_shift_pp: int


@dataclass(frozen=True)
class BrainSetup:
    """What the body hands a brain as it is built: the numbers its judge steers within, and the words its specialist
    prompts cite."""

    judge: JudgeFacts
    specialist: SpecialistFacts


@dataclass(frozen=True)
class JudgeObservation:
    """What a judge round sees: the session ``state`` as plain JSON values (the roster with each branch's progress,
    holds and queued tests, the caps, the budget and its spend, the events since the last round), the digest of
    similar archived sessions (``history``), and, in a session that plays a scheduling playbook, where a new branch
    could open (``open_areas``) and the ``playbook`` rendered; the last three are empty when there is none."""

    state: Mapping[str, object]
    history: str = ""
    open_areas: str = ""
    playbook: str = ""

    @property
    def plays_playbook(self) -> bool:
        return bool(self.playbook)


@dataclass(frozen=True)
class KOrder:
    branch: str
    k: int


@dataclass(frozen=True)
class StopOrder:
    branch: str
    reason: str


@dataclass(frozen=True)
class JudgeAnswer:
    """A judge round's answer before the body's caps: how many branches to open and at which K, the op each opened
    branch should speed up (``""`` leaves it to the default focus), the K orders, the stops, and a non-empty
    ``session_stop`` when the brain asks for a soft stop of the session."""

    open_count: int = 0
    open_k: int = 1
    focus: tuple[str, ...] = ()
    k: tuple[KOrder, ...] = ()
    stop: tuple[StopOrder, ...] = ()
    session_stop: str = ""


@dataclass(frozen=True)
class Exchange:
    """One model call a brain made: what it asked, what came back, and why it gave no answer, ``""`` when it did."""

    prompt: str
    reply: str
    error: str = ""


@dataclass(frozen=True)
class JudgeRound:
    """A judge round's outcome: the ``answer``, ``None`` to keep the status quo, and every model call behind it, in
    order, which the body records."""

    answer: JudgeAnswer | None
    exchanges: tuple[Exchange, ...] = ()


class Brain(Protocol):
    """A brain the body loads: its ``name``, the contract version it was written against, the call sites it answers,
    the versions it records in the session's manifest fingerprint, its answers, and the ``specialist`` voice every
    branch's specialist is told through (:attr:`CallSite.BRANCH_BRIEF`).

    The body calls a site only when the brain lists it. A call may raise only what its ``llm`` raises other than
    :class:`~hyperloom_multibranch.contracts.llm.CompletionFailed`, which the brain turns into no answer.
    """

    @property
    def name(self) -> str: ...

    @property
    def contract_version(self) -> str: ...

    @property
    def call_sites(self) -> frozenset[CallSite]: ...

    @property
    def specialist(self) -> SpecialistVoice: ...

    def versions(self, *, plays_playbook: bool) -> Mapping[str, str]: ...

    async def judge_round(self, observation: JudgeObservation, llm: Completion) -> JudgeRound: ...


__all__ = [
    "BRAIN_CONTRACT_VERSION",
    "BRAIN_KEY",
    "JUDGE_PROMPT_KEY",
    "Brain",
    "BrainSetup",
    "CallSite",
    "Exchange",
    "JudgeAnswer",
    "JudgeFacts",
    "JudgeObservation",
    "JudgeRound",
    "KOrder",
    "StopOrder",
]

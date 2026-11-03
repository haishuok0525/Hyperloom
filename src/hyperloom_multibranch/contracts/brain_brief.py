# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brain's ``branch_brief`` call site: what a branch's specialist is told, and what the body hands it to say so.

The body decides when a specialist is told something (a launch, a wake, a nudge, a resume); the brain's
:class:`SpecialistVoice` decides what it is told. The body's own words a specialist must use or recognize (the
runtime's notice prefix, the re-root prefix, the repro case's file, the bash safety rule, the fake gains the toolbox
refuses, the session's directions) reach the brain as :class:`SpecialistFacts`, and every launch, wake and resume as
plain values, so the brain imports none of the body's records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from hyperloom_multibranch.contracts.branch_log import BranchLogType

#: The ``prompt_versions`` keys of the specialist's role and of its launch and session messages.
SPECIALIST_ROLE_KEY = "specialist_role"
SPECIALIST_LAUNCH_KEY = "specialist_launch"


@dataclass(frozen=True)
class FakeGain:
    """A gain the toolbox refuses as fake: how it shows, and the code it is refused with."""

    shows_as: str
    refusal_code: str


@dataclass(frozen=True)
class SpecialistFacts:
    """The body's words a specialist's prompts cite: the prefix of the runtime's branch-log notices (``notice``), the
    one a specialist's entry opens with to re-root its branch (``reroot``), the repro case's script (``case_script``),
    the rule that keeps a bash command from killing what it must not (``bash_safety``), the native build files a
    change may not touch (``native_refused``), the fake gains the toolbox refuses and the session's directions, each
    named by what its text opens with before a colon."""

    notice: str
    reroot: str
    case_script: str
    bash_safety: str
    native_refused: tuple[str, ...]
    fake_gains: tuple[FakeGain, ...]
    directions: tuple[str, ...]


@dataclass(frozen=True)
class RoleAsk:
    """What a specialist's role is built for: the ``workload`` the session optimizes, each reference source's name and
    the directory it was copied to, how many dev sandboxes the session lends at once (0: the specialist has no GPU),
    and each native tree its worktrees hold, by component, as its path in the upstream repository."""

    workload: str
    references: Mapping[str, str] = field(default_factory=dict)
    dev_leases: int = 0
    native: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LogLine:
    """One branch-log entry as a prompt shows it; ``restart`` marks the runtime's notice that a session ended and the
    next one starts, which brings the next session nothing its own first message does not."""

    entry_id: str
    kind: BranchLogType
    content: str
    request_ids: tuple[str, ...] = ()
    supersedes: str = ""
    restart: bool = False


@dataclass(frozen=True)
class Arrival:
    """One inbox item that woke a branch: its kind, what it points at and its summary."""

    kind: str
    pointer: str
    summary: str


@dataclass(frozen=True)
class RepairBrief:
    """The blocked candidate a repair branch was opened for: its first test, the branch that asked and the commit it
    measured on, its gain and what held it back (none: the accuracy gate), its intent, its change as a JSON delta and
    what the asking specialist found."""

    candidate: str
    from_branch: str
    parent_commit: str
    gain_pct: float
    held_back_by: tuple[str, ...]
    target: str
    mechanism: str
    layer: str
    delta: str
    diagnosis: str


@dataclass(frozen=True)
class LaunchAsk:
    """Where one session of a branch's specialist starts: the branch, the commit it was started at, its head, how many
    tests it may have running at once, its worktree, its direction, brief and repair (empty or ``None`` when it has
    none), its whole log, oldest first, and how the previous session ended (``""`` for the first)."""

    branch: str
    start_commit: str
    head: str
    k: int
    worktree: str
    direction: str = ""
    brief: str = ""
    repair: RepairBrief | None = None
    log: tuple[LogLine, ...] = ()
    last_end: str = ""


@dataclass(frozen=True)
class ResumeAt:
    """Where a crashed or idle CLI's conversation goes on: why it stopped, and the head its worktree is at, ``kept``
    as the CLI left it or reset to that head."""

    reason: str
    head: str
    worktree: str
    kept: bool


@dataclass(frozen=True)
class IntegrationMove:
    """The integration head a conversation's last finished turn saw, and the one it goes on at."""

    seen: str
    now: str


class SpecialistVoice(Protocol):
    """What a branch's specialist is told: its role once, at every launch where its branch stands, and between turns
    what woke it, why it is nudged, or why its conversation goes on; ``versions`` names their texts for the session's
    manifest fingerprint, under :data:`SPECIALIST_ROLE_KEY` and :data:`SPECIALIST_LAUNCH_KEY`."""

    def role(self, ask: RoleAsk) -> str: ...

    def launch(self, ask: LaunchAsk) -> str: ...

    def integration(self, holdings: str) -> str: ...

    def wake(self, arrivals: Sequence[Arrival], notices: Sequence[LogLine], *, integration: str = "") -> str: ...

    def resume(self, at: ResumeAt, arrivals: Sequence[Arrival], notices: Sequence[LogLine]) -> str: ...

    def idle_resume(
        self, at: ResumeAt, integration: IntegrationMove, arrivals: Sequence[Arrival], notices: Sequence[LogLine]
    ) -> str: ...

    def lease_wake(self) -> str: ...

    def wait_wake(self, tests: Sequence[str], profiles: Sequence[str]) -> str: ...

    def nudge(self) -> str: ...

    def versions(self) -> Mapping[str, str]: ...


__all__ = [
    "SPECIALIST_LAUNCH_KEY",
    "SPECIALIST_ROLE_KEY",
    "Arrival",
    "FakeGain",
    "IntegrationMove",
    "LaunchAsk",
    "LogLine",
    "RepairBrief",
    "ResumeAt",
    "RoleAsk",
    "SpecialistFacts",
    "SpecialistVoice",
]

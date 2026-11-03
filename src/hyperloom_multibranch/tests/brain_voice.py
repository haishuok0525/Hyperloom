# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The default brain's specialist voice, called on the body's records as the body calls it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from hyperloom_multibranch.contracts.blackboard.records import BranchLogRecord, InboxItem
from hyperloom_multibranch.contracts.brain_brief import ResumeAt
from hyperloom_multibranch.contracts.sources import NativeTree
from hyperloom_multibranch.scheduling.brain_link import load_brain
from hyperloom_multibranch.specialist.brief_views import arrivals, launch_ask, log_lines, role_ask
from hyperloom_multibranch.specialist.runtime import SessionStart

VOICE = load_brain().specialist
Resume = ResumeAt


def role_prompt(
    workload: str,
    references: Mapping[str, Path] | None = None,
    *,
    dev_leases: int = 0,
    native: Mapping[str, NativeTree] | None = None,
) -> str:
    return VOICE.role(role_ask(workload, references, dev_leases=dev_leases, native=native))


def launch_prompt(
    start: SessionStart, *, head: str, worktree: str, recent: Sequence[BranchLogRecord], last_end: str = ""
) -> str:
    return VOICE.launch(launch_ask(start, head=head, worktree=worktree, recent=recent, last_end=last_end))


def prompt_versions() -> dict[str, str]:
    return dict(VOICE.versions())


def wake_message(items: Sequence[InboxItem], notices: Sequence[BranchLogRecord], *, integration: str = "") -> str:
    return VOICE.wake(arrivals(items), log_lines(notices), integration=integration)


def resume_message(at: ResumeAt, items: Sequence[InboxItem], notices: Sequence[BranchLogRecord]) -> str:
    return VOICE.resume(at, arrivals(items), log_lines(notices))


def nudge_message() -> str:
    return VOICE.nudge()

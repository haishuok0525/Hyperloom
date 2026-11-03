# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The body's records as the brain's specialist voice sees them (:mod:`hyperloom_multibranch.contracts.brain_brief`).

A branch-log record becomes a :class:`LogLine` marked when it is the runtime's restart notice, an inbox item an
:class:`Arrival`, a repair request a :class:`RepairBrief` with its change as sorted JSON, and a session start with its
branch's head, worktree and log a :class:`LaunchAsk`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from hyperloom_multibranch.contracts.blackboard.records import BranchLogRecord, InboxItem
from hyperloom_multibranch.contracts.brain_brief import Arrival, LaunchAsk, LogLine, RepairBrief, RoleAsk
from hyperloom_multibranch.contracts.fork import RepairRequest
from hyperloom_multibranch.contracts.sources import NativeTree
from hyperloom_multibranch.specialist.runtime import SessionStart, restart_notice


def role_ask(
    workload: str,
    references: Mapping[str, Path] | None = None,
    *,
    dev_leases: int = 0,
    native: Mapping[str, NativeTree] | None = None,
) -> RoleAsk:
    """The role of a session that optimizes ``workload``, given ``references``, lending ``dev_leases`` dev sandboxes
    at once and holding the ``native`` trees in its worktrees."""
    return RoleAsk(
        workload,
        references={name: str(path) for name, path in (references or {}).items()},
        dev_leases=dev_leases,
        native={component: tree.root.prefix for component, tree in (native or {}).items()},
    )


def log_lines(records: Sequence[BranchLogRecord]) -> tuple[LogLine, ...]:
    return tuple(_line(record) for record in records)


def _line(record: BranchLogRecord) -> LogLine:
    entry = record.entry
    return LogLine(
        record.entry_id,
        entry.kind,
        entry.content,
        request_ids=entry.request_ids,
        supersedes=entry.supersedes or "",
        restart=restart_notice(entry.content),
    )


def arrivals(items: Sequence[InboxItem]) -> tuple[Arrival, ...]:
    return tuple(Arrival(item.kind.value, item.pointer, item.summary) for item in items)


def repair_brief(request: RepairRequest) -> RepairBrief:
    intent, judgement = request.intent, request.judgement
    return RepairBrief(
        candidate=judgement.request_ids[0],
        from_branch=intent.branch,
        parent_commit=intent.parent_commit,
        gain_pct=judgement.gain_pct,
        held_back_by=tuple(judgement.refused_by),
        target=intent.target,
        mechanism=intent.mechanism,
        layer=intent.layer.value,
        delta=json.dumps(request.candidate_delta.to_document(), sort_keys=True),
        diagnosis=request.diagnosis,
    )


def launch_ask(
    start: SessionStart, *, head: str, worktree: str, recent: Sequence[BranchLogRecord], last_end: str = ""
) -> LaunchAsk:
    """One session of the specialist ``start`` runs, at ``head`` in ``worktree``, its whole branch log ``recent``;
    ``last_end`` says how the previous one ended."""
    task = start.task
    return LaunchAsk(
        branch=start.branch,
        start_commit=start.start_commit,
        head=head,
        k=start.k,
        worktree=worktree,
        direction=task.direction,
        brief=task.brief,
        repair=None if task.repair is None else repair_brief(task.repair),
        log=log_lines(recent),
        last_end=last_end,
    )


__all__ = ["arrivals", "launch_ask", "log_lines", "repair_brief", "role_ask"]

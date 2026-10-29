# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The orchestration prompt: which sections the model reads, in what order, and which half of the prompt holds each.

``context`` holds what holds for the session, so a backend may cache it; ``turn`` holds what moves every tick. The
body reports each section's facts in a view; what to show of them, and where, is decided here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from hyperloom.brain.orchestrate.advisories import (
    acceptance_threshold,
    bottleneck_redirect,
    discarded_escalate_hint,
    specialist_findings,
)
from hyperloom.brain_api import Port, ToolCall
from hyperloom.common.prompt_safety import fit_to_budget

# Ceiling on the inbox tail one turn renders; what it leaves out waits for the next turn.
INBOX_PROMPT_CHARS = 40_000
# Minutes left at which the model is warned off new explore rounds.
_DEADLINE_WARNING_MIN = 5.0
_DEADLINE_WARNING = (
    "WARNING: < 5 min remaining. Avoid new `explore` rounds "
    "(which bench every variant on the stack) — they will "
    "likely be cut by the deadline. The Coordinator will "
    "auto-enqueue `report` when the session closes."
)


@dataclass(frozen=True)
class OrchestrationPrompt:
    """One orchestration prompt, split where a backend may cache it."""

    context: str
    turn: str

    @property
    def text(self) -> str:
        """The whole prompt as one string."""
        return f"{self.context}\n{self.turn}"


#: A context section: the heading it goes under, or ``None`` for a block that carries its own, and what renders it
#: from the view. Row order is the order the model reads the sections in, so it is content.
_AdvisoryRow = tuple[str | None, Callable[[Mapping[str, Any]], str]]

_ADVISORY_ROWS: tuple[_AdvisoryRow, ...] = (
    ("=== Warm start (Recipe KB T0) ===", lambda v: v["advisories"]["warm_start"]),
    ("=== Current gaps ===", lambda v: v["advisories"]["gaps"]),
    (None, lambda v: specialist_findings(v["specialist_rounds"])),
    ("=== External target gap (advisory) ===", lambda v: v["advisories"]["target_gap"]),
    ("=== Specialist proposal scores (advisory) ===", lambda v: v["advisories"]["proposal_scores"]),
    ("=== Priors-match (advisory ordering) ===", lambda v: v["advisories"]["priors_match"]),
    ("=== Intervention mix (telemetry) ===", lambda v: v["advisories"]["intervention_mix"]),
    ("=== Bottleneck redirect (advisory) ===", lambda v: bottleneck_redirect(v["bottleneck"])),
    ("=== Acceptance threshold (advisory) ===", lambda v: acceptance_threshold(v["acceptance"])),
    ("=== Discarded escalation hint (advisory) ===", lambda v: discarded_escalate_hint(v["discarded_escalate_hint"])),
)


def _advisory_sections(view: Mapping[str, Any]) -> list[str]:
    """The advisories that produced something, each under its heading."""
    sections: list[str] = []
    for heading, produce in _ADVISORY_ROWS:
        block = produce(view)
        if not block:
            continue
        if heading:
            sections.append(heading)
        sections.append(block)
    return sections


def _time_budget(budget: Mapping[str, Any] | None) -> list[str]:
    """The time-budget lines, and the deadline warning once little is left outside the closing phase."""
    if budget is None:
        return []
    remaining = budget["remaining_min"]
    lines = [
        "=== Time budget ===",
        f"elapsed={budget['elapsed_min']:.1f}min  remaining={remaining:.1f}min  "
        f"budget={budget['budget_min']}min  "
        f"closing_phase={budget['closing_phase']}",
    ]
    if remaining <= _DEADLINE_WARNING_MIN and not budget["closing_phase"]:
        lines.append(_DEADLINE_WARNING)
    return lines


def _moving_sections(view: Mapping[str, Any]) -> list[str]:
    """The denial summary, proposal queue and plateau pair.

    A denial, a proposal leaving the queue or a plateau count lands between two ticks, so these follow the ledger into
    the turn: the cached context ends above them and holds for the session.
    """
    sections: list[str] = []
    if view["denial_summary"]:
        sections.append(view["denial_summary"])
    queue = view["proposal_queue"]
    if queue is not None:
        # The predictor's rows lead the queue, so its block comes first, under the same section.
        untested = "\n\n".join(block for block in (queue["predictor"], queue["untested"]) if block)
        if untested:
            sections.append("=== Untested proposals (current cycle) ===")
            sections.append(untested)
        if queue["kb_evidence"]:
            sections.append(queue["kb_evidence"])
    if view["plateau"]:
        sections.append("=== Plateau advisory ===")
        sections.append(view["plateau"])
        if view["trajectory_digest"]:
            sections.append("=== Trajectory review (advisory) ===")
            sections.append(view["trajectory_digest"])
    return sections


async def _inbox_sections(inbox: Sequence[Mapping[str, Any]], port: Port) -> list[str]:
    """The inbox tail, oldest first, within its budget; the body is told the last message shown."""
    lines = fit_to_budget([f"  {entry['line']}" for entry in inbox], INBOX_PROMPT_CHARS)
    if not lines:
        return ["=== Inbox for orchestration ===", "(no new messages)"]
    last_seq = int(inbox[len(lines) - 1]["seq"])
    await port.call(ToolCall("inbox.rendered", {"seq": last_seq}))
    sections = ["=== Inbox for orchestration (newest last) ===", *lines]
    if len(lines) < len(inbox):
        sections.append(f"  (+{len(inbox) - len(lines)} newer messages after seq={last_seq} held for the next turn)")
    return sections


async def compose_prompt(view: Mapping[str, Any], port: Port) -> OrchestrationPrompt:
    """The orchestration prompt for the session the body reported in ``view``."""
    context: list[str] = [f"SESSION_DIR={view['session_dir']}"]
    turn: list[str] = []
    if view["phase"]:
        turn += ["=== Phase ===", view["phase"]]
    turn += ["=== Mission progress ===", view["mission"]]
    turn += _time_budget(view["time_budget"])
    turn += ["=== Tick ===", view["tick"]]
    # The session ledger moves with every action that lands, so it belongs to the turn: carried in the context it
    # would rewrite the cached prefix the stable blocks below sit in, on nearly every tick.
    turn += ["=== Shared session state ===", view["shared_state"], f"target_gap_pct={view['target_gap_pct']}"]
    # Not wrapped, unlike the advisory blocks: a latency budget changes what a KEEP means, so losing it silently would
    # have the model route as if the session were unconstrained.
    context += view["constraints"]
    context += ["=== Resource pools ===", view["resource_pools"]]
    context += _advisory_sections(view)
    turn += _moving_sections(view)
    turn += await _inbox_sections(view["inbox"], port)
    return OrchestrationPrompt(context="\n".join(context), turn="\n".join(turn))

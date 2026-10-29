# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The advisories the orchestration model is nudged with, each rendered from the facts the body reports."""

from __future__ import annotations

import json
from operator import itemgetter
from typing import Any, Mapping, Sequence

from hyperloom.common.prompt_safety import fit_to_budget

# Ceilings for the findings sections, which grow with the session; uncapped, a long run outgrows the context window.
FINDINGS_PROMPT_CHARS = 24_000
QUESTIONS_PROMPT_CHARS = 6_000


def specialist_findings(rounds: Sequence[Mapping[str, Any]]) -> str:
    """Render persisted specialist findings, any domain, most recent first.

    Executable proposals are not rendered here: they go through
    ``=== Untested proposals (current cycle) ===`` alongside every other
    domain's, which also drops the ones already benched.

    Rows are ordered by recency rather than by the round's self-reported
    ``confidence``: that field is an audit record of what the specialist
    claimed, never an input to a decision here. Each list keeps its newest
    entries within its character ceiling and counts the rest.
    """
    findings: list[str] = []
    questions: list[str] = []
    seen_questions: set[str] = set()
    for row in reversed(rounds):
        domain_label = str(row.get("domain") or "").strip()
        for finding in row.get("new_findings") or []:
            text = json.dumps(finding, sort_keys=True) if isinstance(finding, dict) else str(finding)
            findings.append(f"- [{domain_label}] {text}")
        for question in row.get("residual_questions") or []:
            text = str(question).strip()
            if text and text not in seen_questions:
                seen_questions.add(text)
                questions.append(f"- [{domain_label}] {text}")
    if not findings and not questions:
        return ""

    lines = ["=== Specialist findings ==="]
    for title, items, budget_chars in (
        ("Findings", findings, FINDINGS_PROMPT_CHARS),
        ("Residual questions", questions, QUESTIONS_PROMPT_CHARS),
    ):
        if not items:
            continue
        shown = fit_to_budget(items, budget_chars)
        lines.append(f"{title}:")
        lines.extend(shown)
        if len(shown) < len(items):
            lines.append(f"(+{len(items) - len(shown)} older omitted)")
    return "\n".join(lines)


def _bottleneck_lines(bottleneck: Mapping[str, Any], saturated: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    if bottleneck["pending_switch"]:
        lines.append(
            "The previous macro-cycle plateaued; redirect this cycle to a "
            "different bottleneck instead of re-mining the exhausted one."
        )
    if saturated:
        lines.append("Roofline ceiling signal: one or more lever families are saturated; deprioritize them.")
        lines.extend(
            f"  saturated_domain={domain} direction={row.get('direction')} "
            f"within={row.get('within_pct')}% threshold={row.get('threshold_pct')}%"
            for domain, row in sorted(saturated.items())
        )
    if bottleneck["last_cycle_bottleneck"]:
        lines.append(f"  plateaued_bottleneck={bottleneck['last_cycle_bottleneck']} (avoid re-targeting)")
    if bottleneck["current_top"]:
        lines.append(f"  current_top_bottleneck={bottleneck['current_top']}")
    return lines


def bottleneck_redirect(bottleneck: Mapping[str, Any] | None) -> str:
    """The cyclic bottleneck-redirect advisory; ``bottleneck`` is reported only in the optimisation phase."""
    if bottleneck is None:
        return ""
    saturated = {k: v for k, v in bottleneck["saturated"].items() if v.get("saturated")}
    if not bottleneck["pending_switch"] and not saturated:
        return ""
    lines = _bottleneck_lines(bottleneck, saturated)
    shift = bottleneck["shift"]
    if shift.get("from") or shift.get("to"):
        lines.append(
            f"  bottleneck_shift: {shift.get('from') or 'unknown'} → {shift.get('to') or 'unknown'} "
            f"(within_delta={shift.get('within_delta')} gap_delta={shift.get('gap_delta')})"
        )
    dominant = bottleneck["dominant"]
    if dominant["direction"]:
        hint = dominant["hint"]
        suffix = f" → suggested specialist domain={hint[0]} tag={hint[1]}" if hint else ""
        lines.append(f"  dominant_direction={dominant['direction']} ({dominant['pct']:.1f}%){suffix}")
    lines.append(f"  macro_cycle={bottleneck['macro_cycle']}")
    lines.append("Advisory only: pick the domain/tag yourself; this nudges focus, it does not gate dispatch.")
    return "\n".join(lines)


def acceptance_threshold(acceptance: Mapping[str, Any]) -> str:
    """The decaying acceptance bar, with the prior measured gains on either side of it as evidence."""
    keep = acceptance["keep_threshold"]
    cycle = acceptance["macro_cycle"]
    if cycle < 1:
        return ""
    stable = keep / 2.0
    above_bar: list[tuple[str, float]] = []
    below_bar: list[tuple[str, float]] = []
    for e in acceptance["entries"]:
        try:
            g = float(e.get("gain_pct"))
        except (TypeError, ValueError):
            continue
        name = str(e.get("name") or e.get("fingerprint") or "")[:48]
        (above_bar if g >= keep else below_bar).append((name, g))
    above_bar.sort(key=itemgetter(1), reverse=True)
    below_bar.sort(key=itemgetter(1), reverse=True)
    lines: list[str] = [
        f"Current acceptance bar (macro_cycle={cycle}): KEEP>={keep:.2f}% stack_stable>={stable:.2f}%.",
        "Historical results are evidence only — any fingerprint may be re-proposed.",
    ]
    if above_bar:
        lines.append("Prior results above the bar (consider re-proposing if conditions changed):")
        lines.extend(f"  {name}: prior gain {g:+.2f}% >= {keep:.2f}%" for name, g in above_bar[:8])
    if below_bar:
        lines.append("Prior results below the bar (reference):")
        lines.extend(f"  {name}: prior gain {g:+.2f}% < {keep:.2f}%" for name, g in below_bar[:5])
    return "\n".join(lines)


def discarded_escalate_hint(discarded: Mapping[str, str]) -> str:
    """The advisory for an escalate_strategy_change hint a phase transition it did not drive dropped."""
    hint = discarded["hint"]
    if not hint:
        return ""
    return (
        f"ADVISORY: your escalate_strategy_change hint '{hint}' (at {discarded['ts']}) was discarded "
        "because a phase transition it did not drive fired before "
        "it could be consumed. Re-emit escalate_strategy_change if still needed."
    )

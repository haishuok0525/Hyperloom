# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Advisory specialist-domain focus for one macro-cycle (M-01)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class _FocusScoreboard:
    """Per-domain cycle-focus scores and the evidence each one rests on."""

    def __init__(self, domains: Iterable[str]) -> None:
        self.scores: dict[str, float] = dict.fromkeys(domains, 0.0)
        self.reasons: dict[str, list[str]] = {domain: [] for domain in self.scores}

    def admit(self, domain: str) -> None:
        """Make ``domain`` selectable at its current score."""
        self.scores.setdefault(domain, 0.0)
        self.reasons.setdefault(domain, [])

    def credit(self, domain: str, delta: float, reason: str) -> None:
        """Move ``domain``'s score by ``delta`` and record why."""
        self.admit(domain)
        self.scores[domain] += delta
        self.reasons[domain].append(reason)

    def leader(self) -> str:
        """The highest-scoring domain, ties broken by name; empty when nothing is scored."""
        if not self.scores:
            return ""
        return max(self.scores.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _score_shift(board: _FocusScoreboard, shift: Mapping[str, Any]) -> None:
    to_domain = str(shift.get("to_domain") or "").strip()
    if to_domain:
        board.credit(to_domain, 5.0, f"matches current bottleneck shift to {shift.get('to') or to_domain}")


def _score_saturation(board: _FocusScoreboard, saturated: Any) -> None:
    if not isinstance(saturated, dict):
        return
    for domain, row in saturated.items():
        if not isinstance(row, dict):
            continue
        name = str(domain or row.get("domain") or "").strip()
        if not name:
            continue
        if bool(row.get("saturated")):
            board.credit(name, -100.0, f"saturated at {row.get('within_pct')}% within roofline; deprioritized")
        else:
            board.credit(name, 1.0, "not saturated in latest roofline snapshot")


def _score_history(board: _FocusScoreboard, log_rows: list[Any]) -> None:
    for row in log_rows:
        if not isinstance(row, dict):
            continue
        domain = str(row.get("focus") or "").strip()
        if not domain:
            continue
        board.admit(domain)
        gain_delta = row.get("gain_delta")
        if isinstance(gain_delta, (int, float)):
            board.credit(
                domain,
                max(-2.0, min(3.0, float(gain_delta))),
                f"historical cycle gain_delta={float(gain_delta):+.2f}%",
            )


def _score_unexplored(board: _FocusScoreboard, domains: list[str], log_rows: list[Any]) -> None:
    tried = {str(row.get("focus") or "") for row in log_rows if isinstance(row, dict)}
    for domain in domains:
        if domain not in tried:
            board.credit(domain, 1.5, "exploration bonus: not yet used as cycle focus")


def _score_negatives(board: _FocusScoreboard, counts: Mapping[str, Any]) -> None:
    for domain, count in counts.items():
        penalty = min(4.0, 0.5 * float(count))
        board.credit(str(domain), -penalty, f"recent negative ledger count={count} penalty={penalty:.1f}")


def plan_cycle_focus(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Pick an advisory specialist-domain focus from the cycle's scored evidence.

    ``facts`` carries ``domains``, ``shift``, ``saturated``, ``log_rows``, ``negative_counts``,
    ``macro_cycle``, ``bottleneck_fallback`` and ``gain_at_start``.
    """
    domains = list(facts["domains"])
    board = _FocusScoreboard(domains)
    shift = facts.get("shift") or {}
    sat = facts.get("saturated") or {}
    log_rows = list(facts.get("log_rows") or [])
    _score_shift(board, shift)
    _score_saturation(board, sat)
    _score_history(board, log_rows)
    _score_unexplored(board, domains, log_rows)
    _score_negatives(board, facts.get("negative_counts") or {})
    focus = board.leader() or "freeform_specialist"
    rationale_bits = board.reasons.get(focus) or ["fallback focus; no stronger cycle-level evidence"]
    return {
        "cycle": int(facts.get("macro_cycle") or 0),
        "focus": focus,
        "score": round(float(board.scores.get(focus, 0.0)), 3),
        "rationale": "; ".join(rationale_bits[:4]),
        "bottleneck_at_start": str(shift.get("to") or facts.get("bottleneck_fallback") or ""),
        "saturated_at_start": sorted(
            str(k)
            for k, v in (sat.items() if isinstance(sat, dict) else [])
            if isinstance(v, dict) and bool(v.get("saturated"))
        ),
        "gain_at_start": float(facts.get("gain_at_start") or 0.0),
        "gain_delta": None,
    }

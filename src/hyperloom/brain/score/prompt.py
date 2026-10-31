# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prompt and parse helpers for advisory multi-model proposal scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from hyperloom.common.coerce import to_float

#: Soft cap on proposals fed to each model.
MAX_PROPOSALS_SCORED: int = 16
MAX_FIELD_CHARS: int = 600

SCORING_INSTRUCTIONS = """
You are scoring candidate serving-config variants proposed by a perf
specialist for an LLM-inference optimization run on AMD GPUs.

For EACH proposal below, output a single composite score from 0 to 10:
how likely / how much this variant improves serving THROUGHPUT for the
stated gap, grounded ONLY in the evidence shown. 10 = very likely a
large win; 0 = irrelevant or likely harmful. Be calibrated, not generous.

Output ONLY one compact JSON object, no prose, no markdown fence needed:

{"scores": {"<proposal_id>": {"score": <0-10 number>, "reason": "<= 15 words"}}}

Rules:
- One entry per proposal, keyed by its exact "id" (not the human-readable name).
- "reason" MUST be <= 15 words. Keep it terse to keep the reply short.
- Do not add keys other than "score" and "reason".
""".strip()


@dataclass(frozen=True)
class ScoringProposal:
    """One proposal prepared for multi-model scoring."""

    stable_id: str
    output_name: str
    label_name: str
    proposal: dict[str, Any]


def clip(value: Any, *, limit: int = MAX_FIELD_CHARS) -> str:
    """Stringify a value and truncate it to a maximum length."""
    s = "" if value is None else str(value)
    return s if len(s) <= limit else (s[:limit] + "…")


def coerce_score(raw: Any) -> float | None:
    """Coerce a model-emitted score into a clamped [0, 10] float."""
    val = to_float(raw)
    return None if val is None else max(0.0, min(10.0, val))


def prepare_scoring_proposals(proposals: list[dict[str, Any]]) -> list[ScoringProposal]:
    """Assign stable ids and reject duplicate display names."""
    seen_labels: set[str] = set()
    out: list[ScoringProposal] = []
    for i, proposal in enumerate(proposals):
        raw_name = proposal.get("name")
        if raw_name is not None and str(raw_name).strip():
            output_name = str(raw_name)
            label_name = output_name.strip()
        else:
            output_name = f"proposal_{i}"
            label_name = output_name
        if label_name in seen_labels:
            raise ValueError(f"duplicate proposal name: {label_name!r}")
        seen_labels.add(label_name)
        out.append(
            ScoringProposal(
                stable_id=f"proposal_{i}",
                output_name=output_name,
                label_name=label_name,
                proposal=proposal,
            )
        )
    return out


def build_scoring_prompt(*, gap: dict[str, Any], scoring_entries: list[ScoringProposal]) -> str:
    """Build ONE group-scoring prompt covering every proposal."""
    lines: list[str] = ["=== Gap ==="]
    lines.append(f"domain: {clip(gap.get('domain'), limit=80)}")
    lines.append(f"gap_canonical_id: {clip(gap.get('gap_canonical_id'), limit=160)}")
    symptom = gap.get("gap_symptom") or gap.get("summary")
    if symptom:
        lines.append(f"symptom: {clip(symptom)}")
    evidence = gap.get("gap_evidence")
    if evidence:
        lines.append(f"evidence: {clip(json.dumps(evidence, sort_keys=True))}")

    lines.append("")
    lines.append("=== Proposals to score ===")
    for entry in scoring_entries:
        p = entry.proposal
        lines.append(f"- id: {entry.stable_id}")
        lines.append(f"  name: {entry.label_name}")
        if p.get("extra_args"):
            lines.append(f"  extra_args: {clip(p.get('extra_args'))}")
        if p.get("extra_envs"):
            lines.append(f"  extra_envs: {clip(json.dumps(p.get('extra_envs'), sort_keys=True))}")
        if p.get("reason"):
            lines.append(f"  reason: {clip(p.get('reason'))}")
        if p.get("kb_evidence"):
            lines.append(f"  kb_evidence: {clip(json.dumps(p.get('kb_evidence'), sort_keys=True))}")
    return "\n".join(lines)


def normalise_model_scores(
    parsed: dict[str, Any],
    *,
    scoring_entries: list[ScoringProposal],
) -> dict[str, dict[str, Any]]:
    """Project parsed ``{\"scores\": {...}}`` onto known stable ids."""
    out: dict[str, dict[str, Any]] = {}
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        return out
    id_to_name = {entry.stable_id: entry.output_name for entry in scoring_entries}
    known = set(id_to_name)
    for proposal_id, entry in scores.items():
        key = str(proposal_id)
        if key not in known or not isinstance(entry, dict):
            continue
        score = coerce_score(entry.get("score"))
        if score is None:
            continue
        out[id_to_name[key]] = {
            "score": score,
            "reason": clip(entry.get("reason"), limit=160),
        }
    return out

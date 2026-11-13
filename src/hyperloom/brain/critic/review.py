# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Critic behind the contract: candidates are held to the review constraints, the model rules on each, and its
ruling reads back as one verdict per candidate."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from hyperloom.brain_api import (
    Candidate,
    Decision,
    LlmRequest,
    PortError,
    Ports,
    Review,
    ReviewAsk,
    SearchState,
    Verdict,
)
from hyperloom_common.jsonio import extract_last_json_matching

from .constraints import LEVER_ORIENTATION, apply_review_constraints

#: The phase a contract review is held in: specialist variants and their patches, before any is measured.
PHASE = "FRAMEWORK_AGENT"
MAX_COMPLETION_TOKENS = 8192
# One retry at this multiple of the cap when a reply stops at the limit.
TRUNCATION_RETRY_FACTOR = 2
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})
# A diff longer than this reaches the model cut, with the cut marked.
MAX_DIFF_CHARS = 6000
_DECISIONS = {
    "approve": Decision.APPROVE,
    "advise": Decision.APPROVE,
    "reject": Decision.REJECT,
    "redirect": Decision.DEFER,
    "needs_review": Decision.DEFER,
}
# The body's rule descriptors, empty: a contract review carries its rules in ``ReviewAsk.constraints``.
_NO_BODY_RULES: Mapping[str, Any] = {
    "scope_domains": "domains",
    "cross_domain_rules": (),
    "advisory_rule_actions": (),
    "quantitative_claim_rule": {},
}

SYSTEM_PROMPT = """
You are the Critic: the review layer of the inference optimizer. The Coordinator hands you candidates a specialist
proposed for the serving stack in the judge bundle; you return one verdict per candidate before any of them is
measured. You do not run, apply or measure anything.

Verdicts: `approve` (measure it), `advise` (measure it, with a non-blocking concern), `reject` (do not spend a bench
on it), `redirect` (the idea is right, the candidate is not: say what would be), `needs_review` (cannot be judged on
what the bundle holds).

Standards, by lever (`review_constraints.lever_kind` and each proposal's `lever_kind`):
- config: approve when the candidate fits the gap and the workload, is not already on the stack or among the
  attempts with the same outcome, and nothing in the bundle (pitfalls, lessons, attempts) contradicts it. A missing
  KB prior is the cold-start default, not a blocker.
- source_patch: the diff is in the bundle. Approve when its scope matches the stated target, it is grounded in the
  source or profile evidence named in its reason, it does not break correctness or another workload path, and it
  rolls back by reverting the diff. Reject a diff that is malformed, touches unrelated code, or defeats an accuracy
  guard.

Hard rules:
- Do not invent missing context. If `judge_bundle.required_context` is non-empty, return `needs_review`.
- Do not return `reject` or `redirect` from historical claims without evidence from the bundle; cite it in
  `packet_evidence`.
- Do not treat a claimed speedup as a measured one; throughput is measured after your review.
- Attach `predicted_gain_pct` to `approve` and `redirect` verdicts: your estimate of the throughput gain over the
  current best, in percent.
- Every rule in `review_constraints.rules` holds.
""".strip()

REVIEW_OUTPUT_INSTRUCTIONS = """
==== OUTPUT FORMAT (REQUIRED) ====
Reply with EXACTLY ONE JSON object that matches this review schema:

{
  "review_verdicts": [
    {
      "target_proposal_msg_id": "<the proposal's msg_id from the bundle>",
      "verdict": "approve" | "reject" | "redirect" | "advise" | "needs_review",
      "source": "critic" | "critic_unavailable",
      "reasoning": "<short, explicit reasoning>",
      "confidence": "low" | "medium" | "high",
      "predicted_gain_pct": <number or null>,
      "kb_evidence": ["<kb_id>", ...],
      "packet_evidence": ["<dotted.path.in.packet>", ...],
      "risks": [{"severity": "blocker|major|minor", "summary": "..."}],
      "notes": ["..."]
    }
  ]
}

Rules:
- Wrap the JSON in a ```json fenced block. Bare JSON is also accepted.
- Free text outside the JSON is ignored.
- Keep `reasoning`/`notes` to new, decision-relevant points; do not restate the
  proposal or context already in the judge_bundle.
- Emit one verdict object PER proposal in `judge_bundle.proposals`.
==== END OUTPUT FORMAT ====
""".strip()

_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"review_verdicts\"[\s\S]*\})", re.DOTALL)


def extract_review_json(text: str) -> dict[str, Any] | None:
    """Pull the Critic's own ``{"review_verdicts": ...}`` object out of a reply."""
    return extract_last_json_matching(text, "review_verdicts", _BARE_JSON_RE)


def lever_of(candidate: Candidate) -> str:
    """The lever a candidate moves: a source patch when it carries one, else configuration."""
    return "source_patch" if candidate.patches else "config"


def _diff_view(candidate: Candidate) -> str:
    text = "".join(p.diff if p.diff.endswith("\n") else p.diff + "\n" for p in candidate.patches)
    if len(text) <= MAX_DIFF_CHARS:
        return text
    return text[:MAX_DIFF_CHARS] + f"\n... ({len(text) - MAX_DIFF_CHARS} more chars cut)\n"


def _proposal(candidate: Candidate) -> dict[str, Any]:
    lever = lever_of(candidate)
    c = candidate.config
    params: dict[str, Any] = {
        "extra_args": c.extra_server_args,
        "extra_envs": dict(c.extra_envs),
        "remove_args": list(c.remove_args),
        "unset_envs": list(c.unset_envs),
        "args_mode": c.args_mode,
    }
    if candidate.patches:
        params["patch_files"] = [p.path for p in candidate.patches]
        params["diff"] = _diff_view(candidate)
    return {
        "msg_id": candidate.name,
        "action_name": "integrate_patch" if candidate.patches else "explore",
        "provenance": "specialist",
        "params": params,
        "reason": candidate.reason,
        "citations": list(candidate.citations),
        "lever_kind": lever,
        "lever_orientation": LEVER_ORIENTATION.get(lever, ""),
    }


def _attempt(a: Any) -> dict[str, Any]:
    return {
        "name": a.candidate.name,
        "outcome": a.outcome.value,
        "gain_pct": a.gain_pct,
        "note": a.note,
        "lever_kind": lever_of(a.candidate),
        "reason": a.candidate.reason,
    }


def judge_bundle(state: SearchState, ask: ReviewAsk) -> dict[str, Any]:
    """The bundle the Critic rules on: the candidates, what the search knows, and the constraints they answer to."""
    w = state.workload
    bundle: dict[str, Any] = {
        "phase": PHASE,
        "merged_context": {
            "workload": {
                "model": w.model_name,
                "framework": w.framework,
                "framework_version": w.framework_version,
                "gpu_type": w.gpu_type,
                "tp": w.tp,
                "precision": w.precision,
                "conc": w.conc,
                "isl": w.isl,
                "osl": w.osl,
                "benchmark_mode": w.benchmark_mode,
            },
            "baseline_tput": state.baseline_tput,
            "current_tput": state.current_tput,
            "keep_threshold_pct": state.keep_threshold_pct,
            "launch": {"extra_args": state.launch.extra_server_args, "extra_envs": dict(state.launch.extra_envs)},
            "stack": [c.name for c in state.stack],
            "attempts": [_attempt(a) for a in state.attempts],
            "gaps": [
                {"canonical_id": g.canonical_id, "severity": g.severity, "layer": g.layer, "symptom": g.symptom}
                for g in state.gaps
            ],
            "roofline": dict(state.evidence.roofline),
            "runtime_findings": state.evidence.runtime_findings,
        },
        "missing_context": [],
        "required_context": [],
        "proposals": [_proposal(c) for c in ask.candidates],
        "kb_priors_by_proposal": {},
        "kb_read_skipped_reason": "",
        "review_constraints": {"rules": list(ask.constraints)},
        "notes": [*map(dict, state.evidence.pitfalls), *map(dict, state.evidence.lessons)],
    }
    apply_review_constraints(
        bundle,
        phase=PHASE,
        action_verdict_policy={},
        rules=_NO_BODY_RULES,
        proposal_levers=[lever_of(c) for c in ask.candidates],
    )
    return bundle


async def reason(bundle: Mapping[str, Any], ports: Ports, *, seed: int | None = None) -> dict[str, Any]:
    """The Critic model's ruling on ``bundle``.

    Raises:
        PortError: if the model could not be reached, or no reply (after one retry with more room when the first
            stopped at the cap) carried a ruling.
    """
    prompt = (
        f"==== JUDGE BUNDLE ====\n{json.dumps(bundle, ensure_ascii=False, separators=(',', ':'))}\n"
        f"==== END JUDGE BUNDLE ====\n\n{REVIEW_OUTPUT_INSTRUCTIONS}"
    )
    max_tokens = MAX_COMPLETION_TOKENS
    reply = await ports.llm(LlmRequest(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=max_tokens, seed=seed))
    review = extract_review_json(reply.text)
    if review is None and reply.finish_reason.strip().lower() in _TRUNCATED_FINISH_REASONS:
        max_tokens *= TRUNCATION_RETRY_FACTOR
        reply = await ports.llm(LlmRequest(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=max_tokens, seed=seed))
        review = extract_review_json(reply.text)
    if review is None:
        raise PortError(
            "the Critic replied without a parseable review_verdicts JSON "
            f"(chars={len(reply.text)}, finish={reply.finish_reason!r}, max_tokens={max_tokens})",
            retryable=True,
        )
    return review


def _gain(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def verdicts_of(review: Mapping[str, Any], ask: ReviewAsk) -> Review:
    """One verdict per candidate of ``ask``; a candidate the ruling does not cover, or covers with an unknown
    verdict, is deferred."""
    ruled: dict[str, Mapping[str, Any]] = {}
    for item in review.get("review_verdicts") or []:
        if isinstance(item, Mapping) and isinstance(item.get("target_proposal_msg_id"), str):
            ruled.setdefault(item["target_proposal_msg_id"], item)
    out = []
    for c in ask.candidates:
        item = ruled.get(c.name)
        if item is None:
            out.append(Verdict(c.name, Decision.DEFER, reason="the Critic gave no verdict"))
            continue
        word = str(item.get("verdict") or "").strip().lower()
        decision = _DECISIONS.get(word, Decision.DEFER)
        reason = str(item.get("reasoning") or "").strip()
        out.append(
            Verdict(
                c.name,
                decision,
                predicted_gain_pct=_gain(item.get("predicted_gain_pct")),
                reason=f"{word}: {reason}" if reason else word,
            )
        )
    return Review(tuple(out))


async def review_candidates(state: SearchState, ask: ReviewAsk, ports: Ports, *, seed: int | None = None) -> Review:
    """The Critic's verdict on every candidate of ``ask``; raises ``PortError``."""
    return verdicts_of(await reason(judge_bundle(state, ask), ports, seed=seed), ask)

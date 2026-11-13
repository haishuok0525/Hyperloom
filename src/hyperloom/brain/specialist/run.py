# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One specialist dispatch behind the contract: brief it from the search state, run it on the agent port, and read
its candidates back from what it replied and the source edits it left."""

from __future__ import annotations

from typing import Any

from hyperloom.brain_api import AgentRequest, Attempt, Gap, Ports, Proposal, ProposeAsk, SearchState

from .brief import build_specialist_prompts
from .deliver import PROPOSAL_SCHEMAS, candidates_of, read_done
from .domains import DOMAINS, FREEFORM, gap_domain
from .inputs import DELIVERY_REPLY, MODE_PATCH, BriefFacts, BriefInputs

#: The turns a brief allows when the ask sets none.
DEFAULT_MAX_TURNS = 30
#: Prior attempts a brief lists.
MAX_PRIOR_ATTEMPTS = 20
_MODE_RESEARCH = "research"
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def target_gap(state: SearchState, ask: ProposeAsk) -> Gap | None:
    """The gap a dispatch is anchored at: the ask's, else the most severe the state knows (first among equals)."""
    if ask.gap is not None:
        return ask.gap
    ranked = sorted(
        enumerate(state.gaps), key=lambda ig: (_SEVERITY_RANK.get(ig[1].severity.strip().lower(), 3), ig[0])
    )
    return ranked[0][1] if ranked else None


def _config_ref(attempt: Attempt) -> str:
    c = attempt.candidate
    parts = [c.config.extra_server_args.strip()]
    parts += [f"{k}={v}" for k, v in c.config.extra_envs.items()]
    parts += [f"-{a}" for a in c.config.remove_args] + [f"unset {e}" for e in c.config.unset_envs]
    if c.patches:
        parts.append("patch " + ", ".join(p.path for p in c.patches))
    shown = "; ".join(p for p in parts if p)
    return f"{c.name} ({shown})" if shown else c.name


def _metric(tput: float) -> str:
    return f"{tput:.1f} tok/s" if tput > 0 else ""


def brief_inputs(state: SearchState, ask: ProposeAsk) -> BriefInputs | None:
    """The inputs of the dispatch that answers ``ask``, or ``None`` when its domain authors no candidates.

    A domain ask runs that domain's specialist. A freeform ask runs the freeform specialist on the ask's task,
    anchored at :func:`target_gap` and carrying the focus of the domain that gap points at.
    """
    gap = target_gap(state, ask)
    if ask.domain:
        domain = DOMAINS.get(ask.domain)
        if domain is None:
            return None
        scope, focus = "domain", ()
    else:
        domain, scope = FREEFORM, "freeform"
        pointed = gap_domain(gap)
        focus = (pointed,) if pointed else ()
    w = state.workload
    gains = {a.candidate.name: a.gain_pct for a in state.attempts if a.gain_pct is not None}
    gain_total = (state.current_tput / state.baseline_tput - 1.0) * 100.0 if state.baseline_tput > 0 else 0.0
    attempts: list[dict[str, Any]] = [
        {"ref": _config_ref(a), "status": a.outcome.value, "gain_pct": a.gain_pct, "why": a.note or a.candidate.reason}
        for a in reversed(state.attempts)
    ][:MAX_PRIOR_ATTEMPTS]
    facts = BriefFacts(
        delivery=DELIVERY_REPLY,
        baseline_metric=_metric(state.baseline_tput),
        current_metric=_metric(state.current_tput),
        focus_tag_domains=focus,
        proposal_schemas=PROPOSAL_SCHEMAS,
    )
    return BriefInputs(
        task_id=f"{domain.key}:{gap.canonical_id if gap else 'task'}",
        domain=domain,
        max_turns=ask.max_turns or DEFAULT_MAX_TURNS,
        gpu_type=w.gpu_type,
        tp=w.tp,
        precision=w.precision,
        conc=w.conc,
        isl=w.isl,
        osl=w.osl,
        max_model_len=w.max_model_len,
        framework=w.framework,
        framework_version=w.framework_version,
        benchmark_mode=w.benchmark_mode,
        model_name=w.model_name,
        model_info=dict(w.model_info),
        gap_canonical_id=gap.canonical_id if gap else "",
        gap_symptom=gap.symptom if gap else "",
        gap_layer=gap.layer if gap else "",
        gap_evidence=dict(gap.evidence) if gap else {},
        experience_kb_block=state.evidence.experience,
        roofline_evidence=dict(state.evidence.roofline),
        runtime_findings=state.evidence.runtime_findings,
        warm_start_lessons=[dict(v) for v in state.evidence.lessons],
        warm_start_pitfalls=[dict(v) for v in state.evidence.pitfalls],
        pr_monitor_available=False,
        extra_focus_tags=focus,
        scope=scope,
        mode=MODE_PATCH if ask.patches else _MODE_RESEARCH,
        task_description=ask.task,
        wall_budget_sec=ask.wall_budget_sec,
        baseline_tput=state.baseline_tput,
        current_tput=state.current_tput,
        cumulative_gain_validated=gain_total,
        keep_threshold_pct=state.keep_threshold_pct,
        applied_stack=[{"variant_name": c.name, "gain_pct": gains.get(c.name) or 0.0} for c in state.stack],
        prior_attempts=attempts,
        facts=facts,
    )


async def run_specialist(state: SearchState, ask: ProposeAsk, ports: Ports, *, seed: int | None = None) -> Proposal:
    """The candidates the specialist that answers ``ask`` delivers, best first; raises ``PortError``."""
    inputs = brief_inputs(state, ask)
    if inputs is None:
        return Proposal(summary=f"domain {ask.domain!r} authors no candidates")
    system_prompt, user_prompt = build_specialist_prompts(inputs)
    reply = await ports.agent(
        AgentRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_turns=inputs.max_turns,
            wall_budget_sec=ask.wall_budget_sec,
            writes_source=ask.patches,
            seed=seed,
        )
    )
    done = read_done(reply.text)
    prefix = inputs.domain.key.removesuffix("_specialist")
    candidates = candidates_of(
        done, reply.patches if ask.patches else (), prefix=prefix, max_candidates=ask.max_candidates
    )
    if done is None:
        return Proposal(candidates, summary=f"the specialist replied without a specialist_done ({reply.finish_reason})")
    try:
        confidence = min(1.0, max(0.0, float(done.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return Proposal(
        candidates,
        summary=str(done.get("summary") or ""),
        confidence=confidence,
        findings=tuple(_texts(done.get("new_findings"))),
        questions=tuple(_texts(done.get("residual_questions"))),
    )


def _texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("what") or item.get("question") or item.get("summary") or ""
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out

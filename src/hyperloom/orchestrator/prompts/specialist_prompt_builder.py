# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The inputs of one specialist dispatch, and the brief the body asks the brain to write for it.

The request carries the inputs and what only the body knows: its registries and rules, the enablement mandate,
the multi-node PD topology and which source trees on the host are git checkouts. Not pure: those last three read
multi-node state and probe the host's source trees.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from hyperloom.brain_api import BrainContractError, CallSite, StateSnapshot, TickBatch, ToolCall, TurnFailed
from hyperloom.common.perf_metric import is_agentx_mode
from hyperloom.inference_optimizer.framework_registry import FRAMEWORKS, format_primary_metric
from .agentx_context import corpus_lines, grading_lines, launch_ownership_lines
from .proposal_schemas import (
    EXPLORE_VARIANT_CONTRACT,
    EXPLORE_VARIANT_EXAMPLE,
    UPSTREAM_CANDIDATE_CONTRACT,
    UPSTREAM_CANDIDATE_EXAMPLE,
)
from ..measurement.runtime_findings import CORRECTNESS_FIX_MAX_DROP_PCT, CORRECTNESS_FIX_RULES
from ..specialists.domains import (
    DEFAULT_SPECIALIST_MAX_TURNS,
    PR_QUERY_REPOS,
    SpecialistDomain,
    domain_for_tag,
)
from ..brain_link import resolve_brain
from ..specialists.profile import MODE_PATCH


# Forbids global process cleanup that could kill the optimizer's serving /
# benchmark process. Shared by bash-enabled specialist and leaf prompts.
BASH_KILL_SAFETY_PREAMBLE = (
    "Do NOT run global process cleanup. Never run `ps aux | grep ... | xargs "
    "kill`, `pgrep -f ... | xargs kill`, or `killall` — these can kill the "
    "optimizer's serving / benchmark process. Only manage processes you "
    "started yourself, by their own PID."
)


@dataclass(frozen=True)
class SpecialistPromptInputs:
    """Typed inputs the Coordinator hands to the prompt builder."""

    # Identity
    task_id: str
    domain: SpecialistDomain
    max_turns: int = DEFAULT_SPECIALIST_MAX_TURNS

    # ``tp`` defaults to 0 (sentinel for "unspecified"), not 1, so
    # comm_specialist doesn't veto its own TP proposals.
    gpu_type: str = ""
    allocated_gpu_ids: tuple[int, ...] = ()
    tp: int = 0
    hbm_gb: float = 0.0
    peak_tflops: float = 0.0
    arch_notes: str = ""
    # Advisory competitor target gap block; direction hint only.
    target_gap_notes: str = ""
    # Already-proven warm-recipe optimizations the research scout should skip.
    already_proven: list[dict[str, str]] = field(default_factory=list)
    # Curated recipe-site URL templates the research scout may mine for
    # verified serve flags / envs; empty falls back to the built-in defaults.
    recipe_sites: tuple[str, ...] = ()
    # Advisory research-hint block; its presence suppresses cold-start fallback.
    research_hints: str = ""
    # Workload context mirrored from SharedState; renders in section 2.
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    # ``framework`` is the active server framework (``sglang`` / ``vllm`` /
    # ``atom``); empty falls back to the canonical sglang/vllm hint blocks.
    # ``framework_version`` is the precise install version (empty => no note).
    framework: str = ""
    framework_version: str = ""
    # ``"agentx"`` for agentic trace replay, else synthetic; selects the
    # workload and grading blocks. ``agentx_corpus_shape`` supplies their
    # numbers, mirrored from SharedState so the prompt describes the corpus the
    # session actually replayed.
    benchmark_mode: str = ""
    agentx_corpus_shape: dict[str, Any] = field(default_factory=dict)
    agentx_grading: dict[str, Any] = field(default_factory=dict)
    agentx_backend: str = ""

    # Gap statement
    gap_canonical_id: str = ""
    gap_symptom: str = ""
    gap_layer: str = ""
    gap_evidence: dict[str, Any] = field(default_factory=dict)

    # Optional structured KB context. Empty in the RecipeKB-first path.
    kb_subgraph: dict[str, Any] = field(default_factory=dict)
    # Experience service ``prompt_block`` read for this dispatch; empty when no Experience was rendered.
    experience_kb_block: str = ""

    # Roofline / TraceLens evidence from ``SharedState.last_trace_analyze``;
    # empty dict renders a placeholder.
    roofline_evidence: dict[str, Any] = field(default_factory=dict)
    # Rendered runtime findings of the current best's server log; empty omits the section.
    runtime_findings: str = ""

    # Recipe summary from the T0 warm-start recipe search
    # (``recipe_kb_t0._cascade_warm_start_search``)
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    # T0 lessons — positive priors from prior KEEPs; rendered in the lessons section.
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    pr_monitor_available: bool = True

    # Extra knowledge-domain tags; each contributes a focus block to Section 1.
    extra_focus_tags: tuple[str, ...] = ()

    # Local source navigation hint. ``worktree_base`` is the tree the worktree
    # stands for -- the checkout it was cut from, or the installed tree it holds
    # a snapshot of -- and is empty when the specialist has no worktree.
    session_framework_tree: str = ""
    framework_source_roots: tuple[str, ...] = ()
    worktree_base: str = ""
    source_hint_directories: tuple[str, ...] = ()
    # Framework package directory relative to the worktree; empty without a worktree.
    worktree_package_dir: str = ""

    # Target model identity warmed from SharedState.model_name / model_path; "" => not recorded.
    model_name: str = ""
    model_path: str = ""
    # Structured model architecture features mirrored from SharedState.model_info;
    # machine-parseable companion to ``arch_notes``. Empty dict => not warmed.
    model_info: dict[str, Any] = field(default_factory=dict)
    # Pre-rendered static-recon checklist block (Markdown); only populated for
    # the static_recon_specialist dispatch.
    static_recon_checklist: str = ""

    # Enablement dispatch evidence, folded into the §1b mandate. Empty for
    # every non-enablement domain; the mandate omits whichever is empty.
    enablement_source_context: str = ""
    enablement_candidate_refs: tuple[str, ...] = ()
    # The serialized FailureSignature the round was dispatched on; required for
    # the enablement domain, empty for every other.
    enablement_failure_signature: dict[str, Any] = field(default_factory=dict)
    # Env / server-arg layers prior advanced rounds accepted; the bench for this
    # round launches with them, so the mandate has to name them.
    enablement_accepted_config: dict[str, Any] = field(default_factory=dict)

    # Workspace path (for transcript / heartbeat instructions)
    workspace_path: str = ""

    # Free-form notes from Orchestration (e.g. previous-round resid_qs)
    notes: str = ""

    # Dispatch profile dials (see orchestrator.specialists.profile) that shape
    # single-domain / cross-domain / freeform / bench prompting.
    scope: str = "domain"
    mode: str = MODE_PATCH
    bench: bool = False
    # Free-form task description (only populated when scope == 'freeform').
    task_description: str = ""

    # Coordinator-injected note for a bounded auto-retry of a prior transient
    # (timeout / crash / stale-heartbeat) attempt; empty on the first attempt.
    auto_retry_reason: str = ""

    # Wall-clock budget (seconds) and dispatch start timestamp (ISO-8601 UTC)
    # so the specialist can self-throttle. 0 / "" => not supplied (the budget
    # section renders nothing).
    wall_budget_sec: float = 0.0
    started_at_iso: str = ""

    # Run-status snapshot for §0 MANDATE; 0/[] = absent (section renders gracefully).
    baseline_tput: float = 0.0
    current_tput: float = 0.0
    cumulative_gain_validated: float = 0.0
    keep_threshold_pct: float = 0.0
    applied_stack: list[dict[str, Any]] = field(default_factory=list)

    # Structured mandate payload; routes §0 boilerplate and per-dispatch context.
    task_kind: str = ""
    prior_attempts: list[dict[str, Any]] = field(default_factory=list)
    pr_lead: dict[str, Any] = field(default_factory=dict)

    # Operator-defined specialist focus, rendered verbatim; empty when not configured.
    custom_specialist_prompt: str = ""


class _BriefPort:
    """What a brief call lends the brain: no model and no tool."""

    llm: dict[str, Any] = {}

    async def call(self, call: ToolCall) -> dict[str, Any]:
        raise TurnFailed(f"a specialist brief cannot call {call.name!r}")


async def brief_from_brain(prompt_inputs: SpecialistPromptInputs) -> tuple[str, str]:
    """The ``(system_prompt, user_prompt)`` the brain writes for this dispatch.

    Raises:
        BrainContractError: if the brain answers with anything but one ``specialist.run`` carrying both prompts.
    """
    brain = resolve_brain()
    batch = TickBatch(
        call_site=CallSite.BRIEF,
        tick=0,
        now_unix=time.time(),
        events=(),
        snapshot=StateSnapshot(phase="", data={"specialist_brief": brief_request(prompt_inputs)}),
    )
    turn = await brain.on_brief(batch, _BriefPort())
    calls = turn.tool_calls
    if len(calls) == 1 and calls[0].name == "specialist.run":
        system_prompt, user_prompt = calls[0].args.get("system_prompt"), calls[0].args.get("user_prompt")
        if isinstance(system_prompt, str) and isinstance(user_prompt, str):
            return system_prompt, user_prompt
    raise BrainContractError(f"brain {brain.name!r} answered a brief with {[c.name for c in calls]}")


def brief_request(inp: SpecialistPromptInputs) -> dict[str, Any]:
    """What the brain writes this dispatch's brief from: ``{"inputs": {...}, "facts": {...}}``."""
    inputs = {f.name: getattr(inp, f.name) for f in fields(inp)}
    domain = inp.domain
    inputs["domain"] = {
        "key": domain.key,
        "layer": domain.layer,
        "kb_anchor": domain.kb_anchor,
        "description": domain.description,
    }
    return {"inputs": inputs, "facts": _brief_facts(inp)}


def _brief_facts(inp: SpecialistPromptInputs) -> dict[str, Any]:
    from hyperloom.orchestrator.actions.executors._multi_node_env import pd_topology_from_state
    from hyperloom.orchestrator.policy.gate import PR_MONITOR_TOOL_NAMES

    session_tree = (inp.session_framework_tree or "").strip()
    worktree_base = (inp.worktree_base or "").rstrip("/")
    return {
        "bash_kill_safety": BASH_KILL_SAFETY_PREAMBLE,
        "python_packages": {name: spec.python_package for name, spec in FRAMEWORKS.items()},
        "baseline_metric": format_primary_metric(inp.framework, inp.baseline_tput),
        "current_metric": format_primary_metric(inp.framework, inp.current_tput),
        "agentx_workload": _agentx_workload(inp),
        "focus_tag_domains": [getattr(domain_for_tag(t), "key", "") for t in inp.extra_focus_tags],
        "proposal_schemas": {
            "explore_variant": {"example": EXPLORE_VARIANT_EXAMPLE, "contract": list(EXPLORE_VARIANT_CONTRACT)},
            "upstream_candidate": {
                "example": UPSTREAM_CANDIDATE_EXAMPLE,
                "contract": list(UPSTREAM_CANDIDATE_CONTRACT),
            },
        },
        "correctness_fix_rules": sorted(CORRECTNESS_FIX_RULES),
        "correctness_fix_max_drop_pct": CORRECTNESS_FIX_MAX_DROP_PCT,
        "pr_monitor_tools": sorted(PR_MONITOR_TOOL_NAMES),
        "pr_query_repos": list(PR_QUERY_REPOS),
        "enablement_mandate": _enablement_mandate(inp),
        "pd_topology": pd_topology_from_state(),
        "worktree_base_has_git": bool(worktree_base) and (Path(worktree_base) / ".git").exists(),
        "git_checkouts": [r for r in (session_tree, *inp.framework_source_roots) if (Path(r) / ".git").is_dir()],
        "session_tree_is_package": bool(session_tree) and (Path(session_tree.rstrip("/")) / "__init__.py").is_file(),
    }


def _agentx_workload(inp: SpecialistPromptInputs) -> list[str]:
    if not is_agentx_mode(inp.benchmark_mode):
        return []
    return [
        *corpus_lines(inp.agentx_corpus_shape),
        *grading_lines(inp.agentx_grading, inp.agentx_backend),
        *launch_ownership_lines(inp.agentx_backend),
    ]


def _enablement_mandate(inp: SpecialistPromptInputs) -> str:
    """The enablement mandate's task description, built from the dispatch's verdict and evidence."""
    if inp.domain.key != "enablement_specialist":
        return ""
    from hyperloom.common.failure_signature import EnablementRequest, FailureSignature
    from hyperloom.orchestrator.enablement.mandate import build_mandate

    model = str((inp.gap_evidence or {}).get("model") or "").strip()
    req = EnablementRequest(
        framework=(inp.framework or "").strip().lower(),
        model=model or "(target model)",
        repo_url="",
        gpu_type=(inp.gpu_type or "").strip().lower(),
    )
    mandate = build_mandate(
        req,
        FailureSignature.from_dict(inp.enablement_failure_signature),
        candidate_refs=inp.enablement_candidate_refs,
        source_context=inp.enablement_source_context,
    )
    return mandate.task_description


__all__ = [
    "BASH_KILL_SAFETY_PREAMBLE",
    "SpecialistPromptInputs",
    "brief_from_brain",
    "brief_request",
]

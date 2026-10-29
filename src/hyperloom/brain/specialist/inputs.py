# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist brief is written from: the dispatch's inputs, and the facts only the body can supply."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

#: The dispatch mode in which a specialist authors patches.
MODE_PATCH = "patch"


@dataclass(frozen=True)
class DomainCard:
    """The specialist domain a dispatch runs under, as the brief names it."""

    key: str
    layer: str
    kb_anchor: str
    description: str = ""


@dataclass(frozen=True)
class BriefFacts:
    """What the body knows that a brief states: its registries and rules, and what it read off the host."""

    #: The body's rule against global process cleanup, stated among the iron rules.
    bash_kill_safety: str = ""
    #: Registered framework -> the package it imports as, ``None`` when it ships none.
    python_packages: Mapping[str, str | None] = field(default_factory=dict)
    #: The run's baseline and current best, formatted in the framework's primary metric.
    baseline_metric: str = ""
    current_metric: str = ""
    #: The AgentX corpus, grading and launch-ownership rows; empty outside AgentX.
    agentx_workload: tuple[str, ...] = ()
    #: The domain each extra focus tag names, ``""`` for a tag that names none; in tag order.
    focus_tag_domains: tuple[str, ...] = ()
    #: ``proposal_set`` item schemas the body parses, by kind: ``{"example": {...}, "contract": [...]}``.
    proposal_schemas: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    correctness_fix_rules: tuple[str, ...] = ()
    correctness_fix_max_drop_pct: float = 0.0
    pr_monitor_tools: tuple[str, ...] = ()
    pr_query_repos: tuple[str, ...] = ()
    #: The enablement mandate's task description; empty outside the enablement domain.
    enablement_mandate: str = ""
    #: The PD-disaggregation topology; empty unless disaggregated.
    pd_topology: Mapping[str, Any] = field(default_factory=dict)
    #: Whether the tree the worktree stands for has a ``.git`` entry.
    worktree_base_has_git: bool = False
    #: The source roots that hold a ``.git`` directory, as the inputs name them.
    git_checkouts: frozenset[str] = frozenset()
    #: Whether the session's framework tree holds an ``__init__.py``.
    session_tree_is_package: bool = False

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "BriefFacts":
        """Facts as the body sends them, sequences restored to the types the brief reads."""
        values = dict(data)
        for name in _tuple_fields(cls):
            if name in values:
                values[name] = tuple(values[name])
        if "git_checkouts" in values:
            values["git_checkouts"] = frozenset(values["git_checkouts"])
        return cls(**values)


@dataclass(frozen=True)
class BriefInputs:
    """The inputs of one specialist dispatch, field for field the body's ``SpecialistPromptInputs``."""

    task_id: str
    domain: DomainCard
    max_turns: int
    gpu_type: str = ""
    allocated_gpu_ids: tuple[int, ...] = ()
    tp: int = 0
    hbm_gb: float = 0.0
    peak_tflops: float = 0.0
    arch_notes: str = ""
    target_gap_notes: str = ""
    already_proven: list[dict[str, str]] = field(default_factory=list)
    recipe_sites: tuple[str, ...] = ()
    research_hints: str = ""
    precision: str = ""
    conc: int = 0
    isl: int = 0
    osl: int = 0
    max_model_len: int = 0
    framework: str = ""
    framework_version: str = ""
    benchmark_mode: str = ""
    agentx_corpus_shape: dict[str, Any] = field(default_factory=dict)
    agentx_grading: dict[str, Any] = field(default_factory=dict)
    agentx_backend: str = ""
    gap_canonical_id: str = ""
    gap_symptom: str = ""
    gap_layer: str = ""
    gap_evidence: dict[str, Any] = field(default_factory=dict)
    kb_subgraph: dict[str, Any] = field(default_factory=dict)
    experience_kb_block: str = ""
    roofline_evidence: dict[str, Any] = field(default_factory=dict)
    runtime_findings: str = ""
    warm_start_recipe: dict[str, Any] = field(default_factory=dict)
    warm_start_pitfalls: list[dict[str, Any]] = field(default_factory=list)
    warm_start_lessons: list[dict[str, Any]] = field(default_factory=list)
    pr_monitor_available: bool = True
    extra_focus_tags: tuple[str, ...] = ()
    session_framework_tree: str = ""
    framework_source_roots: tuple[str, ...] = ()
    worktree_base: str = ""
    source_hint_directories: tuple[str, ...] = ()
    worktree_package_dir: str = ""
    model_name: str = ""
    model_path: str = ""
    model_info: dict[str, Any] = field(default_factory=dict)
    static_recon_checklist: str = ""
    enablement_source_context: str = ""
    enablement_candidate_refs: tuple[str, ...] = ()
    enablement_failure_signature: dict[str, Any] = field(default_factory=dict)
    enablement_accepted_config: dict[str, Any] = field(default_factory=dict)
    workspace_path: str = ""
    notes: str = ""
    scope: str = "domain"
    mode: str = MODE_PATCH
    bench: bool = False
    task_description: str = ""
    auto_retry_reason: str = ""
    wall_budget_sec: float = 0.0
    started_at_iso: str = ""
    baseline_tput: float = 0.0
    current_tput: float = 0.0
    cumulative_gain_validated: float = 0.0
    keep_threshold_pct: float = 0.0
    applied_stack: list[dict[str, Any]] = field(default_factory=list)
    task_kind: str = ""
    prior_attempts: list[dict[str, Any]] = field(default_factory=list)
    pr_lead: dict[str, Any] = field(default_factory=dict)
    custom_specialist_prompt: str = ""

    facts: BriefFacts = field(default_factory=BriefFacts)

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> "BriefInputs":
        """The inputs of a brief request: ``{"inputs": {..., "domain": {...}}, "facts": {...}}``."""
        values = dict(request["inputs"])
        values["domain"] = DomainCard(**values["domain"])
        for name in _tuple_fields(cls):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values, facts=BriefFacts.from_wire(request.get("facts") or {}))


def _tuple_fields(cls: type) -> list[str]:
    return [f.name for f in fields(cls) if str(f.type).startswith("tuple[")]

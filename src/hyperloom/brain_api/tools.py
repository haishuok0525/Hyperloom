# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tools the body offers the brain, and how today's intents and actions name them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from hyperloom.brain_api.wire import WireError, json_value


class ToolKind(str, Enum):
    """How a tool answers."""

    #: Queued and run by the body; returns a task id, and the outcome arrives later as an event.
    JOB = "job"
    #: Answered inline from body state, with no side effect.
    QUERY = "query"
    #: Answered inline with accepted or denied; the body may refuse it.
    CONTROL = "control"


class Caller(str, Enum):
    """Which part of the brain may call a tool."""

    #: A model turn (orchestration, critic), through the intents it emits.
    LLM = "llm"
    #: Deterministic brain code: phase strategy, autosubmit, verdict follow-up.
    STRATEGY = "strategy"


class VerdictOutcome(str, Enum):
    """What a settled Critic verdict makes of its proposal, as ``verdict.settle`` names it."""

    #: Run it: the proposal becomes a task.
    MATERIALIZE = "materialize"
    #: File the refusal of an upstream-PR pre-screen against its candidate.
    RECORD_DENIAL = "record_denial"
    #: Re-arm the enablement lane the refused patch came from.
    REARM_ENABLEMENT = "rearm_enablement"
    #: Hand the Critic's feedback back to the proposal's author.
    REAUTHOR = "reauthor"
    #: Nothing further.
    NONE = "none"


@dataclass(frozen=True)
class ToolSpec:
    """One entry of the tool catalogue."""

    name: str
    kind: ToolKind
    callers: frozenset[Caller]
    #: Kept only until the body splits it into ``candidate.apply`` + ``candidate.benchmark`` (an ``action.*``), or
    #: until the brain makes the decisions the body step still makes (a ``pump.*``).
    transitional: bool = False


_BOTH = frozenset({Caller.LLM, Caller.STRATEGY})
_STRATEGY_ONLY = frozenset({Caller.STRATEGY})
_LLM_ONLY = frozenset({Caller.LLM})

#: Executors the brain reaches as ``action.<name>``; the LLM may propose only those in ``_LLM_ACTIONS``.
_ACTIONS = (
    "baseline",
    "conc_sweep",
    "kernel_agent",
    "profile",
    "replay_warm_recipe",
    "report",
    "roofline",
    "session_breakdown",
    "target_analysis",
    "targeted_build",
)
_LLM_ACTIONS = frozenset({"baseline", "report", "target_analysis"})
_TRANSITIONAL_ACTIONS = ("explore", "integrate_patch")

#: Read-only context pulls, as orchestration calls them today.
QUERY_NAMES: tuple[str, ...] = (
    "get_mission_status",
    "get_shared_state",
    "get_gaps",
    "get_warm_start",
    "get_proposal_scores",
    "get_intervention_mix",
    "why_denied",
    "show_analysis_md",
    "get_runtime_findings",
    "get_inbox",
    "get_recent_outcomes",
    "get_running_tasks",
    "read_reference",
    "get_failure",
    "get_variant_failures",
)


#: FRAMEWORK pump steps the brain runs as ``pump.<name>``, in the order it chooses, each still deciding its own work.
_PUMP_STEPS = (
    "framework_agent",
    "bench_untested",
    "research_scout",
    "config_plateau_recon",
    "stalled_domain",
    "custom_specialist",
    "advisory_plateau",
    "predictor",
    "finish_custom_cycle",
)


def _phase_strategy_tools() -> tuple[ToolSpec, ...]:
    """The FRAMEWORK candidate track, its discovery, the untested-proposal bench and the predictor."""
    return (
        # The FRAMEWORK candidate track. Candidate ids in flight, ``{}`` -> ``{"ids"}``; the next unprocessed candidate
        # outside ``exclude``, ``{"exclude"}`` -> ``{"found", "id"}``; whether an authored patch for an unprocessed
        # candidate is still out, ``{}`` -> ``{"inflight"}``; whether the source arm has gone dry, ``{}`` ->
        # ``{"plateaued"}``.
        ToolSpec("query.framework_inflight_candidates", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("query.framework_next_candidate", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("query.framework_authoring_inflight", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("query.source_arm_plateau", ToolKind.QUERY, _STRATEGY_ONLY),
        # Submit a candidate for review, ``{"id"}`` -> ``{"id"}`` (its key once submitted); ask for candidate
        # discovery or a local source exploration, ``{"reason"}`` -> ``{"dispatched"}``; close the candidate track,
        # ``{"reason"}``.
        ToolSpec("framework.submit_candidate", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("framework.discover", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("framework.local_explore", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("framework.phase_done", ToolKind.CONTROL, _STRATEGY_ONLY),
        # Whether a discovery round is queued or running, ``{}`` -> ``{"inflight"}``; dispatch one, ``{}``; or count
        # the one in flight as this ask's, ``{}``.
        ToolSpec("query.candidate_discovery_inflight", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("discovery.dispatch", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("discovery.wait", ToolKind.CONTROL, _STRATEGY_ONLY),
        # The untested-proposal bench. Dispatch admission, ``{}`` -> ``{"frozen", "paused"}``; the queue,
        # ``{}`` -> ``{"explores", "custom_rows", "operator_work"}``; untested non-operator rows, ``{}`` ->
        # ``{"count"}``; bench one grid of the ``"custom"`` or ``"untested"`` rows, ``{"rows", "limit"}`` ->
        # ``{"created"}``.
        ToolSpec("query.dispatch_admission", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("query.bench_queue", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("query.untested_rows", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("bench.enqueue", ToolKind.JOB, _STRATEGY_ONLY),
        # The predictor. The phase's headroom, ``{}`` -> ``{"headroom_sec"}`` (``None`` when unbounded); the re-profile
        # the decision point is held for, ``{}`` -> ``{"pending", "held_sec"}``; ask it, ``{}``; do not, with the
        # reason, ``{"reason"}``; hold FRAMEWORK entry for its answer, ``{"timeout_sec"}``.
        ToolSpec("query.phase_headroom", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("query.profile_hold", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("predictor.ask", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("predictor.skip", ToolKind.CONTROL, _STRATEGY_ONLY),
        ToolSpec("predictor.wait", ToolKind.CONTROL, _STRATEGY_ONLY),
        # Forced stalled-domain / custom specialist / wave split.
        ToolSpec("query.task_idempotency", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("query.patch_preflight", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("query.custom_specialist_gap", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("specialist.force_stalled", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("specialist.ensure_custom", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("specialist.wave_split", ToolKind.JOB, _STRATEGY_ONLY),
        # Macro-cycle focus and cycle-start reprofile.
        ToolSpec("macro.set_focus", ToolKind.CONTROL, _STRATEGY_ONLY),
        ToolSpec("prelude.reprofile", ToolKind.JOB, _STRATEGY_ONLY),
        # KERNEL enter and agent route.
        ToolSpec("kernel.enter", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("kernel.run", ToolKind.JOB, _STRATEGY_ONLY),
        # SWEEP directive and conc_sweep.
        ToolSpec("sweep.ask_directive", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("sweep.skip_conc", ToolKind.CONTROL, _STRATEGY_ONLY),
        ToolSpec("sweep.enqueue_conc", ToolKind.JOB, _STRATEGY_ONLY),
    )


def _catalogue() -> dict[str, ToolSpec]:
    specs = [
        ToolSpec("candidate.apply", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("candidate.benchmark", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("specialist.run", ToolKind.JOB, _BOTH),
        ToolSpec("kernel.integrate", ToolKind.JOB, _BOTH),
        ToolSpec("kernel.gemm_tuning", ToolKind.JOB, _BOTH),
        ToolSpec("kb.experience.read", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("kb.recipe.read", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("kb.agent.read", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("kb.static_recon.read", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("kb.critic.read", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("kb.critic.upsert", ToolKind.CONTROL, _STRATEGY_ONLY),
        ToolSpec("predictor.predict", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("task.extend_lease", ToolKind.CONTROL, _BOTH),
        ToolSpec("task.prune", ToolKind.CONTROL, _BOTH),
        ToolSpec("message.send", ToolKind.CONTROL, _BOTH),
        ToolSpec("message.alert", ToolKind.CONTROL, _BOTH),
        ToolSpec("message.request", ToolKind.CONTROL, _LLM_ONLY),
        ToolSpec("verdict.record", ToolKind.CONTROL, _BOTH),
        # What one Critic verdict on a pending proposal settles to, ``{"verdict", "authored", "variants",
        # "approved_variants", "holds", "outcome"}``: the verdict acted on and the one the Critic wrote, each variant's
        # verdict, the variants that may run (``None`` for all), every verdict held to a cited rule
        # (``{"variant", "to_verdict", "failure_reason_code"}``), and a :class:`VerdictOutcome`.
        ToolSpec("verdict.settle", ToolKind.CONTROL, _STRATEGY_ONLY),
        # The settled specialist's deliverable asked about goes to the Critic as an integrate_patch proposal, ``{}``;
        # the body assembles what it reviews.
        ToolSpec("review.submit", ToolKind.CONTROL, _STRATEGY_ONLY),
        # A specialist claimed patches none of which is on disk, ``{"claimed"}``; the body records it.
        ToolSpec("review.skip", ToolKind.CONTROL, _STRATEGY_ONLY),
        # The config lever a settled specialist delivered, ``{}`` -> ``{"levers"}`` (empty when none parses).
        ToolSpec("query.config_levers", ToolKind.QUERY, _STRATEGY_ONLY),
        # Whether the KB already rejected a config lever, ``{"levers"}`` -> ``{"known_bad"}``.
        ToolSpec("query.config_lever_known_bad", ToolKind.QUERY, _STRATEGY_ONLY),
        # Whether a re-author's candidate is already being applied or benchmarked, ``{}`` -> ``{"materializing"}``.
        ToolSpec("query.candidate_materializing", ToolKind.QUERY, _STRATEGY_ONLY),
        # A needs_review deliverable is re-authored, ``{"attempt", "critic_feedback"}``: the attempt it is and the
        # feedback the next authoring round is seeded with.
        ToolSpec("reauthor.dispatch", ToolKind.CONTROL, _STRATEGY_ONLY),
        # A needs_review verdict asked for no evidence, so its candidate is closed instead, ``{"rationale"}``.
        ToolSpec("reauthor.decline", ToolKind.CONTROL, _STRATEGY_ONLY),
        # A candidate spent its re-author budget and is closed, ``{"attempts", "rationale"}``.
        ToolSpec("reauthor.exhausted", ToolKind.CONTROL, _STRATEGY_ONLY),
        # Coordinator-owned read-only specialists the phase strategy dispatches: the research scout,
        # ``{"reason", "round_id"}``; static recon, ``{"reason", "slug"}``; the plateau trajectory review, ``{"domain"}``.
        ToolSpec("specialist.research_scout", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("specialist.static_recon", ToolKind.JOB, _STRATEGY_ONLY),
        ToolSpec("specialist.trajectory_review", ToolKind.JOB, _STRATEGY_ONLY),
        # Whether the config arm has gone dry, ``{}`` -> ``{"plateaued"}``.
        ToolSpec("query.config_arm_plateau", ToolKind.QUERY, _STRATEGY_ONLY),
        # Whether the search has plateaued, ``{}`` -> ``{"active"}``.
        ToolSpec("query.plateau_advisory", ToolKind.QUERY, _STRATEGY_ONLY),
        # The specialist domain the dominant roofline bottleneck points at, ``{}`` -> ``{"hinted_domain"}`` (empty
        # when none does).
        ToolSpec("query.trajectory_direction", ToolKind.QUERY, _STRATEGY_ONLY),
        ToolSpec("phase.request_exit", ToolKind.CONTROL, _BOTH),
        # One intent a model turn emitted, ``{"intent_type", "payload"}``, handed to the body's intent router whole.
        ToolSpec("intent.route", ToolKind.CONTROL, _LLM_ONLY),
        # The last inbox message a composed prompt showed, ``{"seq"}``; the body advances its cursor there once the
        # turn has run, and holds the rest for the next one.
        ToolSpec("inbox.rendered", ToolKind.CONTROL, _STRATEGY_ONLY),
    ]
    specs += [ToolSpec(f"action.{a}", ToolKind.JOB, _BOTH if a in _LLM_ACTIONS else _STRATEGY_ONLY) for a in _ACTIONS]
    specs += [ToolSpec(f"action.{a}", ToolKind.JOB, _BOTH, transitional=True) for a in _TRANSITIONAL_ACTIONS]
    specs += [ToolSpec(f"query.{q}", ToolKind.QUERY, _BOTH) for q in QUERY_NAMES]
    specs += [ToolSpec(f"pump.{p}", ToolKind.JOB, _STRATEGY_ONLY, transitional=True) for p in _PUMP_STEPS]
    return {s.name: s for s in (*specs, *_phase_strategy_tools())}


TOOLS: Mapping[str, ToolSpec] = MappingProxyType(_catalogue())

#: Intent type (``IntentType`` value) -> the tool it is a call of. ``delegate`` and ``propose_action`` name their
#: tool through ``action_name``; see :func:`tool_for_action`.
INTENT_TOOLS: Mapping[str, str | None] = MappingProxyType(
    {
        "send_message": "message.send",
        "delegate": None,
        "propose_action": None,
        "alert": "message.alert",
        "request": "message.request",
        "review_verdict": "verdict.record",
        "extend_lease": "task.extend_lease",
        "prune_branch": "task.prune",
        "escalate_strategy_change": "phase.request_exit",
    }
)

_ACTION_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "specialist": "specialist.run",
        "integrate": "kernel.integrate",
        "gemm_tuning": "kernel.gemm_tuning",
    }
)


def tool_for_action(action_name: str) -> str:
    """The tool an action proposal or delegation is a call of.

    Raises:
        KeyError: if no tool serves ``action_name``.
    """
    name = _ACTION_ALIASES.get(action_name, f"action.{action_name}")
    if name not in TOOLS:
        raise KeyError(f"no tool serves action {action_name!r}")
    return name


@dataclass(frozen=True)
class ToolCall:
    """One call the brain asks the body to make."""

    name: str
    args: Mapping[str, Any]
    #: The body runs a call at most once per key; a replayed turn repeats its keys.
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if self.name not in TOOLS:
            raise WireError(f"unknown tool {self.name!r}")
        object.__setattr__(self, "args", json_value(dict(self.args), f"tool {self.name} args"))

    def to_wire(self) -> dict[str, Any]:
        """This call as a JSON object."""
        return {"name": self.name, "args": self.args, "idempotency_key": self.idempotency_key}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "ToolCall":
        """The inverse of :meth:`to_wire`."""
        return cls(name=data["name"], args=data.get("args") or {}, idempotency_key=data.get("idempotency_key"))

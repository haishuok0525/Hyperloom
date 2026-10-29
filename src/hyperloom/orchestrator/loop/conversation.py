# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator prompt composition: the Critic's prompt, the orchestration view, MCP context readers, the advisory blocks phases share, and reactor conversation tracing."""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from ..bus.gpu_pool import gpus_by_task_sync
from ..phases import dispatch_order
from ..phases import machine_state as _phase_state
from ..policy.projection import resource_pools_summary
from ..specialists.in_flight import in_flight_domains
from ..specialists.reap_watch import newest_activity, specialist_activity_paths
from ..roles.base import BackendTurnResult
from ..bus.message_bus import Message
from hyperloom.inference_optimizer.tracelens_md import strip_base64_data_urls
from hyperloom.inference_optimizer.trace.conversation_trace import ConversationRecord, append_conversation
from ..state.shared_state import KbInjection
from hyperloom.common.coerce import to_unix
from hyperloom.common.prompt_safety import fit_to_budget

from .inbox_lines import _format_inbox_event
from .orchestration_view import ORCHESTRATION, orchestration_view
from ..state.task_registry import Task
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ..collaborator import CoordinatorCollaborator
import logging as _logging

if TYPE_CHECKING:
    from hyperloom.inference_optimizer.experience_kb import ExperienceKBEvidence
    from .coordinator import Coordinator

log = _logging.getLogger(__name__)

# Per-variant failure lines expanded by get_recent_outcomes, and the total cap that keeps a wide top_k from flooding
# the turn.
_RECENT_OUTCOMES_VARIANT_ROWS = 12
_RECENT_OUTCOMES_LINE_CAP = 120

# Ceiling on the inbox tail one prompt renders; uncapped, a long run outgrows the context window.
_INBOX_PROMPT_CHARS = 40_000
# Task kinds that queue for the benchmark lane, in the order the backlog line lists them.
_BENCH_LANE_KINDS = ("integrate_patch", "explore")
# Longest a KERNEL_AGENT session goes without an Orchestration turn while kernel progress is unchanged.
_KERNEL_ORCHESTRATION_HEARTBEAT_SEC = 900.0


@dataclass(frozen=True)
class ComposedPrompt:
    """One agent's prompt: ``context`` holds what changes rarely, ``turn`` what changes every tick."""

    context: str
    turn: str

    @property
    def text(self) -> str:
        """The whole prompt as one string."""
        return f"{self.context}\n{self.turn}"


@dataclass(frozen=True)
class TurnPrompt:
    """What one reactor turn hands its backend; ``context`` is the part of the prompt carried in ``system_prompt``."""

    system_prompt: str
    prompt: str
    context: str
    #: The orchestration turn's view, which the brain composes ``prompt`` and ``context`` from; ``None`` otherwise.
    view: dict[str, Any] | None = None


class ConversationCollaborator(CoordinatorCollaborator):
    """Manages conversation rounds: context tools, prompt injection, and round history."""

    def __init__(self, coordinator: "Coordinator") -> None:
        super().__init__(coordinator)
        # The Experience KB read rendered into the latest orchestration prompt; proposals emitted on that tick cite it.
        self.kb_last_read: ExperienceKBEvidence | None = None
        # Per-agent seq of the last message its prompt rendered.
        self._rendered_cursor: dict[str, int] = {}
        # The kernel progress fingerprint at Orchestration's last admitted KERNEL turn, and when it was admitted.
        self._admitted_kernel_fingerprint = ""
        self._admitted_at = 0.0

    async def _kb_prompt_block(self, untested_proposals: str) -> str:
        """Read shared Experience evidence once per FRAMEWORK_AGENT orchestration tick."""
        integration = self._coord.experience_kb
        if integration is None:
            return ""
        evidence = await asyncio.to_thread(
            integration.read_for_framework,
            self.shared_state,
            untested_proposals=untested_proposals,
        )
        self.kb_last_read = evidence
        if evidence.status != "completed" or not evidence.prompt_block:
            return ""
        block = "\n".join(
            (
                "=== Experience KB warm-start evidence ===",
                ("INVARIANT: the original Recipe benchmark measurement remains the gain-accounting baseline."),
                (
                    "Historical configurations may seed proposals/current-best "
                    "candidates only; never replace benchmark_baseline."
                ),
                evidence.prompt_block,
            )
        )
        self.shared_state.record_experience_kb_injection(
            KbInjection(
                consumer="orchestration",
                read_id=evidence.read_id,
                experience_ids=[str(ref.get("id") or "") for ref in evidence.rendered_refs],
                experiences=[dict(item) for item in evidence.experiences],
                prompt_block=block,
            )
        )
        return block

    def attach_orchestration_context_tools(self) -> None:
        """Bind a read-only ContextProvider to the orchestration backend (no-op without setter)."""
        backend = self.backends.get("orchestration")
        setter = getattr(backend, "set_context_provider", None)
        if setter is None:
            return
        from ..roles.mcp_context_tools import ContextProvider

        provider = ContextProvider(
            shared_state=self.shared_state,
            inbox_reader=self._context_inbox_reader,
            analysis_reader=self._context_analysis_reader,
            recent_outcomes_reader=self._context_recent_outcomes_reader,
            running_tasks_reader=self._context_running_tasks_reader,
            reference_reader=self._context_reference_reader,
        )
        setter(provider)

    def _context_reference_reader(self, name: str = "") -> str:
        """Resolve a reference doc by stem; reject path traversal."""
        from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir

        refs_dir = asset_prompt_references_dir()
        stem = (name or "").strip()
        if not stem or "/" in stem or "\\" in stem or stem.startswith("."):
            available = sorted(p.stem for p in refs_dir.glob("*.md"))
            return f"(read_reference: invalid name {name!r}; available: {available})"
        candidate = (refs_dir / stem).with_suffix(".md").resolve()
        if candidate.parent != refs_dir.resolve():
            return f"(read_reference: path traversal rejected for {name!r})"
        if not candidate.exists():
            available = sorted(p.stem for p in refs_dir.glob("*.md"))
            return f"(read_reference: {name!r} not found; available: {available})"
        return candidate.read_text(encoding="utf-8")

    def _context_inbox_reader(self, since_seq: int = 0) -> str:
        """Synchronous projection of the orchestration inbox tail (sync SQLite path)."""
        msgs = self.bus.inbox_context_sync("orchestration", after_seq=int(since_seq or 0))
        if not msgs:
            return "(no inbox events)"

        lines = [_format_inbox_event(m) for m in msgs]
        return "\n".join(lines)

    def _context_recent_outcomes_reader(self, top_k: int = 8) -> str:
        """Synchronous projection of recent action outcomes."""
        k = max(1, min(top_k or 8, 50))
        newest_first = self.bus.recent_outcomes_context_sync(limit=k)
        if not newest_first:
            return "(no recent outcomes)"
        # Flip newest-first query to newest-last for chronological reading.
        msgs = newest_first[::-1]

        header = "=== Recent action outcomes (newest last) ==="
        body_lines: list[str] = []
        body_lines.extend(_format_inbox_event(m, max_variant_rows=_RECENT_OUTCOMES_VARIANT_ROWS) for m in msgs)
        rendered = "\n".join(body_lines).splitlines()
        if len(rendered) > _RECENT_OUTCOMES_LINE_CAP:
            rendered = rendered[-_RECENT_OUTCOMES_LINE_CAP:]
            return "\n".join(
                [header]
                + rendered
                + [f"(truncated at {_RECENT_OUTCOMES_LINE_CAP} lines; re-query with a smaller top_k)"]
            )
        return "\n".join([header] + rendered)

    def _context_running_tasks_reader(self) -> str:
        """Project in-flight tasks and their held resources from three reads, not one snapshot."""
        tasks = self.tasks.running_context_sync()
        backlog = "\n".join(line for line in (self._queued_backlog_line(), self._queued_specialists_line()) if line)
        if not tasks:
            return "\n".join(line for line in ("(no tasks in flight)", backlog) if line)

        lanes_by_task = self.locks.lanes_by_task_sync()
        gpus_by_task = gpus_by_task_sync(self.db)
        now_unix = time.time()
        lines = ["=== Tasks in flight ==="]
        for task in tasks:
            lanes, expires_at = lanes_by_task.get(task.task_id, ([], ""))
            gpus = gpus_by_task.get(task.task_id, [])
            params = task.params or {}
            started = to_unix(task.updated_at, 0.0)
            running_sec = max(0.0, now_unix - started) if started > 0 else 0.0
            parts = [
                f"  - task_id={task.task_id}",
                f"kind={task.kind!r}",
                f"running_sec={int(running_sec)}",
            ]
            domain = str(params.get("domain") or "")
            gap = str(params.get("gap_canonical_id") or "")
            if domain:
                parts.append(f"domain={domain!r}")
            if gap:
                parts.append(f"gap={gap!r}")
            parts.append(f"idempotency_key={task.idempotency_key!r}")
            if task.lease_ttl_sec:
                parts.append(f"lease_ttl_sec={task.lease_ttl_sec}")
            if expires_at:
                exp_unix = to_unix(expires_at, 0.0)
                if exp_unix > 0:
                    parts.append(f"lease_expires_in_sec={int(exp_unix - now_unix)}")
            if lanes:
                parts.append(f"lanes={sorted(lanes)}")
            if gpus:
                parts.append(f"gpu_ids={sorted(gpus)}")
            hb_age = self._task_heartbeat_age_sec(task, now_unix=now_unix)
            if hb_age is not None:
                parts.append(f"heartbeat_age_sec={int(hb_age)}")
            lines.append(" ".join(parts))
        if backlog:
            lines.append(backlog)
        return "\n".join(lines)

    def _queued_backlog_line(self) -> str:
        """Benchmark-lane backlog: queued integrate_patch / explore tasks in dispatch order, and untested proposals."""
        counts = self.tasks.queued_kind_counts_sync()
        counts["untested proposals"] = len(self.shared_state.untested_proposal_rows())
        parts = [f"{counts[kind]} {kind}" for kind in (*_BENCH_LANE_KINDS, "untested proposals") if counts.get(kind)]
        if not parts:
            return ""
        first = dispatch_order.PHASE_DISPATCH_FIRST.get(self.shared_state.phase, frozenset())
        queued_bench = [kind for kind in _BENCH_LANE_KINDS if counts.get(kind)]
        ahead = [kind for kind in queued_bench if kind in first]
        behind = [kind for kind in queued_bench if kind not in first]
        order = (
            f" ({', '.join(ahead)} dispatches before any queued {', '.join(behind)} in this phase,"
            " including one queued later)"
            if ahead and behind
            else ""
        )
        return "queued backlog: " + ", ".join(parts) + order

    def _queued_specialists_line(self) -> str:
        """Domains a queued specialist holds: a specialist delegate for one of them is refused until it finishes."""
        held = in_flight_domains(self.tasks.queued_context_sync())
        if not held:
            return ""
        return "queued specialists (each holds its domain): " + ", ".join(
            f"{domain} task_id={task.task_id}" for domain, task in held.items()
        )

    def _task_heartbeat_age_sec(self, task: "Task", *, now_unix: float) -> float | None:
        """Age of a specialist's freshest liveness signal, read from the same paths the reaper polls."""
        if (task.kind or "").strip() != "specialist":
            return None
        newest = newest_activity(specialist_activity_paths(runs_dir(self.session_dir, "specialist", task.task_id)))
        if newest is None:
            return None
        return max(0.0, now_unix - newest)

    def _context_analysis_reader(self) -> str:
        """Return the latest TraceLens analysis.md snapshot text."""
        lta = self.shared_state.last_trace_analyze or {}
        text = lta.get("analysis_md_text") or ""
        if text:
            return strip_base64_data_urls(text)
        path = str(lta.get("analysis_md_path") or "")
        if path:
            try:
                from pathlib import Path as _Path

                return strip_base64_data_urls(_Path(path).read_text(encoding="utf-8"))
            except OSError as exc:
                return f"(analysis.md unreadable at {path}: {exc!r})"
        return "(no analysis.md snapshot yet)"

    def record_reactor_conversation(
        self,
        agent_name: str,
        result: BackendTurnResult,
        *,
        context: str = "",
    ) -> None:
        """Append one ``conversations.jsonl`` row for a reactor turn, with ``context`` ahead of the backend's prompt."""
        metadata = result.metadata or {}
        prompt = metadata.get("prompt")
        response = metadata.get("response")
        if not prompt and not response:
            return
        record = ConversationRecord(
            session_id=self.session_dir.name,
            component=agent_name,
            # Same turn metadata the token row is built from, so both halves carry the backend's call_id when it
            # stamped one.
            call_id=metadata.get("call_id"),
            role=agent_name,
            tick=int(self.shared_state.tick or 0),
            phase=(self.shared_state.phase or "") or None,
            model=metadata.get("model"),
            prompt=f"{context}\n{prompt}" if context else prompt or "",
            response=response or "",
        )
        append_conversation(session_dir=self.session_dir, record=record)

    async def _explore_grid_fingerprints(self) -> set[str]:
        """Content fingerprints of the variants in every queued or running explore grid."""
        from ..actions.executors._proposal_identity import content_fingerprint

        return {
            content_fingerprint(variant)
            for task in (*await self.tasks.queued(), *await self.tasks.running())
            if task.kind == "explore"
            for variant in (task.params or {}).get("grid") or ()
            if isinstance(variant, dict)
        }

    def trajectory_digest(self) -> str:
        """The deterministic trajectory digest, or "" when building it fails."""
        from ..knowledge import trajectory_reviewer

        try:
            return trajectory_reviewer.build_trajectory_digest(self.session_dir, self.shared_state)
        except (OSError, ValueError, TypeError):
            log.exception("Coordinator: trajectory review digest failed")
            return ""

    async def _render_inbox_sections(self, agent_name: str) -> list[str]:
        """Render the inbox tail since this agent's cursor, oldest first, and note what it showed.

        What the character budget leaves out waits for the next turn.
        """
        cursor = await self.cursors.load(agent_name)
        unread = await self.bus.replay_for(agent_name, after_seq=cursor.last_processed_seq)
        rendered = await self._augment_critic_inbox_with_pending(unread) if agent_name == "critic" else unread
        # Only Orchestration acts on variant-level failures; the reviewers do not.
        inbox_lines = fit_to_budget(
            [f"  {_format_inbox_event(m, max_variant_rows=0)}" for m in rendered],
            _INBOX_PROMPT_CHARS,
        )
        unread_ids = {m.msg_id for m in unread}
        read = [m for m in rendered[: len(inbox_lines)] if m.msg_id in unread_ids]
        last_seq = cursor.last_processed_seq
        if read:
            last_seq = int(read[-1].seq)
            self._rendered_cursor[agent_name] = last_seq
        if not inbox_lines:
            return [f"=== Inbox for {agent_name} ===", "(no new messages)"]
        sections = [f"=== Inbox for {agent_name} (newest last) ===", *inbox_lines]
        if len(read) < len(unread):
            sections.append(
                f"  (+{len(unread) - len(read)} newer messages after seq={last_seq} held for the next turn)"
            )
        return sections

    async def compose_prompt(self, agent_name: str) -> ComposedPrompt:
        """Compose the prompt for *agent_name*: the session context, and the per-tick blocks ending with the inbox tail.

        The orchestration prompt is the brain's; see :meth:`compose_turn`.
        """
        context: list[str] = [f"SESSION_DIR={self.session_dir}"]
        turn: list[str] = []

        phase_block = _phase_state.phase_status_summary(self.shared_state)
        if phase_block:
            turn.append("=== Phase ===")
            turn.append(phase_block)

        turn.append("=== Tick ===")
        turn.append(self.shared_state.to_tick_summary())
        # The session ledger moves with every action that lands, so it belongs to
        # the turn: carried in ``context`` it would rewrite the cached prefix the
        # stable blocks below sit in, on nearly every tick.
        turn.append("=== Shared session state ===")
        turn.append(self.shared_state.to_prompt_summary())
        turn.append(f"target_gap_pct={self._coord.target_gap_pct():.2f}")

        # A latency budget changes what a KEEP means, so it is never dropped. Pure string assembly, no I/O.
        context.extend(self.shared_state.to_constraint_sections())
        context.append("=== Resource pools ===")
        context.append(resource_pools_summary(self.shared_state))
        turn.extend(await self._render_inbox_sections(agent_name))

        return ComposedPrompt(context="\n".join(context), turn="\n".join(turn))

    def admit_turn(self, agent_name: str, now_unix: float) -> bool:
        """Whether ``agent_name`` takes a turn this tick; in KERNEL_AGENT, Orchestration waits for kernel progress or the heartbeat."""
        state = self.shared_state
        if agent_name != "orchestration" or (state.phase or "").upper() != _phase_state.PHASE_KERNEL_AGENT:
            return True
        fingerprint = state.kernel_progress_fingerprint
        last_turn = state.agent_last_active.get("orchestration", 0.0)
        handled = fingerprint == self._admitted_kernel_fingerprint and last_turn >= self._admitted_at
        if handled and now_unix - last_turn < _KERNEL_ORCHESTRATION_HEARTBEAT_SEC:
            return False
        self._admitted_kernel_fingerprint, self._admitted_at = fingerprint, now_unix
        return True

    async def compose_turn(self, agent_name: str, backend: Any) -> TurnPrompt:
        """The system prompt and prompt for one turn; a backend that caches its system prompt gets the session context in it.

        The orchestration turn carries the view the brain composes its prompt from instead of a prompt.
        """
        system_prompt = await self.load_system_prompt(agent_name)
        if agent_name == ORCHESTRATION:
            return TurnPrompt(system_prompt, "", "", view=await orchestration_view(self))
        composed = await self.compose_prompt(agent_name)
        if getattr(backend, "system_prompt_cached", False):
            return TurnPrompt(f"{system_prompt}\n\n{composed.context}", composed.turn, composed.context)
        return TurnPrompt(system_prompt, composed.text, "")

    def mark_rendered(self, agent_name: str, seq: int) -> None:
        """Note the last message ``agent_name``'s prompt showed; :meth:`advance_rendered_cursor` moves its cursor there."""
        self._rendered_cursor[agent_name] = seq

    async def advance_rendered_cursor(self, agent_name: str) -> None:
        """Advance an agent's read cursor to the last message its prompt rendered."""
        seq = self._rendered_cursor.get(agent_name)
        if seq is not None:
            await self.cursors.advance(agent_name, seq=seq)

    async def _augment_critic_inbox_with_pending(self, rendered: list["Message"]) -> list["Message"]:
        """Ensure every undecided proposal awaiting a Critic verdict is present."""
        pending = list(self.state.pending_proposals.values())
        if not pending:
            return rendered
        seen = {getattr(m, "msg_id", None) for m in rendered}
        extra: list["Message"] = []
        for p in pending:
            pid = str(getattr(p, "proposal_msg_id", "") or "")
            if not pid or pid in seen:
                continue
            pm = await self.bus.lookup_by_id(pid)
            if pm is not None:
                extra.append(pm)
                seen.add(pid)
        if not extra:
            return rendered
        merged = list(rendered) + extra
        merged.sort(key=lambda m: int(getattr(m, "seq", 0) or 0))
        return merged

    async def load_system_prompt(self, agent_name: str) -> str:
        """Load the system prompt for an agent, honoring overrides."""
        override = self.orch_prompt.get(agent_name)
        if override is not None:
            return override
        role = self.role_registry[agent_name]
        try:
            return role.load_system_prompt()
        except FileNotFoundError:
            return f"(no system prompt for {agent_name})"

    # Advisory prompt blocks (folded in from the former AdvisoryCollaborator).
    def plateau_advisory_block(self) -> str:
        """Render the plateau-judgment advisory block for the current phase."""
        state = self.shared_state
        phase = (state.phase or "").strip().upper()
        lines: list[str] = []
        if phase == _phase_state.PHASE_FRAMEWORK_AGENT:
            # The advisory reads the same predicate the exit rule does, so the
            # model is never shown a plateau the phase machine disagrees with.
            _, evidence = _phase_state.per_lever_dryness(state)
            config_dry = bool(evidence.get("config_arm_plateaued"))
            source_dry = bool(evidence.get("source_arm_plateaued"))
            config_ev = evidence
            source_ev = evidence
            if config_dry:
                lines.append("OPTIMIZE config arm plateaued: low recent KEEP gain plus specialist empty streak.")
                lines.append(
                    "  recent_keep_gain_pct="
                    f"{config_ev.get('recent_keep_gain_pct', 0.0)} "
                    f"threshold={config_ev.get('keep_gain_threshold_pct', 0.0)} "
                    f"empty_streak={config_ev.get('empty_streak', 0)} "
                    f"streak_threshold={config_ev.get('empty_streak_threshold', 0)}"
                )
            streak = int(source_ev.get("source_consecutive_no_keep", 0) or 0)
            if source_dry:
                lines.append("OPTIMIZE source arm plateaued: candidates exhausted or no KEEP on the trailing ones.")
            elif streak > 0:
                lines.append("OPTIMIZE source arm approaching plateau: no KEEP on the trailing candidates.")
            if source_dry or streak > 0:
                lines.append(
                    f"  consecutive_no_keep={streak} "
                    f"threshold={source_ev.get('source_threshold', 0)} "
                    f"candidates_exhausted={source_ev.get('source_candidates_exhausted', False)}"
                )
            if config_dry != source_dry:
                lines.append("  Only one arm is dry: the other lever is still live, and the phase stays open.")
        if not lines:
            return ""
        lines.append(
            "Note: OPTIMIZE advances to KERNEL_AGENT only when BOTH arms are dry "
            "(reason=optimize_no_more_leverage) -- a non-terminal lever switch, not "
            "the end of the run. Either arm going quiet also flags the next "
            "macro-cycle to steer off this bottleneck. You may request an earlier "
            "advance with an escalate_strategy_change hint, or keep working the live "
            "arm until the plateau / budget gate fires."
        )
        return "\n".join(lines)

    def dominant_roofline_direction(self) -> tuple[str, float]:
        """Return ``(direction, pct)`` for the most-saturated roofline direction in the latest snapshot; ``("", 0.0)`` when no snapshot is available."""
        from hyperloom.inference_optimizer.roofline_snapshot import dominant_direction

        snaps = self.shared_state.roofline_snapshots or []
        if not snaps or not isinstance(snaps[-1], dict):
            return "", 0.0
        return dominant_direction(snaps[-1])

    def _current_target_gap(self) -> dict[str, Any] | None:
        """Current-best vs the competitor target; None when the advisory is off or no target is set."""
        if not self.shared_state.target_advisory_enabled:
            return None
        from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

        target = _research_hints.load_competitor_target(self.session_dir)
        return _research_hints.gap_for_state(target, self.shared_state) if target else None

    def target_gap_advisory_block(self) -> str:
        """Build the advisory \"External target gap\" prompt block (current-best vs competitor target; never gates)."""
        from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

        return _research_hints.full_gap_summary(self._current_target_gap())

    def _current_primary_gap(self) -> str | None:
        """Resolve latency/throughput; None when advisory is off or no competitor target is set."""
        gap = self._current_target_gap()
        return str((gap or {}).get("primary_gap") or "").strip() or None

    def _recent_proposed_variants(
        self,
        *,
        max_rounds: int = 2,
    ) -> list[dict[str, Any]]:
        """Collect proposal_set rows from the most recent specialist rounds (deduped by name; fail-soft)."""
        rounds = [
            r
            for r in (self.shared_state.specialist_rounds or [])
            if isinstance(r, dict) and isinstance(r.get("proposal_set"), list)
        ]
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in rounds[-max_rounds:]:
            for variant in r.get("proposal_set") or []:
                if not isinstance(variant, dict):
                    continue
                name = str(variant.get("name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append(variant)
        return out

    def _priors_match_advisory_block(self) -> str:
        """Flag recently proposed variants aligning with proven priors / dominant external gap (advisory ordering)."""
        from hyperloom.inference_optimizer.baseline_comparison import research_hints as _research_hints

        variants = self._recent_proposed_variants()
        if not variants:
            return ""
        hints = _research_hints.load_hints(self.session_dir)
        primary_gap = self._current_primary_gap()
        return _research_hints.priors_match_summary(
            variants,
            hints,
            primary_gap=primary_gap,
        )

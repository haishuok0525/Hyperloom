# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CriticAgentBackend — bridges the ``hyperloom.agents.critic`` runtime into the Coordinator as a real Critic Backend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from hyperloom.brain_api import (
    BrainTurn,
    CallSite,
    LlmCallError,
    LlmReply,
    StateSnapshot,
    TickBatch,
    ToolCall,
    TurnFailed,
    WireError,
)
from hyperloom.brain_api.wire import json_value
from hyperloom.common.framework_arm import review_row_id
from hyperloom.common.llm_config import (
    DEFAULT_CODEX_MODEL,
    AnthropicCompletionRequest,
    LLMConfigError,
    aanthropic_completion,
    achat_completion,
    anthropic_transport_ready,
    apply_reasoning_effort,
    build_http_timeout,
    get_async_openai_client,
)
from hyperloom.inference_optimizer.protocol.intent import (
    IntentValidationError,
    NoIntentEmitted,
    validate_envelope,
)
from hyperloom.inference_optimizer.session.session_paths import allocate_turn_workdir, manifest_path
from hyperloom.common.coerce import optional_positive_int, to_int
from hyperloom.common.token_usage import anthropic_usage_counts, openai_usage_counts
from hyperloom.inference_optimizer.trace.conversation_trace import ConversationRecord, append_conversation
from hyperloom.inference_optimizer.trace.llm_trace import LLMCallRecord, LLMCallSite, append_llm_call, new_call_id
from hyperloom.inference_optimizer.trace.parse_usage import reasoning_output_tokens
from hyperloom.inference_optimizer.trace.trajectory_trace import current_context
from hyperloom.orchestrator.brain_link import resolve_brain
from hyperloom.orchestrator.lever import patch_lever_kind
from .base import BackendError, BackendTurnResult, LLMCallFailed, build_chat_messages, parse_call_timeout_env

if TYPE_CHECKING:
    from hyperloom.inference_optimizer.breakdown.recorder.critic_out import CriticIteration


log = logging.getLogger(__name__)


CRITIC_AGENT_RUNTIME_TIMEOUT_SEC = 30  # prepare-review / commit-review wall cap
# Characters of the child's stderr kept in a failure message.
_STDERR_TRUNCATE_CHARS = 500
# Output-token cap for both review paths.
CRITIC_AGENT_MAX_COMPLETION_TOKENS = 32000


def _accumulate_usage(acc: dict[str, int], counts: dict[str, int], usage: Any) -> None:
    """Fold one reply's token counts and reasoning-output tokens into the accumulator."""
    for key, count in counts.items():
        acc[key] = acc.get(key, 0) + count
    reasoning = reasoning_output_tokens(usage)
    if reasoning is not None:
        acc["reasoning_output_tokens"] = acc.get("reasoning_output_tokens", 0) + reasoning


# HTTP client timeout defaults for critic review calls.
CRITIC_AGENT_LLM_CONNECT_TIMEOUT_SEC = 10.0
CRITIC_AGENT_LLM_RW_TIMEOUT_SEC = 300.0

# Cap on per-turn workdirs kept on disk; older ones are pruned each turn.
CRITIC_AGENT_WORKDIR_KEEP_COUNT = 50


@dataclass
class RuntimeCall:
    """One ``runtime.cli`` invocation, captured for tests + logging."""

    phase: str
    request_path: Path
    out_path: Path
    cwd: Path
    env: dict[str, str]
    review_path: Path | None = None


RuntimeCaller = Callable[[RuntimeCall], None]
"""Callable that performs (or fakes) a ``runtime.cli`` invocation.

The default real caller shells out via :mod:`subprocess`; tests inject a fake
that writes the output JSON to ``out_path`` directly.
"""


def _default_runtime_caller(call: RuntimeCall) -> None:
    """Real implementation — runs ``python -m hyperloom.agents.critic.runtime.cli <phase> ...``."""
    cmd = [
        sys.executable,
        "-m",
        "hyperloom.agents.critic.runtime.cli",
        call.phase,
        "--request",
        str(call.request_path),
        "--out",
        str(call.out_path),
    ]
    if call.phase == "commit-review":
        if call.review_path is None:
            raise BackendError("commit-review invocation missing --review path")
        cmd += ["--review", str(call.review_path)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(call.cwd),
            env=call.env,
            capture_output=True,
            text=True,
            timeout=CRITIC_AGENT_RUNTIME_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackendError(
            f"critic-agent runtime.cli {call.phase} timed out after {CRITIC_AGENT_RUNTIME_TIMEOUT_SEC}s "
            f"(cwd={call.cwd})"
        ) from exc
    except FileNotFoundError as exc:
        raise BackendError(
            f"critic-agent runtime.cli {call.phase} could not start (python={sys.executable!r}, cwd={call.cwd}): {exc}"
        ) from exc

    if proc.returncode != 0:
        raise BackendError(
            f"critic-agent runtime.cli {call.phase} exited rc={proc.returncode}: "
            f"stderr={proc.stderr.strip()[:_STDERR_TRUNCATE_CHARS]!r}"
        )


def _reviewed_msg_ids_from_bundle(judge_bundle: dict[str, Any]) -> list[str] | None:
    """Pull the proposal ``msg_id``s out of a judge bundle, or ``None``."""
    proposals = judge_bundle.get("proposals") if isinstance(judge_bundle, dict) else None
    if not isinstance(proposals, list):
        return None
    out: list[str] = []
    seen: set[str] = set()
    for p in proposals:
        if not isinstance(p, dict):
            continue
        mid = str(p.get("msg_id") or "").strip()
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out or None


def _review_subjects(judge_bundle: dict[str, Any]) -> dict[str, str]:
    """Map each reviewed proposal's message id to the row it is recorded under.

    The two arms identify a proposal differently -- a configuration grid by the
    bus message that raised it, an upstream candidate by its candidate id --
    and evidence filed under the wrong one opens a second, near-empty row
    beside the proposal it was about.

    Args:
        judge_bundle (dict[str, Any]): The bundle of proposals reviewed.

    Returns:
        dict[str, str]: ``{msg_id: row_id}``, holding only the proposals whose
            row id is not their message id.
    """
    out: dict[str, str] = {}
    for proposal in judge_bundle.get("proposals") or []:
        if not isinstance(proposal, dict):
            continue
        msg_id = str(proposal.get("msg_id") or "")
        payload = proposal.get("payload") if isinstance(proposal.get("payload"), dict) else {}
        row_id = review_row_id(payload)
        if msg_id and row_id:
            out[msg_id] = row_id
    return out


def _kb_writes_by_target(emit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """What the commit emit reports about each KB write, keyed by the proposal the write was asked for."""
    writes: dict[str, dict[str, Any]] = {}
    for write in emit.get("kb_writes") or []:
        if not isinstance(write, dict):
            continue
        target = str(write.get("target_proposal_msg_id") or "")
        if not target:
            continue
        result = write.get("result") if isinstance(write.get("result"), dict) else {}
        writes[target] = {
            "trigger": str(write.get("trigger") or ""),
            "status": str(result.get("status") or ""),
            "detail": str(result.get("detail") or result.get("error") or ""),
        }
    return writes


def _verdict_references_kb(review: dict[str, Any] | None) -> bool:
    """Whether any final review verdict cites KB evidence."""
    if not isinstance(review, dict):
        return False
    for v in review.get("review_verdicts") or []:
        if isinstance(v, dict) and v.get("kb_evidence"):
            return True
    return False


def _proposal_lever(proposal: Any) -> str:
    """The lever a reviewed proposal moves: its own markers first, then its params'."""
    if not isinstance(proposal, dict):
        return ""
    lever = patch_lever_kind(proposal)
    if not lever:
        params = proposal.get("params")
        lever = patch_lever_kind(params if isinstance(params, dict) else None)
    return lever


def _ruling(turn: BrainTurn) -> dict[str, Any]:
    """The ruling a Critic turn asks to record; a turn that records none rules on nothing."""
    names = [call.name for call in turn.tool_calls]
    if any(name != "verdict.record" for name in names) or len(names) > 1:
        raise BackendError(
            f"CriticAgentBackend: a critic turn records one ruling and nothing else; the brain asked for {names}"
        )
    return dict(turn.tool_calls[0].args) if turn.tool_calls else {}


class _CriticReviewModel:
    """The review model a Critic turn is lent: the backend's transport, its trace rows and its call ids."""

    role = "critic"

    def __init__(self, backend: "CriticAgentBackend", port: "_CriticTurnPort") -> None:
        self._backend = backend
        self._port = port
        self._calls = 0

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        prefix: str | None = None,
        max_tokens: int | None = None,
        tools: Any = None,
        max_turns: int | None = None,
    ) -> LlmReply:
        """One review call; a failed one reaches the brain as :class:`LlmCallError`."""
        del tools, max_turns
        # The turn's first call takes the id of an ``llm.call`` span its caller opened; a retry gets its own.
        call_id = (current_context().call_id if self._calls == 0 else None) or new_call_id()
        self._calls += 1
        try:
            text, finish = await self._backend._review_call(
                system_prompt=system_prompt,
                preamble=prefix or "",
                review_prompt=prompt,
                max_tokens=int(max_tokens or CRITIC_AGENT_MAX_COMPLETION_TOKENS),
                call_id=call_id,
            )
        except BackendError as exc:
            raise self._port.own(LlmCallError(str(exc))) from exc
        return LlmReply(raw_text=text, finish_reason=finish)


class _CriticTurnPort:
    """What one Critic turn lends the brain: the review model, and the judge bundle prepare-review builds."""

    def __init__(
        self,
        backend: "CriticAgentBackend",
        *,
        request_path: Path,
        judge_path: Path,
        env: dict[str, str],
    ) -> None:
        self._backend = backend
        self._request_path = request_path
        self._judge_path = judge_path
        self._env = env
        self._owned: list[TurnFailed] = []
        self._levers: list[str] = []
        #: The bundle prepare-review wrote, once the brain has read it.
        self.judge_bundle: dict[str, Any] | None = None
        self.llm = {"critic": _CriticReviewModel(backend, self)}

    async def call(self, call: ToolCall) -> dict[str, Any]:
        """Answer ``kb.critic.read`` with the judge bundle and the lever of each proposal in it."""
        if call.name != "kb.critic.read":
            raise TurnFailed(f"CriticAgentBackend: a critic turn cannot call {call.name!r}")
        if self.judge_bundle is None:
            try:
                bundle = await self._backend._prepare_review(
                    request_path=self._request_path, judge_path=self._judge_path, env=self._env
                )
            except BackendError as exc:
                raise self.own(TurnFailed(str(exc))) from exc
            proposals = bundle.get("proposals") if isinstance(bundle, dict) else None
            self._levers = [_proposal_lever(p) for p in proposals] if isinstance(proposals, list) else []
            self.judge_bundle = bundle
        return {"judge_bundle": json_value(self.judge_bundle, "judge bundle"), "proposal_levers": list(self._levers)}

    def own(self, failure: TurnFailed) -> TurnFailed:
        """Remember ``failure`` as one this port raised for a body error, so the body's error can be restored."""
        self._owned.append(failure)
        return failure

    def body_error(self, failure: TurnFailed) -> BaseException:
        """The error the backend raises for a failed brain turn.

        A failure this port raised and the brain let through is the body's own error, raised as it was. Any other is
        the brain's: a :class:`BackendError` with its message, caused by the body error underneath it, if any.
        """
        if any(failure is owned for owned in self._owned) and failure.__cause__ is not None:
            return failure.__cause__
        cause = failure.__cause__
        while cause is not None and not isinstance(cause, BackendError):
            cause = cause.__cause__
        error = BackendError(str(failure))
        error.__cause__ = cause
        return error


@dataclass
class CriticAgentBackend:
    """Real Critic backend that drives the critic-agent runtime."""

    critic_agent_root: Path
    session_dir: Path
    codex_model: str = DEFAULT_CODEX_MODEL
    codex_client_factory: Callable[[], Any] | None = None
    kb_mode: Literal["inmemory", "live"] = "inmemory"
    runtime_caller_factory: Callable[[], RuntimeCaller] | None = None
    # Per-action verdict policy enriched onto ``review_constraints.action_verdict_policy`` post prepare-review.
    action_verdict_policy: dict[str, str] = field(default_factory=dict)
    name: str = "critic-agent"
    # Review inference protocol.
    protocol: Literal["openai", "anthropic"] = "openai"
    # Claude model id used when ``protocol == "anthropic"`` (falls back to ``codex_model`` when unset).
    claude_model: str | None = None

    # ``_runtime_caller`` is assigned on the instance in __post_init__ (not as a dataclass field) to avoid descriptor
    # binding as a method.
    _client: Any = field(default=None, init=False, repr=False)
    _turn_idx: int = field(default=0, init=False, repr=False)
    # Trace context the Coordinator stamps before each reactor ``run()`` so the critic's self-written llm_calls row
    # carries the timeline keys.
    _trace_tick: int | None = field(default=None, init=False, repr=False)
    _trace_phase: str | None = field(default=None, init=False, repr=False)
    _trace_macro_cycle: int | None = field(default=None, init=False, repr=False)
    # Proposal msg_ids reviewed by the current turn, snapshotted for llm_calls attribution.
    _trace_reviewed_msg_ids: list[str] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _skill_preamble: str | None = field(default=None, init=False, repr=False)
    _static_context: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    # Resolved review model id (protocol-aware).
    _review_model: str = field(default="", init=False, repr=False)
    # What the brain returned from its last Critic turn, handed back on the next.
    _brain_state: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate config, wire transports, and resolve static context."""
        self.critic_agent_root = Path(self.critic_agent_root)
        self.session_dir = Path(self.session_dir)
        if not (self.critic_agent_root / "runtime" / "cli.py").is_file():
            raise BackendError(
                f"CriticAgentBackend: runtime/cli.py not found under "
                f"{self.critic_agent_root!s} — set CRITIC_AGENT_ROOT or "
                f"check the install"
            )
        if self.kb_mode not in ("inmemory", "live"):
            raise BackendError(f"CriticAgentBackend: kb_mode={self.kb_mode!r} not in {{'inmemory','live'}}")

        if self.runtime_caller_factory is not None:
            object.__setattr__(
                self,
                "_runtime_caller",
                self.runtime_caller_factory(),
            )
        else:
            object.__setattr__(
                self,
                "_runtime_caller",
                _default_runtime_caller,
            )

        self._review_model = (
            (self.claude_model or self.codex_model) if self.protocol == "anthropic" else self.codex_model
        )
        if self.protocol == "anthropic":
            self._require_anthropic_transport()
        elif self.codex_client_factory is not None:
            self._client = self.codex_client_factory()
        else:
            connect_timeout_s, rw_timeout_s = self._resolve_llm_timeouts()
            try:
                self._client = get_async_openai_client(
                    timeout=build_http_timeout(connect=connect_timeout_s, read=rw_timeout_s),
                )
            except LLMConfigError as exc:
                raise BackendError(
                    str(exc).replace(
                        "OpenAI-compatible client",
                        "CriticAgentBackend cannot reach Codex for review reasoning",
                    )
                ) from exc

        # Resolve static per-session context once.
        self._static_context = self._load_static_context_from_manifest()
        log.info("critic_agent_backend static_context keys=%s", sorted(self._static_context.keys()))

    @staticmethod
    def _resolve_llm_timeouts() -> tuple[float, float]:
        """Return the ``(connect, read/write/pool)`` review-call timeouts in seconds."""
        return (
            parse_call_timeout_env(
                "CRITIC_AGENT_LLM_CONNECT_TIMEOUT_S",
                default=CRITIC_AGENT_LLM_CONNECT_TIMEOUT_SEC,
            ),
            parse_call_timeout_env(
                "CRITIC_AGENT_LLM_RW_TIMEOUT_S",
                default=CRITIC_AGENT_LLM_RW_TIMEOUT_SEC,
            ),
        )

    @staticmethod
    def _resolve_max_completion_tokens() -> int:
        """Return the output-token cap one review call may spend."""
        raw = os.environ.get("CRITIC_AGENT_MAX_COMPLETION_TOKENS")
        if raw is None or not raw.strip():
            return CRITIC_AGENT_MAX_COMPLETION_TOKENS
        value = optional_positive_int(raw)
        if value is None:
            log.warning(
                "CRITIC_AGENT_MAX_COMPLETION_TOKENS=%r is not a positive integer; using default %d",
                raw,
                CRITIC_AGENT_MAX_COMPLETION_TOKENS,
            )
            return CRITIC_AGENT_MAX_COMPLETION_TOKENS
        return value

    def _require_anthropic_transport(self) -> None:
        """Fail fast when the Anthropic side cannot serve a review call."""
        if anthropic_transport_ready():
            return
        raise BackendError(
            "CriticAgentBackend(protocol=anthropic) review reasoning requires a usable "
            "Anthropic transport: an Anthropic-side credential (ANTHROPIC_API_KEY / "
            "ANTHROPIC_AUTH_TOKEN / CLAUDE_CODE_OAUTH_TOKEN), plus the claude CLI when "
            "that credential is a subscription token"
        )

    # Public API — Backend.run
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        """One Critic turn — run the prepare → reason → commit pipeline."""
        del tools, max_turns  # Critic is single-turn / no tool palette.

        turn_idx = self._turn_idx
        self._turn_idx += 1

        workdir = allocate_turn_workdir(
            self.session_dir, "critic-workdir", turn_idx, keep=CRITIC_AGENT_WORKDIR_KEEP_COUNT
        )
        request_path = workdir / "request.json"
        judge_path = workdir / "judge_bundle.json"
        review_path = workdir / "review.json"
        emit_path = workdir / "emit.json"

        session_id = self.session_dir.name
        request = self._write_review_request(request_path, prompt)

        env = self._build_runtime_env()

        # The brain judges; this turn's port runs prepare-review when it reads the bundle.
        port = _CriticTurnPort(self, request_path=request_path, judge_path=judge_path, env=env)
        ruling = await self._ask_brain(port, system_prompt)
        judge_bundle = ruling.get("judge_bundle")
        if not isinstance(judge_bundle, dict):
            judge_bundle = port.judge_bundle or {}
        review = ruling.get("review") if isinstance(ruling.get("review"), dict) else {"review_verdicts": []}
        llm_text = str(ruling.get("raw_text") or "")
        llm_finish = ruling.get("finish")
        proposals = judge_bundle.get("proposals") or []

        review_path.write_text(
            json.dumps(review, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # commit-review.
        await asyncio.to_thread(
            self._runtime_caller,
            RuntimeCall(
                phase="commit-review",
                request_path=request_path,
                review_path=review_path,
                out_path=emit_path,
                cwd=self.critic_agent_root,
                env=env,
            ),
        )
        try:
            emit = json.loads(emit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"CriticAgentBackend: failed to read emit.json from {emit_path}: {exc}") from exc

        envelope = emit.get("intent_envelope")
        if not isinstance(envelope, dict):
            raise BackendError(
                f"CriticAgentBackend: emit.json missing intent_envelope (got keys={sorted(emit.keys())!r})"
            )
        try:
            intents = validate_envelope(envelope)
        except IntentValidationError as exc:
            raise NoIntentEmitted(f"critic_agent_envelope_invalid: {exc}") from exc

        kb_skipped = judge_bundle.get("kb_read_skipped_reason")
        required_context = list(judge_bundle.get("required_context") or [])
        verdicts_summary = [
            (i.payload.get("verdict"), i.payload.get("source")) for i in intents if i.type.value == "review_verdict"
        ]
        kb_priors_trace = self._build_kb_priors_trace(judge_bundle, review)
        log.info(
            "critic_agent_backend turn=%d session=%s proposals=%d "
            "verdicts=%s kb_skipped=%s required_context=%s finish=%s kb_priors=%d",
            turn_idx,
            session_id,
            len(proposals),
            verdicts_summary,
            kb_skipped,
            required_context,
            llm_finish,
            kb_priors_trace.get("prior_count") or 0,
        )
        from hyperloom.inference_optimizer.breakdown.recorder import critic_out

        iteration = critic_out.CriticIteration(
            iter_n=turn_idx,
            request=request,
            judge_bundle=judge_bundle,
            review=review,
            emit=emit,
            workdir=workdir,
            kb_priors=kb_priors_trace,
        )
        # Record this critic iteration before the workdir can be pruned.
        critic_out.record_critic_iteration(self.session_dir, iteration)

        # Attach what each ruling was grounded in to the ruling itself, on the
        # proposal it judged. Done here because these are the turn's own facts:
        # the artifacts are this runtime's files, and a KB write's result only
        # comes back on the emit.
        self._record_review_evidence(iteration)

        # Mirror the KB integration trace into Langfuse (opt-in, best-effort).
        self._mirror_kb_trace_to_langfuse(
            turn_idx=turn_idx,
            kb_priors=kb_priors_trace,
        )

        return BackendTurnResult(
            intents=intents,
            raw_text=llm_text,
            metadata={"model": self._review_model},
        )

    # Helpers

    def _write_review_request(self, request_path: Path, prompt: str) -> dict[str, Any]:
        """Write the coordinator-inbox request prepare-review reads, and return what it was given."""
        # prepare-review runs as a subprocess; the phase reaches it via context.
        context = dict(self._static_context)
        if self._trace_phase:
            context["phase"] = str(self._trace_phase).strip().upper()
        if self._trace_macro_cycle is not None:
            context["macro_cycle"] = self._trace_macro_cycle
        request: dict[str, Any] = {
            "session_id": self.session_dir.name,
            "raw_prompt": prompt,
            "context": context,
        }
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return request

    def _load_static_context_from_manifest(self) -> dict[str, Any]:
        """Derive per-session context for ``request.context`` from manifest.json (model / framework / gpu_type / model_path / tp / workload / precision); empty values dropped."""
        path = manifest_path(self.session_dir)
        try:
            raw = path.read_text(encoding="utf-8")
            manifest = json.loads(raw)
        except FileNotFoundError:
            log.warning(
                "critic_agent_backend: manifest.json not found at %s — "
                "request.context will be empty; critic-agent runtime will "
                "report missing_critical_context for every verdict",
                path,
            )
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "critic_agent_backend: failed to load manifest.json at %s (%s: %s); request.context will be empty",
                path,
                type(exc).__name__,
                exc,
            )
            return {}

        ctx: dict[str, Any] = {}
        # CRITICAL keys — runtime hard-fails the verdict if either is missing.
        if manifest.get("model_name"):
            ctx["model"] = manifest["model_name"]
        if manifest.get("framework"):
            ctx["framework"] = manifest["framework"]
        if manifest.get("gpu_type"):
            ctx["gpu_type"] = manifest["gpu_type"]
        if manifest.get("model_path"):
            ctx["model_path"] = manifest["model_path"]
        tp = manifest.get("tp")
        if isinstance(tp, int) and tp > 0:
            ctx["tp"] = tp
        workload = manifest.get("workload")
        if isinstance(workload, dict):
            cleaned = {k: v for k, v in workload.items() if v not in (None, "")}
            if cleaned:
                ctx["workload"] = cleaned
                if cleaned.get("precision"):
                    ctx["precision"] = cleaned["precision"]
        return ctx

    def _build_runtime_env(self) -> dict[str, str]:
        """Build the subprocess environment for ``runtime.cli`` invocations."""
        env = dict(os.environ)
        # Co-locate session memory inside the Coordinator session.
        memory_dir = self.session_dir / "critic-session-memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("CRITIC_SESSION_MEMORY_DIR", str(memory_dir))
        env["CRITIC_KB_CLIENT_MODE"] = self.kb_mode

        # Dead-letter dir under the session so cron replays don't cross sessions.
        dlq_dir = self.session_dir / "critic-kb-dead-letter"
        env.setdefault("KB_DEAD_LETTER_DIR", str(dlq_dir))

        if self.kb_mode == "live":
            if not env.get("KB_BASE_URL"):
                raise BackendError("CriticAgentBackend: kb_mode=live but KB_BASE_URL is not set (export KB_BASE_URL)")
        return env

    def _mirror_kb_trace_to_langfuse(
        self,
        *,
        turn_idx: int,
        kb_priors: dict[str, Any],
    ) -> None:
        """Mirror the per-iteration KB trace into Langfuse (best-effort)."""
        try:
            from hyperloom.inference_optimizer.trace.langfuse_emitter import get_emitter

            emitter = get_emitter(self.session_dir)
            if not emitter.enabled:
                return
            if kb_priors:
                emitter.record_kb_span(
                    name=f"kb_priors:iter_{turn_idx}",
                    agent="critic",
                    output=kb_priors,
                    metadata={
                        "kind": "kb_priors",
                        "iter": turn_idx,
                        "prior_count": kb_priors.get("prior_count") or 0,
                        "referenced_in_verdict": bool(kb_priors.get("referenced_in_verdict")),
                    },
                )
        except Exception:
            log.debug("critic_agent: langfuse kb mirror failed", exc_info=True)

    def _record_review_evidence(self, iteration: CriticIteration) -> None:
        """Record what each of this turn's rulings was grounded in.

        Written onto the proposal each verdict targets, because a ruling and
        its grounds are one fact about one proposal: the alternative is a
        per-turn stream a reader has to join back to the proposals, keyed on a
        turn index that resume reuses.

        The KB write is matched to its verdict by target, not spread across
        them: the Critic asks for a lesson to be persisted per verdict, and a
        turn that reviewed six proposals and wrote one lesson would otherwise
        report the write six times.

        Args:
            iteration (CriticIteration): The turn's review request (read for
                the cycle), reviewed bundle (read for the row each verdict's
                target is recorded under), parsed review, commit emit (read for
                the KB writes), workdir holding the artifacts, and priors trace.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import record_review_evidence

        context = iteration.request.get("context") if isinstance(iteration.request.get("context"), dict) else {}
        macro_cycle = context.get("macro_cycle")
        if macro_cycle is None:
            return
        subjects = _review_subjects(iteration.judge_bundle)
        writes = _kb_writes_by_target(iteration.emit)
        artifacts = {
            name: str(iteration.workdir / filename)
            for name, filename in (
                ("request_path", "request.json"),
                ("judge_bundle_path", "judge_bundle.json"),
                ("review_path", "review.json"),
                ("emit_path", "emit.json"),
            )
        }
        for verdict in (iteration.review or {}).get("review_verdicts") or []:
            if not isinstance(verdict, dict):
                continue
            target = str(verdict.get("target_proposal_msg_id") or "")
            if not target:
                continue
            kb: dict[str, Any] = {"persist_requested": bool(verdict.get("persist_to_kb"))}
            if iteration.kb_priors:
                kb["priors"] = iteration.kb_priors
            if target in writes:
                kb["write"] = writes[target]
            record_review_evidence(
                macro_cycle=macro_cycle,
                proposal_id=subjects.get(target) or target,
                artifacts=artifacts,
                kb=kb,
            )

    @staticmethod
    def _build_kb_priors_trace(
        judge_bundle: dict[str, Any],
        review: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Assemble the historical-priors KB trace for one critic iteration."""
        trace = dict(judge_bundle.get("kb_priors_trace") or {})
        by_proposal = judge_bundle.get("kb_priors_by_proposal") or {}
        skipped = judge_bundle.get("kb_read_skipped_reason")
        if not trace and not by_proposal and not skipped:
            return {}
        total = sum(len(v) for v in by_proposal.values() if isinstance(v, list))
        return {
            "configured": bool(trace.get("configured")),
            "mode": trace.get("mode"),
            "client_mode": trace.get("client_mode"),
            "scope_filter": trace.get("scope_filter") or {},
            "limit": trace.get("limit"),
            "requests": trace.get("requests") or [],
            "skipped_reason": skipped,
            "prior_count": total,
            "referenced_in_verdict": _verdict_references_kb(review),
        }

    async def _ask_brain(self, port: "_CriticTurnPort", system_prompt: str | None) -> dict[str, Any]:
        """Run the brain's Critic turn and return the ruling it asks to record.

        Raises:
            BackendError: the body's own error when one of its calls failed under the brain, else one carrying the
                brain's reason for failing the turn.
        """
        failure: BaseException | None = None
        try:
            batch = TickBatch(
                call_site=CallSite.LLM_TURN,
                tick=self._trace_tick or 0,
                now_unix=time.time(),
                events=(),
                snapshot=StateSnapshot(
                    phase=self._trace_phase or "",
                    data={"critic": self._brain_inputs(system_prompt)},
                ),
                brain_state=self._brain_state,
            )
            turn = await resolve_brain().on_llm_turn(batch, port)
        except TurnFailed as exc:
            failure = port.body_error(exc)
        except WireError as exc:
            failure = BackendError(f"CriticAgentBackend: {exc}")
            failure.__cause__ = exc
        if failure is not None:
            # Raised outside the handler, so a restored body error keeps the chain it was raised with.
            raise failure
        self._brain_state = turn.brain_state
        return _ruling(turn)

    def _brain_inputs(self, system_prompt: str | None) -> dict[str, Any]:
        """What the brain's Critic turn is given: its prompts, its output cap, and the review rules this body enforces."""
        from ..specialists.patch_safety import (
            ADVISORY_RULE_PROPOSAL_KINDS,
            SCOPE_DOMAINS_LITERAL,
            cross_domain_rule_descriptors,
            quantitative_claim_rule_descriptor,
        )

        return {
            "system_prompt": system_prompt,
            "skill_preamble": self._load_skill_preamble(),
            "max_completion_tokens": self._resolve_max_completion_tokens(),
            "action_verdict_policy": dict(self.action_verdict_policy),
            "rules": {
                "scope_domains": SCOPE_DOMAINS_LITERAL,
                "cross_domain_rules": cross_domain_rule_descriptors(),
                "quantitative_claim_rule": quantitative_claim_rule_descriptor(),
                "advisory_rule_actions": sorted(ADVISORY_RULE_PROPOSAL_KINDS),
            },
        }

    async def _prepare_review(self, *, request_path: Path, judge_path: Path, env: dict[str, str]) -> dict[str, Any]:
        """Run prepare-review and return the judge bundle it wrote."""
        # prepare-review runs off-thread because subprocess.run blocks.
        await asyncio.to_thread(
            self._runtime_caller,
            RuntimeCall(
                phase="prepare-review",
                request_path=request_path,
                review_path=None,
                out_path=judge_path,
                cwd=self.critic_agent_root,
                env=env,
            ),
        )
        try:
            judge_bundle = json.loads(judge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"CriticAgentBackend: failed to read judge_bundle from {judge_path}: {exc}") from exc

        # Snapshot the reviewed proposal msg_ids for trace attribution.
        self._trace_reviewed_msg_ids = _reviewed_msg_ids_from_bundle(judge_bundle)
        return judge_bundle

    async def _review_call(
        self,
        *,
        system_prompt: str | None,
        preamble: str,
        review_prompt: str,
        max_tokens: int,
        call_id: str,
    ) -> tuple[str, str | None]:
        """One review model call, recorded on conversations.jsonl once it answers."""
        text, finish = await self._run_reasoning_loop(
            system_prompt=system_prompt,
            preamble=preamble,
            review_prompt=review_prompt,
            max_tokens=max_tokens,
            call_id=call_id,
        )
        # Mirror the full prompt + reply onto conversations.jsonl so the critic turn is replayable.
        self._record_critic_conversation(
            system_prompt=system_prompt,
            user_prompt=f"{preamble}\n\n{review_prompt}",
            response=text,
            call_id=call_id,
        )
        return text, finish

    async def _run_reasoning_loop(
        self,
        *,
        system_prompt: str | None,
        preamble: str,
        review_prompt: str,
        max_tokens: int,
        call_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Issue one review inference call and return ``(text, finish_reason)``."""
        if self.protocol == "anthropic":
            return await self._run_anthropic_reasoning(
                system_prompt=system_prompt,
                preamble=preamble,
                review_prompt=review_prompt,
                max_tokens=max_tokens,
                call_id=call_id,
            )
        return await self._run_openai_reasoning(
            system_prompt=system_prompt,
            user_prompt=f"{preamble}\n\n{review_prompt}",
            max_tokens=max_tokens,
            call_id=call_id,
        )

    async def _run_openai_reasoning(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        max_tokens: int,
        call_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Issue one Codex chat-completions call and return ``(text, finish_reason)``."""
        kwargs: dict[str, Any] = {
            "model": self._review_model,
            "messages": build_chat_messages(system_prompt, user_prompt),
            "max_completion_tokens": max_tokens,
        }
        apply_reasoning_effort(kwargs)
        usage_acc = {"input_tokens": 0, "output_tokens": 0}
        _t0 = time.perf_counter()
        try:
            result = await achat_completion(
                self._client,
                component="critic",
                operation="review",
                **kwargs,
            )
        except Exception as exc:
            raise self._llm_call_failed(
                f"Codex API call failed (critic-agent reasoning): {exc!r}",
                latency_ms=int((time.perf_counter() - _t0) * 1000),
                call_id=call_id,
            ) from exc
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        _accumulate_usage(usage_acc, openai_usage_counts(result.usage), result.usage)
        self._trace_critic_llm_call(usage_acc, latency_ms=latency_ms, call_id=call_id)
        return result.text, result.finish_reason

    async def _run_anthropic_reasoning(
        self,
        *,
        system_prompt: str | None,
        preamble: str,
        review_prompt: str,
        max_tokens: int,
        call_id: str | None = None,
    ) -> tuple[str, str | None]:
        """Issue one single-shot Anthropic completion for the review."""
        content = []
        if preamble:
            content.append({"type": "text", "text": preamble, "cache_control": {"type": "ephemeral"}})
        content.append({"type": "text", "text": review_prompt})
        connect_timeout_s, rw_timeout_s = self._resolve_llm_timeouts()
        _t0 = time.perf_counter()
        try:
            request = AnthropicCompletionRequest(
                model=self._review_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                system=system_prompt,
                timeout=build_http_timeout(connect=connect_timeout_s, read=rw_timeout_s),
                timeout_s=rw_timeout_s,
            )
            result = await aanthropic_completion(request=request, component="critic", operation="review")
        except Exception as exc:
            raise self._llm_call_failed(
                f"Anthropic completion failed (critic-agent reasoning): {exc!r}",
                latency_ms=int((time.perf_counter() - _t0) * 1000),
                call_id=call_id,
            ) from exc
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        usage_acc = {"input_tokens": 0, "output_tokens": 0}
        _accumulate_usage(usage_acc, anthropic_usage_counts(result.usage), result.usage)
        self._trace_critic_llm_call(usage_acc, latency_ms=latency_ms, call_id=call_id)
        stop_reason = result.stop_reason
        return (result.text or "", stop_reason if isinstance(stop_reason, str) and stop_reason else None)

    def set_trace_context(
        self,
        *,
        tick: int | None = None,
        phase: str | None = None,
        macro_cycle: int | None = None,
    ) -> None:
        """Stamp the timeline keys for the next reactor turn and request."""
        self._trace_tick = to_int(tick)
        self._trace_phase = (str(phase) or None) if phase else None
        self._trace_macro_cycle = to_int(macro_cycle)

    def _trace_critic_llm_call(
        self,
        usage_acc: dict[str, int],
        *,
        latency_ms: int | None = None,
        call_id: str | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a critic reasoning loop."""
        try:
            record = LLMCallRecord(
                session_id=self.session_dir.name,
                component="critic",
                role="critic",
                call_id=call_id,
                model=self._review_model,
                tick=self._trace_tick,
                phase=self._trace_phase,
                input_tokens=usage_acc.get("input_tokens"),
                output_tokens=usage_acc.get("output_tokens"),
                cache_read_input_tokens=usage_acc.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage_acc.get("cache_creation_input_tokens"),
                reasoning_output_tokens=usage_acc.get("reasoning_output_tokens"),
                latency_ms=latency_ms,
                reviewed_msg_ids=self._trace_reviewed_msg_ids,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:
            log.debug(
                "full-trace: critic llm_call append failed",
                exc_info=True,
            )

    def _llm_call_failed(
        self,
        message: str,
        *,
        latency_ms: int | None = None,
        call_id: str | None = None,
    ) -> LLMCallFailed:
        """Record a failed review-model call and return the error to raise."""
        error = LLMCallFailed(message)
        self._trace_llm_failure(error, latency_ms=latency_ms, call_id=call_id)
        return error

    def _trace_llm_failure(
        self,
        error: BaseException,
        *,
        latency_ms: int | None = None,
        call_id: str | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a call that never returned."""
        try:
            site = LLMCallSite(
                session_id=self.session_dir.name,
                component="critic",
                role="critic",
                tick=self._trace_tick,
                phase=self._trace_phase,
            )
            record = LLMCallRecord.for_failure(
                site, error, model=self._review_model, call_id=call_id, latency_ms=latency_ms
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:
            log.debug(
                "full-trace: critic llm_call failure append failed",
                exc_info=True,
            )

    def _record_critic_conversation(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        response: str,
        call_id: str | None = None,
    ) -> None:
        """Append one ``conversations.jsonl`` row for a critic reasoning loop."""
        prompt = f"{system_prompt}\n---\n{user_prompt}" if system_prompt else user_prompt
        if not prompt and not response:
            return
        record = ConversationRecord(
            session_id=self.session_dir.name,
            component="critic",
            role="critic",
            call_id=call_id,
            model=self._review_model,
            prompt=prompt or "",
            response=response or "",
        )
        append_conversation(session_dir=self.session_dir, record=record)

    def _load_skill_preamble(self) -> str:
        """Load and cache the critic-agent skill/action markdown preamble."""
        if self._skill_preamble is not None:
            return self._skill_preamble
        parts: list[str] = []
        for rel in ("SKILL.md", "actions/review_coordinator_inbox.md"):
            path = self.critic_agent_root / rel
            try:
                parts.append(f"==== {rel} ====\n{path.read_text(encoding='utf-8').strip()}")
            except OSError:
                continue
        self._skill_preamble = "\n\n".join(parts) if parts else ""
        return self._skill_preamble


__all__ = [
    "CRITIC_AGENT_MAX_COMPLETION_TOKENS",
    "CRITIC_AGENT_RUNTIME_TIMEOUT_SEC",
    "CRITIC_AGENT_WORKDIR_KEEP_COUNT",
    "CriticAgentBackend",
    "RuntimeCall",
    "RuntimeCaller",
    "_default_runtime_caller",
]

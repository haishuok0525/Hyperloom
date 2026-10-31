# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ProposalScorer — body organ that lends models and traces while the brain scores proposals."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple

from hyperloom.brain_api import CallSite, LlmReply
from hyperloom.common.llm_config import (
    DEFAULT_CLAUDE_MODEL,
    LLMConfigError,
    apply_reasoning_effort,
    astream_chat_completion_text,
    get_async_openai_client,
)
from hyperloom.common.token_usage import openai_usage_counts
from hyperloom.inference_optimizer.trace.conversation_trace import ConversationRecord, append_conversation
from hyperloom.inference_optimizer.trace.llm_trace import LLMCallRecord, LLMCallSite, append_llm_call, new_call_id
from hyperloom.inference_optimizer.trace.parse_usage import reasoning_output_tokens
from hyperloom.orchestrator.brain_link import DecisionPort, ask_brain, one_decision
from hyperloom.orchestrator.roles.base import parse_call_timeout_env

log = logging.getLogger(__name__)

#: Snapshot key the brain reads a scoring ask under (must match ``brain.score.PROPOSAL_SCORE_KEY``).
_PROPOSAL_SCORE_KEY = "proposal_score"

DEFAULT_SCORER_MODELS: tuple[str, ...] = (
    DEFAULT_CLAUDE_MODEL,
    "gpt-5.6-sol",
    # Gemini MUST carry the ``gemini/`` prefix (bare slug routes to a broken Vertex ADC path).
    "dvue-aoai-005-Kimi-K2.6",
    "gemini/gemini-3.1-pro-preview",
)


class ScorerCallKeys(NamedTuple):
    """The task, tick and phase a scoring call is attributed to in the traces."""

    task_id: str | None = None
    tick: int | None = None
    phase: str | None = None


class _ScorerModel:
    """One gateway model lent to the brain for a scoring call; the body owns credentials and traces."""

    def __init__(self, scorer: "ProposalScorer", model: str, keys: ScorerCallKeys) -> None:
        self.role = model
        self._scorer = scorer
        self._model = model
        self._keys = keys

    async def run(self, prompt: str, **kwargs: Any) -> LlmReply:
        """Stream one scoring completion; failures raise so the brain records them per model."""
        client = self._scorer._ensure_client()
        messages = [{"role": "user", "content": prompt}]
        _t0 = time.perf_counter()
        create_params = apply_reasoning_effort(
            {
                "model": self._model,
                "messages": messages,
                "max_completion_tokens": int(kwargs.get("max_tokens") or self._scorer.max_completion_tokens),
            }
        )
        try:
            text, usage = await asyncio.wait_for(
                astream_chat_completion_text(
                    client,
                    component="proposal_scorer",
                    operation="score_proposal",
                    **create_params,
                ),
                timeout=self._scorer.call_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            error = RuntimeError(f"timed out after {self._scorer.call_timeout_s:.0f}s")
            self._scorer._trace_scorer_llm_failure(
                self._model, error, self._keys, latency_ms=int((time.perf_counter() - _t0) * 1000)
            )
            raise error from exc
        except Exception as exc:
            self._scorer._trace_scorer_llm_failure(
                self._model, exc, self._keys, latency_ms=int((time.perf_counter() - _t0) * 1000)
            )
            raise
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        call_id = new_call_id()
        self._scorer._trace_scorer_llm_call(self._model, usage, self._keys, latency_ms=latency_ms, call_id=call_id)
        self._scorer._record_scorer_conversation(self._model, prompt, text, self._keys, call_id=call_id)
        return LlmReply(raw_text=text)


@dataclass
class ProposalScorer:
    """Score proposals without ranking or selecting them; the brain decides, this organ calls models."""

    models: tuple[str, ...] = DEFAULT_SCORER_MODELS
    # Large cap so reasoning raters can finish reasoning and still emit the scores JSON.
    max_completion_tokens: int = 4096
    call_timeout_s: float = field(
        default_factory=lambda: parse_call_timeout_env(
            "INFERENCE_OPTIMIZER_PROPOSAL_SCORER_TIMEOUT_SEC",
            default=120.0,
        )
    )

    # Test seam — set to bypass real OpenAI client construction.
    client_factory: Callable[[], Any] | None = None

    # When set, each call appends token usage to the trace; ``None`` disables trace writes.
    session_dir: Path | None = None

    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize the model list and optionally eager-build the client."""
        self.models = tuple(m for m in (str(x).strip() for x in (self.models or ())) if m)
        if len(self.models) != len(set(self.models)):
            raise ValueError("duplicate scorer model slug(s) in models")
        if self.client_factory is not None:
            self._client = self.client_factory()
            return
        # Lazy: construct on first use so an unconfigured env degrades per-call.
        self._client = None

    def _ensure_client(self) -> Any:
        """Return the cached client, constructing one on first use."""
        if self._client is not None:
            return self._client
        try:
            self._client = get_async_openai_client()
        except LLMConfigError as exc:
            raise RuntimeError(str(exc).replace("OpenAI-compatible client", "ProposalScorer")) from exc
        return self._client

    def _trace_scorer_llm_call(
        self,
        model: str,
        usage: Any,
        keys: ScorerCallKeys,
        *,
        latency_ms: int | None = None,
        call_id: str | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a proposal-scoring call."""
        if self.session_dir is None:
            return
        try:
            counts = openai_usage_counts(usage)
            record = LLMCallRecord(
                session_id=self.session_dir.name,
                component="proposal_scorer",
                role="proposal_scorer",  # must match the conversation row's role
                call_id=call_id,
                model=str(model),
                task_id=keys.task_id,
                tick=keys.tick,
                phase=keys.phase,
                input_tokens=counts.get("input_tokens"),
                output_tokens=counts.get("output_tokens"),
                cache_read_input_tokens=counts.get("cache_read_input_tokens"),
                cache_creation_input_tokens=counts.get("cache_creation_input_tokens"),
                # A scoring model with a reasoning split bills it separately; reading it here keeps the ledger
                # consistent with the backends that report it on their turn metadata.
                reasoning_output_tokens=reasoning_output_tokens(usage),
                latency_ms=latency_ms,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:
            log.debug(
                "full-trace: proposal_scorer llm_call append failed for model=%s",
                model,
                exc_info=True,
            )

    def _trace_scorer_llm_failure(
        self,
        model: str,
        error: BaseException,
        keys: ScorerCallKeys,
        *,
        latency_ms: int | None = None,
    ) -> None:
        """Append one ``status=\"error\"`` row for a scoring call that never returned."""
        if self.session_dir is None:
            return
        try:
            site = LLMCallSite(
                session_id=self.session_dir.name,
                component="proposal_scorer",
                role="proposal_scorer",
                task_id=keys.task_id,
                tick=keys.tick,
                phase=keys.phase,
            )
            record = LLMCallRecord.for_failure(site, error, model=str(model), latency_ms=latency_ms)
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:
            log.debug(
                "full-trace: proposal_scorer llm_call failure append failed for model=%s",
                model,
                exc_info=True,
            )

    def _record_scorer_conversation(
        self,
        model: str,
        prompt: str,
        response: str,
        keys: ScorerCallKeys,
        *,
        call_id: str | None = None,
    ) -> None:
        """Append one ``conversations.jsonl`` row for a proposal-scoring call."""
        if self.session_dir is None:
            return
        if not prompt and not response:
            return
        record = ConversationRecord(
            session_id=self.session_dir.name,
            component="proposal_scorer",
            role="proposal_scorer",
            call_id=call_id,
            task_id=keys.task_id,
            tick=keys.tick,
            phase=keys.phase,
            model=str(model),
            prompt=prompt or "",
            response=response or "",
        )
        append_conversation(session_dir=self.session_dir, record=record)

    async def score(
        self,
        *,
        gap: dict[str, Any],
        proposals: list[dict[str, Any]],
        task_id: str | None = None,
        tick: int | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        """Ask the brain to score ``proposals`` against ``gap`` with every configured model."""
        keys = ScorerCallKeys(task_id=task_id, tick=tick, phase=phase)
        handles = {m: _ScorerModel(self, m, keys) for m in self.models}
        port = DecisionPort("proposal scoring", llm=handles)
        turn = await ask_brain(
            CallSite.SCORE,
            {
                _PROPOSAL_SCORE_KEY: {
                    "gap": gap,
                    "proposals": list(proposals or []),
                    "models": list(self.models),
                    "max_completion_tokens": self.max_completion_tokens,
                }
            },
            port=port,
            tick=int(tick or 0),
            phase=phase or "",
        )
        decision = one_decision(turn, ("score.record",), what="proposal scores")
        if decision is None:
            return {"scale": "0-10", "models": {}, "errors": {}}
        return {
            "scale": str(decision.args.get("scale") or "0-10"),
            "models": dict(decision.args.get("models") or {}),
            "errors": dict(decision.args.get("errors") or {}),
        }


__all__ = ["ProposalScorer", "DEFAULT_SCORER_MODELS"]

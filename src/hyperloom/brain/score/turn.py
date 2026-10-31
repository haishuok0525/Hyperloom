# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Advisory multi-model proposal scoring (CallSite.SCORE)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, NamedTuple

from hyperloom.brain.score.prompt import (
    MAX_PROPOSALS_SCORED,
    SCORING_INSTRUCTIONS,
    build_scoring_prompt,
    normalise_model_scores,
    prepare_scoring_proposals,
)
from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall
from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom.common.timeutil import format_exc_brief

log = logging.getLogger(__name__)

PROPOSAL_SCORE_KEY = "proposal_score"


class _ModelOutcome(NamedTuple):
    model: str
    scores: dict[str, dict[str, Any]] | None
    error: str | None


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


def _record(batch: TickBatch, *, models: Mapping[str, Any], errors: Mapping[str, str]) -> BrainTurn:
    return _turn(batch, ToolCall("score.record", {"scale": "0-10", "models": dict(models), "errors": dict(errors)}))


async def score_proposals(batch: TickBatch, port: Port) -> BrainTurn:
    """Score ``proposals`` against ``gap`` with every model lent on ``port.llm``."""
    request: Mapping[str, Any] = batch.snapshot.data[PROPOSAL_SCORE_KEY]
    proposals = [p for p in (request.get("proposals") or []) if isinstance(p, dict)]
    models = tuple(str(m) for m in (request.get("models") or ()) if str(m).strip())
    if not proposals or not models:
        return _record(batch, models={}, errors={})
    if len(proposals) > MAX_PROPOSALS_SCORED:
        proposals = proposals[:MAX_PROPOSALS_SCORED]
    try:
        scoring_entries = prepare_scoring_proposals(proposals)
    except ValueError as exc:
        return _record(batch, models={}, errors={"input": format_exc_brief(exc, limit=200)})

    prompt = build_scoring_prompt(gap=dict(request.get("gap") or {}), scoring_entries=scoring_entries)
    full_prompt = f"{prompt}\n\n{SCORING_INSTRUCTIONS}"
    max_tokens = int(request.get("max_completion_tokens") or 4096)

    async def _one(model: str) -> _ModelOutcome:
        handle = port.llm.get(model)
        if handle is None:
            return _ModelOutcome(model, None, f"no model handle for {model!r}")
        try:
            reply = await handle.run(full_prompt, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — every model failure is recorded, never raised
            log.warning("proposal score: model=%s failed: %r", model, exc)
            return _ModelOutcome(model, None, format_exc_brief(exc, limit=200))
        parsed = extract_last_json_with_key(reply.raw_text, "scores")
        if parsed is None:
            return _ModelOutcome(model, None, f"no parseable scores JSON (reply_chars={len(reply.raw_text)})")
        normalised = normalise_model_scores(parsed, scoring_entries=scoring_entries)
        if not normalised:
            return _ModelOutcome(model, None, "no usable scores returned")
        return _ModelOutcome(model, normalised, None)

    results = await asyncio.gather(*(_one(m) for m in models))
    models_out: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for outcome in results:
        if outcome.error is not None:
            errors[outcome.model] = outcome.error
            continue
        if outcome.scores:
            models_out[outcome.model] = outcome.scores
    return _record(batch, models=models_out, errors=errors)

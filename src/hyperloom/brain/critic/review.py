# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One Critic turn: read the judge bundle, decide what to ask the model, read its ruling back."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping

from hyperloom.brain.critic.constraints import apply_review_constraints
from hyperloom.brain_api import BrainTurn, LlmCallError, LlmHandle, Port, TickBatch, ToolCall, TurnFailed
from hyperloom.common.jsonio import extract_last_json_matching

log = logging.getLogger(__name__)

#: The role whose model a Critic turn is lent.
CRITIC_ROLE = "critic"

# One retry at this multiple of the cap when a reply stops at the limit.
TRUNCATION_RETRY_FACTOR = 2
# Finish/stop reasons that mean "cut off at the output cap": OpenAI reports ``length``, the Anthropic Messages API
# reports ``max_tokens``.
_TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})

# Output instructions for the exact ``review.json`` shape commit-review validates.
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
      "required_evidence": ["<key>", ...],
      "notes": ["..."],
      "failure_reason_code": "<failure_reason_code of the review_constraints rule this verdict rests on, else \"\">",
      "persist_to_kb": false,
      "topic": "<slug>"
    }
  ],
  "advice": [
    { "target_proposal_msg_id": "<msg_id>", "body_md": "..." }
  ]
}

Rules (mirror SKILL.md Hard Rules + Approve Standard):
- Wrap the JSON in a ```json fenced block. Bare JSON is also accepted.
- Free text outside the JSON is ignored.
- Keep `reasoning`/`notes` to new, decision-relevant points; do not restate the
  proposal or context already in the judge_bundle.
- Emit one verdict object PER proposal in `judge_bundle.proposals`.
- If `judge_bundle.required_context` is non-empty, every verdict MUST be
  `needs_review` with `source = "critic_unavailable"` and list the
  missing keys in `notes`.
- If `judge_bundle.kb_read_skipped_reason == "kb_unreachable"`, prefer
  `advise` / `needs_review` over `approve` and mention the missing KB
  recall in `notes`.
- If there are no proposals, return `{"review_verdicts": []}` — the
  runtime falls back to a heartbeat.
- `approve` requires comparable before/after benchmark, accuracy gate
  (or waiver), active-path proof when relevant, and a clear rollback.
- Omit `alternative_action`.
- When a verdict rests on a rule from `review_constraints`, copy that
  rule's `failure_reason_code` verbatim into the verdict's own
  `failure_reason_code`; leave it `""` when the verdict rests on your
  own judgement. Some of those rules declare `advise` as their
  `failure_verdict`, and naming the rule is how the Coordinator tells
  a verdict resting on one apart from a substantive refusal.
==== END OUTPUT FORMAT ====
""".strip()


# Bare {...} fallback carrying "review_verdicts" (fenced case handled by helper).
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"review_verdicts\"[\s\S]*\})", re.DOTALL)


def extract_review_json(text: str) -> dict[str, Any] | None:
    """Pull the Critic's own ``{\"review_verdicts\": ...}`` object out of a reply."""
    return extract_last_json_matching(text, "review_verdicts", _BARE_JSON_RE)


def is_truncated_finish(finish: str | None) -> bool:
    """Report whether a finish/stop reason means the reply hit the output cap."""
    return isinstance(finish, str) and finish.strip().lower() in _TRUNCATED_FINISH_REASONS


# The verdicts a lesson is worth keeping for, when the Critic asked for one.
_KB_LESSON_VERDICTS = frozenset({"reject", "redirect", "approve"})


def kb_write_targets(review: Mapping[str, Any]) -> list[str]:
    """The proposals whose verdict the KB keeps as a lesson: those the Critic asked to persist, on a verdict worth it."""
    verdicts = review.get("review_verdicts")
    if not isinstance(verdicts, list):
        return []
    return [
        item["target_proposal_msg_id"]
        for item in verdicts
        if isinstance(item, dict)
        and isinstance(item.get("target_proposal_msg_id"), str)
        and item.get("verdict") in _KB_LESSON_VERDICTS
        and item.get("persist_to_kb")
    ]


async def review_turn(batch: TickBatch, port: Port) -> BrainTurn:
    """Judge the proposals the body has gathered, and ask it to record the ruling.

    ``batch.snapshot.data["critic"]`` carries the turn's inputs: ``system_prompt``, ``skill_preamble``,
    ``max_completion_tokens``, ``action_verdict_policy``, and the body's ``rules``.

    Raises:
        TurnFailed: if the model could not be reached or its reply carried no ruling.
    """
    critic = batch.snapshot.data["critic"]
    prepared = await port.call(ToolCall("kb.critic.read", {}))
    judge_bundle = dict(prepared["judge_bundle"])
    apply_review_constraints(
        judge_bundle,
        phase=batch.snapshot.phase,
        action_verdict_policy=critic["action_verdict_policy"],
        rules=critic["rules"],
        proposal_levers=prepared["proposal_levers"],
    )
    # No model call when there is nothing to judge.
    if not (judge_bundle.get("proposals") or []):
        review: dict[str, Any] = {"review_verdicts": []}
        text, finish = "(skipped — no proposals)", "skipped"
    else:
        review, text, finish = await reason(judge_bundle, critic, port.llm[CRITIC_ROLE])
    record = {
        "review": review,
        "raw_text": text,
        "finish": finish,
        "judge_bundle": judge_bundle,
        "kb_writes": kb_write_targets(review),
    }
    return BrainTurn(tool_calls=(ToolCall("verdict.record", record),), brain_state=batch.brain_state)


async def reason(
    judge_bundle: Mapping[str, Any],
    critic: Mapping[str, Any],
    model: LlmHandle,
) -> tuple[dict[str, Any], str, str | None]:
    """Drive the review model with the judge bundle and parse a review object.

    Raises:
        TurnFailed: if the call failed, the retry after a truncated reply was rejected, or no reply parsed.
    """
    preamble = str(critic["skill_preamble"] or "")
    system_prompt = critic["system_prompt"]
    bundle_view: dict[str, Any] = {
        "session_id": judge_bundle.get("session_id"),
        "decision_id": judge_bundle.get("decision_id"),
        "phase": judge_bundle.get("phase"),
        "merged_context": judge_bundle.get("merged_context"),
        "missing_context": judge_bundle.get("missing_context"),
        "required_context": judge_bundle.get("required_context"),
        "proposals": judge_bundle.get("proposals"),
        "kb_priors_by_proposal": judge_bundle.get("kb_priors_by_proposal"),
        "kb_read_skipped_reason": judge_bundle.get("kb_read_skipped_reason"),
        "review_constraints": judge_bundle.get("review_constraints"),
        "notes": judge_bundle.get("notes"),
    }
    bundle_text = json.dumps(bundle_view, ensure_ascii=False, separators=(",", ":"))
    review_prompt = f"==== JUDGE BUNDLE ====\n{bundle_text}\n==== END JUDGE BUNDLE ====\n\n{REVIEW_OUTPUT_INSTRUCTIONS}"
    max_tokens = int(critic["max_completion_tokens"])
    reply = await model.run(review_prompt, system_prompt=system_prompt, prefix=preamble, max_tokens=max_tokens)
    text, finish = reply.raw_text, reply.finish_reason
    review = extract_review_json(text)

    if review is None and is_truncated_finish(finish):
        # Retrying a cap-truncated reply under the same cap would truncate again at the same byte, so the retry
        # only makes sense with more room.
        retry_tokens = max_tokens * TRUNCATION_RETRY_FACTOR
        log.warning(
            "critic_agent_backend: review reply stopped at the %d-token cap "
            "(chars=%d, finish=%s); retrying once with %d",
            max_tokens,
            len(text),
            finish,
            retry_tokens,
        )
        try:
            reply = await model.run(
                review_prompt, system_prompt=system_prompt, prefix=preamble, max_tokens=retry_tokens
            )
        except LlmCallError as exc:
            # A provider whose own output limit sits below the doubled cap rejects the retry outright.
            raise TurnFailed(
                f"CriticAgentBackend: review reply was truncated at {max_tokens} tokens "
                f"and the retry at {retry_tokens} was rejected: {exc}"
            ) from exc
        text, finish = reply.raw_text, reply.finish_reason
        review = extract_review_json(text)
        max_tokens = retry_tokens

    if review is None:
        # A reply carrying no verdicts is a review that failed to arrive, not a review that found nothing to say.
        raise TurnFailed(
            "CriticAgentBackend: review reply carried no parseable review_verdicts JSON "
            f"(chars={len(text)}, finish={finish!r}, max_tokens={max_tokens})"
        )
    return review, text, finish

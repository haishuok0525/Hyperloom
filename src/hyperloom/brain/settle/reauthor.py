# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a needs_review verdict on an authored candidate comes to: a re-author round seeded with the Critic's
feedback, or the candidate's close when there is nothing to act on or its budget is spent."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall
from hyperloom.common.coerce import first_str, to_list

# Snapshot key the body asks under.
REAUTHOR_KEY = "critic_reauthor"
# Re-author rounds one candidate gets.
REAUTHOR_MAX_ATTEMPTS = 3


def _stripped(value: Any) -> list[str]:
    return [str(x).strip() for x in to_list(value) if str(x).strip()]


async def settle_reauthor(batch: TickBatch, port: Port) -> BrainTurn:
    """Re-author once more while the Critic named evidence to supply and the candidate has budget left.

    A candidate already being applied or benchmarked is left alone.
    """
    request: Mapping[str, Any] = batch.snapshot.data[REAUTHOR_KEY]
    advisory = request["advisory"]
    advice = first_str(advisory.get("advice_text"))
    required = _stripped(advisory.get("required_evidence"))
    if not required:
        call = ToolCall("reauthor.decline", {"rationale": advice[:500]})
    elif (await port.call(ToolCall("query.candidate_materializing", {})))["materializing"]:
        return BrainTurn(brain_state=batch.brain_state)
    elif (prior := int(request["prior_attempts"])) >= REAUTHOR_MAX_ATTEMPTS:
        rationale = f"reauthor attempts >= cap ({REAUTHOR_MAX_ATTEMPTS})"
        call = ToolCall("reauthor.exhausted", {"attempts": prior, "rationale": rationale})
    else:
        feedback = {"required_evidence": required, "advice_text": advice, "risks": _stripped(advisory.get("risks"))}
        call = ToolCall("reauthor.dispatch", {"attempt": prior + 1, "critic_feedback": feedback})
    return BrainTurn(tool_calls=(call,), brain_state=batch.brain_state)

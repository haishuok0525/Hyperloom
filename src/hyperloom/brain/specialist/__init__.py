# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist brief: the system and user prompt a specialist runs on."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, TickBatch, ToolCall, TurnFailed

from .brief import build_specialist_prompts
from .inputs import BriefFacts, BriefInputs, DomainCard

#: Where a brief request sits in the snapshot of a brief call.
BRIEF_KEY = "specialist_brief"


def write_brief(request: Mapping[str, Any]) -> tuple[str, str]:
    """The ``(system_prompt, user_prompt)`` of a brief request."""
    return build_specialist_prompts(BriefInputs.from_request(request))


def brief_turn(batch: TickBatch) -> BrainTurn:
    """Write the brief of the specialist the body is about to run, as the ``specialist.run`` it runs on.

    Raises:
        TurnFailed: if the batch carries no brief request.
    """
    request = batch.snapshot.data.get(BRIEF_KEY)
    if not isinstance(request, Mapping):
        raise TurnFailed(f"a brief call carries no {BRIEF_KEY!r}")
    system_prompt, user_prompt = write_brief(request)
    call = ToolCall("specialist.run", {"system_prompt": system_prompt, "user_prompt": user_prompt})
    return BrainTurn(tool_calls=(call,), brain_state=batch.brain_state)


__all__ = [
    "BRIEF_KEY",
    "BriefFacts",
    "BriefInputs",
    "DomainCard",
    "brief_turn",
    "write_brief",
]

# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One orchestration turn: compose the prompt from the body's view, ask the model it, route every intent it emits."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain.orchestrate.prompt import compose_prompt
from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall, TurnFailed

#: The role whose model an orchestration turn is lent.
ORCHESTRATION_ROLE = "orchestration"
#: Snapshot key of the turn the body reports: ``system_prompt``, ``view``, ``request``, ``tools``, ``max_turns``.
ORCHESTRATION_KEY = "orchestration_turn"


def _with_request(prompt: str, turn: Mapping[str, Any]) -> str:
    """The prompt, with the turn's request after it when there is one."""
    request = turn.get("request") or ""
    return f"{prompt}\n\n{request}" if request else prompt


async def orchestration_turn(batch: TickBatch, port: Port) -> BrainTurn:
    """Run the orchestration model once and hand each intent it emitted to the body's router, in order.

    The session context goes in as the prompt's cacheable prefix; the backend decides where a cached prefix sits.

    Raises:
        TurnFailed: if the body reported no turn; a failed model call reaches the body as it was raised.
    """
    turn = batch.snapshot.data.get(ORCHESTRATION_KEY)
    if not isinstance(turn, Mapping):
        raise TurnFailed(f"an orchestration turn needs the {ORCHESTRATION_KEY!r} the body reports")
    prompt = await compose_prompt(turn["view"], port)
    reply = await port.llm[ORCHESTRATION_ROLE].run(
        _with_request(prompt.turn, turn),
        system_prompt=turn.get("system_prompt"),
        prefix=prompt.context,
        tools=turn.get("tools"),
        max_turns=turn.get("max_turns"),
    )
    calls = tuple(ToolCall("intent.route", intent) for intent in reply.intents)
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)

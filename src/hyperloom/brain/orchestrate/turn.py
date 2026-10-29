# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One orchestration turn: ask the model the composed prompt and the turn's request, route every intent it emits."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall, TurnFailed

#: The role whose model an orchestration turn is lent.
ORCHESTRATION_ROLE = "orchestration"
#: Snapshot key of the turn the body composed: ``system_prompt``, ``prompt``, ``request``, ``tools``, ``max_turns``.
ORCHESTRATION_KEY = "orchestration_turn"


def _prompt(turn: Mapping[str, Any]) -> str:
    """The composed prompt, with the turn's request after it when there is one."""
    request = turn.get("request") or ""
    return f"{turn['prompt']}\n\n{request}" if request else turn["prompt"]


async def orchestration_turn(batch: TickBatch, port: Port) -> BrainTurn:
    """Run the orchestration model once and hand each intent it emitted to the body's router, in order.

    Raises:
        TurnFailed: if the body composed no turn; a failed model call reaches the body as it was raised.
    """
    turn = batch.snapshot.data.get(ORCHESTRATION_KEY)
    if not isinstance(turn, Mapping):
        raise TurnFailed(f"an orchestration turn needs the {ORCHESTRATION_KEY!r} the body composed")
    reply = await port.llm[ORCHESTRATION_ROLE].run(
        _prompt(turn),
        system_prompt=turn.get("system_prompt"),
        tools=turn.get("tools"),
        max_turns=turn.get("max_turns"),
    )
    calls = tuple(ToolCall("intent.route", intent) for intent in reply.intents)
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)

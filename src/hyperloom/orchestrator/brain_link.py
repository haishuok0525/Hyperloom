# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the body finds the brain it calls: the ``brain`` plug when a manifest selects one, else Brain v0."""

from __future__ import annotations

import threading
import time
from typing import Any, Awaitable, Callable, Collection, Mapping

from hyperloom.brain_api import (
    CALL_SITE_METHODS,
    Brain,
    BrainContractError,
    BrainTurn,
    CallSite,
    StateSnapshot,
    TickBatch,
    ToolCall,
    TurnFailed,
)
from hyperloom.brain_api.loader import load_brain
from hyperloom.plugs import active

_BRAIN_SOCKET = "brain"
_default: Brain | None = None
_lock = threading.Lock()


def resolve_brain() -> Brain:
    """The brain of the current plug host, or the default brain loaded once by name."""
    host = active()
    if host.has(_BRAIN_SOCKET):
        return host.get(_BRAIN_SOCKET)  # type: ignore[return-value]
    global _default
    with _lock:
        if _default is None:
            _default = load_brain()
        return _default


#: Answers one query a deciding brain asks, from the call's args.
QueryAnswer = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]


class DecisionPort:
    """What a brain call that only decides is lent: no model, and no tool beyond the queries it answers."""

    llm: dict[str, Any] = {}

    def __init__(self, what: str, queries: Mapping[str, QueryAnswer] | None = None) -> None:
        self._what = what
        self._queries = dict(queries or {})

    async def call(self, call: ToolCall) -> dict[str, Any]:
        answer = self._queries.get(call.name)
        if answer is None:
            raise TurnFailed(f"{self._what} cannot call {call.name!r}")
        return await answer(call.args)


async def ask_brain(
    call_site: CallSite, question: Mapping[str, Mapping[str, Any]], *, port: DecisionPort, tick: int, phase: str
) -> BrainTurn:
    """Ask the brain, at ``call_site``, the question ``question`` puts under its one snapshot key."""
    batch = TickBatch(
        call_site=call_site,
        tick=tick,
        now_unix=time.time(),
        events=(),
        snapshot=StateSnapshot(phase=phase, data={key: dict(request) for key, request in question.items()}),
    )
    method = getattr(resolve_brain(), CALL_SITE_METHODS[call_site])
    return await method(batch, port)


def step_answer(run: Callable[[], Awaitable[Any]]) -> QueryAnswer:
    """The answer to a call that runs one body step and reports nothing back."""

    async def _answer(_args: Mapping[str, Any]) -> dict[str, Any]:
        await run()
        return {}

    return _answer


def one_decision(turn: BrainTurn, allowed: Collection[str], *, what: str) -> ToolCall | None:
    """The one decision ``turn`` makes about ``what``, ``None`` when it makes none.

    Raises:
        BrainContractError: if the turn makes more than one, or one outside ``allowed``.
    """
    calls = turn.tool_calls
    if len(calls) > 1 or any(c.name not in allowed for c in calls):
        raise BrainContractError(f"brain decided {what} with {[c.name for c in calls]}, not one of {sorted(allowed)}")
    return calls[0] if calls else None

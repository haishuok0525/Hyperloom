# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where the body finds the brain it calls: the ``brain`` plug when a manifest selects one, else Brain v0."""

from __future__ import annotations

import threading
from typing import Any

from hyperloom.brain_api import Brain, ToolCall, TurnFailed
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


class DecisionPort:
    """What a brain call that only decides is lent: no model and no tool."""

    llm: dict[str, Any] = {}

    def __init__(self, what: str) -> None:
        self._what = what

    async def call(self, call: ToolCall) -> dict[str, Any]:
        raise TurnFailed(f"{self._what} cannot call {call.name!r}")

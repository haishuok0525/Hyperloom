# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reading the judge model's reply: its last JSON object carrying ``"open"`` is the answer.

A reply with no such object answers only when its last object carrying ``"stop_session"`` asks to stop the session:
that answer opens nothing and gives no order. Any other reply, and an object whose fields are malformed, holds none.
"""

from __future__ import annotations

from collections.abc import Mapping

from hyperloom.common.jsonio import extract_last_json_with_key
from hyperloom_multibranch.contracts.brain import JudgeAnswer, KOrder, StopOrder

_KEY_OPEN = "open"
_KEY_OPEN_K = "open_k"
_KEY_FOCUS = "focus"
_KEY_BRANCHES = "branches"
_KEY_STOP = "stop"
_KEY_STOP_SESSION = "stop_session"
_KEY_BRANCH = "branch"
_KEY_REASON = "reason"


class _MalformedReply(ValueError):
    pass


def _int_field(item: Mapping[str, object], key: str) -> int:
    value = item.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _MalformedReply(f"{key} is not an integer: {value!r}")
    return value


def _str_field(item: Mapping[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _MalformedReply(f"{key} is not a non-empty string: {value!r}")
    return value.strip()


def _objects(payload: Mapping[str, object], key: str) -> list[Mapping[str, object]]:
    items = payload.get(key) or []
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise _MalformedReply(f"{key} is not a list of objects")
    return items


def _focus(payload: Mapping[str, object]) -> tuple[str, ...]:
    items = payload.get(_KEY_FOCUS) or []
    if not isinstance(items, list) or not all(item is None or isinstance(item, str) for item in items):
        raise _MalformedReply(f"{_KEY_FOCUS} is not a list of ops")
    return tuple((item or "").strip() for item in items)


def _session_stop(payload: Mapping[str, object]) -> str:
    request = payload.get(_KEY_STOP_SESSION)
    if request is None:
        return ""
    if not isinstance(request, dict):
        raise _MalformedReply(f"{_KEY_STOP_SESSION} is neither null nor an object")
    return _str_field(request, _KEY_REASON)


def _stop_only(reply: str) -> JudgeAnswer | None:
    payload = extract_last_json_with_key(reply, _KEY_STOP_SESSION)
    if payload is None or payload[_KEY_STOP_SESSION] is None:
        return None
    try:
        return JudgeAnswer(session_stop=_session_stop(payload))
    except _MalformedReply:
        return None


def parse_answer(reply: str) -> JudgeAnswer | None:
    """The answer in ``reply``, ``None`` when it holds none (see the module's docstring)."""

    payload = extract_last_json_with_key(reply, _KEY_OPEN)
    if payload is None:
        return _stop_only(reply)
    try:
        return JudgeAnswer(
            open_count=_int_field(payload, _KEY_OPEN),
            open_k=_int_field(payload, _KEY_OPEN_K) if _KEY_OPEN_K in payload else 1,
            focus=_focus(payload),
            k=tuple(
                KOrder(_str_field(item, _KEY_BRANCH), _int_field(item, "k"))
                for item in _objects(payload, _KEY_BRANCHES)
            ),
            stop=tuple(
                StopOrder(_str_field(item, _KEY_BRANCH), _str_field(item, _KEY_REASON))
                for item in _objects(payload, _KEY_STOP)
            ),
            session_stop=_session_stop(payload),
        )
    except _MalformedReply:
        return None


__all__ = ["parse_answer"]

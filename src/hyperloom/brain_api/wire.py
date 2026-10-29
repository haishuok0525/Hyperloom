# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What may cross the brain boundary: plain JSON values, and opaque brain state as base64."""

from __future__ import annotations

import base64
import json
from typing import Any


class WireError(ValueError):
    """A value that cannot cross the brain boundary unchanged."""


def json_value(value: Any, where: str) -> Any:
    """Return ``value`` as it reads back from JSON; refuse anything that would not survive the round trip.

    Tuples become lists, mapping keys must already be strings, and floats must be finite. A live object (a task
    handle, a path, a dataclass) is refused rather than stringified, so nothing leaks through by accident. Key order
    is kept: a brain that renders a mapping into a prompt sees it in the order the body built it.
    """
    try:
        text = json.dumps(value, allow_nan=False, default=_refuse)
    except (TypeError, ValueError) as exc:
        raise WireError(f"{where}: {exc}") from None
    decoded = json.loads(text)
    if not _same_keys(value, decoded):
        raise WireError(f"{where}: mapping keys must be strings")
    return decoded


def _refuse(obj: object) -> Any:
    raise TypeError(f"{type(obj).__name__} is not a JSON value")


def _same_keys(original: Any, decoded: Any) -> bool:
    if isinstance(original, dict):
        if not all(isinstance(k, str) for k in original):
            return False
        return all(_same_keys(original[k], decoded[k]) for k in original)
    if isinstance(original, (list, tuple)):
        return all(_same_keys(a, b) for a, b in zip(original, decoded))
    return True


def encode_state(state: bytes | None) -> str | None:
    """Brain state as it travels: base64 text, or ``None``."""
    return None if state is None else base64.b64encode(state).decode("ascii")


def decode_state(text: str | None) -> bytes | None:
    """The inverse of :func:`encode_state`."""
    if text is None:
        return None
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise WireError(f"brain_state is not base64: {exc}") from None

# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0's specialist brief, driven from the body's inputs.

Every name of the brief's modules is importable from here; a function called with a ``SpecialistPromptInputs``
first gets the inputs the brain would see for it, facts included.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from hyperloom.brain.specialist import brief, context, focus
from hyperloom.brain.specialist.inputs import BriefInputs
from hyperloom.brain_api.wire import json_value
from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, brief_request


def brief_inputs(inp: SpecialistPromptInputs) -> BriefInputs:
    """The inputs Brain v0 writes the brief of ``inp`` from, after the request crossed the wire."""
    return BriefInputs.from_request(json_value(brief_request(inp), "brief request"))


def _on_body_inputs(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], SpecialistPromptInputs):
            args = (brief_inputs(args[0]), *args[1:])
        return fn(*args, **kwargs)

    return call


def __getattr__(name: str) -> Any:
    module = next((m for m in (brief, context, focus) if hasattr(m, name)), None)
    if module is None:
        raise AttributeError(name)
    value = getattr(module, name)
    return _on_body_inputs(value) if callable(value) and not isinstance(value, type) else value

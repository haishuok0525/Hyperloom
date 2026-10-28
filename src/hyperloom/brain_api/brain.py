# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brain the body drives, the model handle it is lent, and the check that a brain honours the contract."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from hyperloom.brain_api.events import BrainTurn, CallSite, TickBatch
from hyperloom.brain_api.wire import json_value

#: ``major.minor``. A brain built for another major is refused; a newer minor only adds.
CONTRACT_VERSION = "1.0"


class BrainContractError(RuntimeError):
    """A brain that does not honour the contract it was loaded against."""


@dataclass(frozen=True)
class LlmReply:
    """One model turn: the intents it emitted, its text, and backend metadata."""

    intents: tuple[Mapping[str, Any], ...] = ()
    raw_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "intents", tuple(json_value(list(self.intents), "llm intents")))
        object.__setattr__(self, "metadata", json_value(dict(self.metadata), "llm metadata"))


@runtime_checkable
class LlmHandle(Protocol):
    """A model the body lends the brain for one call; the body counts its cost and holds its credentials."""

    role: str

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: Sequence[str] | None = None,
        max_turns: int | None = None,
    ) -> LlmReply:
        """Run one turn."""
        ...


@runtime_checkable
class Brain(Protocol):
    """Decides what the body does next. Called at three points of every tick, always in :class:`CallSite` order."""

    name: str
    contract_version: str

    async def on_llm_turn(self, batch: TickBatch, llm: Mapping[str, LlmHandle]) -> BrainTurn:
        """The orchestration and critic turns; ``llm`` holds one handle per role this tick may run."""
        ...

    async def on_settlement(self, batch: TickBatch, llm: Mapping[str, LlmHandle]) -> BrainTurn:
        """One settled task, the only event in ``batch``."""
        ...

    async def on_phase_pump(self, batch: TickBatch, llm: Mapping[str, LlmHandle]) -> BrainTurn:
        """The current phase's strategy."""
        ...


#: Call site -> the brain method the body calls there.
CALL_SITE_METHODS: Mapping[CallSite, str] = {
    CallSite.LLM_TURN: "on_llm_turn",
    CallSite.SETTLEMENT: "on_settlement",
    CallSite.PHASE_PUMP: "on_phase_pump",
}


def _major(version: object) -> str | None:
    if not isinstance(version, str):
        return None
    major, dot, minor = version.partition(".")
    return major if dot and major.isdigit() and minor.isdigit() else None


def validate_brain(brain: object) -> Brain:
    """Return ``brain`` if it honours this contract.

    Raises:
        BrainContractError: if it names no brain, was built for another contract major, or lacks a call site.
    """
    name = getattr(brain, "name", None)
    if not (isinstance(name, str) and name):
        raise BrainContractError(f"{type(brain).__name__} has no name")
    version = getattr(brain, "contract_version", None)
    if _major(version) is None:
        raise BrainContractError(f"brain {name!r} contract_version {version!r} is not major.minor")
    if _major(version) != _major(CONTRACT_VERSION):
        raise BrainContractError(f"brain {name!r} targets contract {version}, the body speaks {CONTRACT_VERSION}")
    missing = [m for m in CALL_SITE_METHODS.values() if not inspect.iscoroutinefunction(getattr(brain, m, None))]
    if missing:
        raise BrainContractError(f"brain {name!r} has no async {', '.join(missing)}")
    return brain  # type: ignore[return-value]

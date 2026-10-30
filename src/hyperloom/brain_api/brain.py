# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The brain the body drives, what the body lends it for one call, and the check that a brain honours the contract."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from hyperloom.brain_api.events import BrainTurn, CallSite, TickBatch
from hyperloom.brain_api.tools import INTENT_TOOLS, ToolCall
from hyperloom.brain_api.wire import json_value

#: ``major.minor``. A brain built for another major is refused; a newer minor only adds.
CONTRACT_VERSION = "1.5"


class BrainContractError(RuntimeError):
    """A brain that does not honour the contract it was loaded against."""


class TurnFailed(RuntimeError):
    """A brain call that could not decide; the body records the turn as failed with this message."""


class LlmCallError(TurnFailed):
    """A model call the body made for the brain failed. A brain that lets it through fails the turn as the body would."""


@dataclass(frozen=True)
class LlmReply:
    """One model turn: the intents it emitted, its text, why it stopped, and backend metadata."""

    intents: tuple[Mapping[str, Any], ...] = ()
    raw_text: str = ""
    #: The transport's finish or stop reason, as it reported it; ``None`` when it reported none.
    finish_reason: str | None = None
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
        prefix: str | None = None,
        max_tokens: int | None = None,
        tools: Sequence[str] | None = None,
        max_turns: int | None = None,
    ) -> LlmReply:
        """Run one turn. ``prefix`` is a stable leading part of the prompt the backend may cache.

        Raises:
            LlmCallError: if the call failed.
        """
        ...


@runtime_checkable
class Port(Protocol):
    """What the body lends the brain for one call: its models, and the tools that answer inline."""

    llm: Mapping[str, LlmHandle]

    async def call(self, call: ToolCall) -> Mapping[str, Any]:
        """Make a query or control call now and return its answer; a job is asked for in the returned turn instead.

        Raises:
            TurnFailed: if the body refuses or fails the call.
        """
        ...


@runtime_checkable
class Brain(Protocol):
    """Decides what the body does next. Called at three points of every tick, in :class:`CallSite` order, and to brief
    each specialist the body runs."""

    name: str
    contract_version: str
    #: Model role -> the intent types its model may emit, in the order it is offered them.
    role_intents: Mapping[str, Sequence[str]]
    #: The Critic verdicts on a patch's review subject that let ``integrate_patch`` apply it.
    patch_approval_verdicts: Sequence[str]

    async def on_llm_turn(self, batch: TickBatch, port: Port) -> BrainTurn:
        """The model turns; ``port.llm`` holds one handle per role that runs now."""
        ...

    async def on_settlement(self, batch: TickBatch, port: Port) -> BrainTurn:
        """One follow-up of a settlement, named by its ``snapshot.data`` key: a Critic verdict on a pending proposal
        (``"review_verdict_settlement"``), a settled specialist's patches or config lever (``"specialist_patch_delivery"``,
        ``"specialist_config_delivery"``), or a needs_review verdict's re-author (``"critic_reauthor"``)."""
        ...

    async def on_phase_pump(self, batch: TickBatch, port: Port) -> BrainTurn:
        """The current phase's strategy, named by its ``snapshot.data`` key: the FRAMEWORK pump's steps
        (``"framework_pump"``) and candidate track (``"framework_agent_pump"``, ``"candidate_discovery"``), the
        untested-proposal bench (``"auto_bench"``), the predictor (``"predictor_ask"``, ``"predictor_entry"``), or
        whether a Coordinator-owned specialist is dispatched (``"research_scout"``, ``"static_recon"``,
        ``"trajectory_review"``)."""
        ...

    async def on_brief(self, batch: TickBatch, port: Port) -> BrainTurn:
        """The brief of the specialist about to run, returned as the one ``specialist.run`` call it runs on."""
        ...


#: Call site -> the brain method the body calls there.
CALL_SITE_METHODS: Mapping[CallSite, str] = {
    CallSite.LLM_TURN: "on_llm_turn",
    CallSite.SETTLEMENT: "on_settlement",
    CallSite.PHASE_PUMP: "on_phase_pump",
    CallSite.BRIEF: "on_brief",
}


def _major(version: object) -> str | None:
    if not isinstance(version, str):
        return None
    major, dot, minor = version.partition(".")
    return major if dot and major.isdigit() and minor.isdigit() else None


def _role_intents_fault(role_intents: object) -> str | None:
    if not isinstance(role_intents, Mapping):
        return "has no role_intents mapping"
    for role, intents in role_intents.items():
        if not isinstance(role, str) or isinstance(intents, str) or not isinstance(intents, Sequence):
            return f"role_intents[{role!r}] is not a sequence of intent types"
        unknown = [i for i in intents if i not in INTENT_TOOLS]
        if unknown:
            return f"role_intents[{role!r}] names unknown intent types {unknown!r}"
    return None


def _approval_verdicts_fault(verdicts: object) -> str | None:
    if isinstance(verdicts, str) or not isinstance(verdicts, Sequence):
        return "has no patch_approval_verdicts sequence"
    if not all(isinstance(v, str) and v for v in verdicts):
        return f"patch_approval_verdicts {list(verdicts)!r} are not all verdict names"
    return None


def validate_brain(brain: object) -> Brain:
    """Return ``brain`` if it honours this contract.

    Raises:
        BrainContractError: if it names no brain, was built for another contract major, lacks a call site, or does not
            say which intents each model role may emit or which Critic verdicts approve a patch.
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
    fault = _role_intents_fault(getattr(brain, "role_intents", None)) or _approval_verdicts_fault(
        getattr(brain, "patch_approval_verdicts", None)
    )
    if fault:
        raise BrainContractError(f"brain {name!r} {fault}")
    return brain  # type: ignore[return-value]

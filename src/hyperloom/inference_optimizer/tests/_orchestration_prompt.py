# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prompt an agent's turn reads, as the coordinator builds it: the orchestration prompt is composed by the brain
from the body's view, which crosses the contract as JSON; every other role's by the body."""

from __future__ import annotations

from typing import Any

from hyperloom.brain.orchestrate import advisories
from hyperloom.brain.orchestrate import compose_prompt as _brain_compose
from hyperloom.brain_api.wire import json_value
from hyperloom.orchestrator.loop import orchestration_view as _view
from hyperloom.orchestrator.phases import machine_state as _phase_state


class _Port:
    llm: dict = {}

    def __init__(self, conversation) -> None:
        self._conversation = conversation

    async def call(self, call):
        assert call.name == "inbox.rendered", call.name
        self._conversation.mark_rendered("orchestration", int(call.args["seq"]))
        return {"accepted": True}


async def compose_prompt(conversation, agent_name: str) -> Any:
    """The prompt ``agent_name`` reads this tick: ``.context``, ``.turn`` and ``.text``."""
    if agent_name != "orchestration":
        return await conversation.compose_prompt(agent_name)
    view = json_value(await _view.orchestration_view(conversation), "orchestration view")
    return await _brain_compose(view, _Port(conversation))


def _framework(conversation) -> bool:
    return (conversation.shared_state.phase or "").strip().upper() == _phase_state.PHASE_FRAMEWORK_AGENT


def bottleneck_redirect(conversation) -> str:
    """The bottleneck-redirect advisory the orchestration prompt carries now."""
    facts = _view._bottleneck(conversation) if _framework(conversation) else None
    return advisories.bottleneck_redirect(json_value(facts, "bottleneck"))


def acceptance_threshold(conversation) -> str:
    """The acceptance-threshold advisory the orchestration prompt carries now."""
    return advisories.acceptance_threshold(json_value(_view._acceptance(conversation.shared_state), "acceptance"))


def specialist_findings(conversation) -> str:
    """The specialist-findings block the orchestration prompt carries now."""
    rounds = _view._specialist_rounds(conversation.shared_state)
    return advisories.specialist_findings(json_value(rounds, "specialist rounds"))

# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The contract's names stay in step with the body that serves them."""

from __future__ import annotations

import pytest

from hyperloom.brain_api import INTENT_TOOLS, QUERY_NAMES, TOOLS, TOPICS, Caller, tool_for_action
from hyperloom.inference_optimizer.protocol.action_surfaces import (
    ACTION_CATALOGUE,
    COORDINATOR_INTERNAL_ACTIONS,
    FULL_ENABLED_ACTIONS,
    KERNEL_AGENT_OWNED_ACTIONS,
)
from hyperloom.inference_optimizer.protocol.intent import IntentType
from hyperloom.orchestrator.bus.message_bus import TOPIC_ALLOWLIST
from hyperloom.orchestrator.mcp_tool_names import CONTEXT_TOOL_NAMES

_LLM_PROPOSABLE = frozenset(FULL_ENABLED_ACTIONS) - COORDINATOR_INTERNAL_ACTIONS


def test_topics_are_the_bus_allowlist() -> None:
    assert TOPICS == TOPIC_ALLOWLIST


def test_every_intent_type_names_its_tool() -> None:
    assert set(INTENT_TOOLS) == {t.value for t in IntentType}
    assert all(name in TOOLS for name in INTENT_TOOLS.values() if name is not None)


def test_queries_are_the_context_tools_in_mount_order() -> None:
    assert QUERY_NAMES == CONTEXT_TOOL_NAMES


@pytest.mark.parametrize("action", sorted(set(ACTION_CATALOGUE) | COORDINATOR_INTERNAL_ACTIONS))
def test_every_action_is_served_by_a_tool(action) -> None:
    assert tool_for_action(action) in TOOLS


@pytest.mark.parametrize("action", sorted(KERNEL_AGENT_OWNED_ACTIONS))
def test_kernel_owned_actions_are_kernel_tools(action) -> None:
    assert tool_for_action(action).startswith("kernel.")


def test_no_action_tool_without_an_action_behind_it() -> None:
    actions = {n.removeprefix("action.") for n in TOOLS if n.startswith("action.")}
    assert actions <= set(ACTION_CATALOGUE) | COORDINATOR_INTERNAL_ACTIONS


@pytest.mark.parametrize("action", sorted(set(ACTION_CATALOGUE) | COORDINATOR_INTERNAL_ACTIONS))
def test_the_llm_may_call_exactly_the_actions_it_may_propose_today(action) -> None:
    assert (Caller.LLM in TOOLS[tool_for_action(action)].callers) == (action in _LLM_PROPOSABLE)


def test_only_the_combined_measurement_actions_are_transitional() -> None:
    assert {n for n, s in TOOLS.items() if s.transitional} == {"action.explore", "action.integrate_patch"}


def test_an_unknown_action_has_no_tool() -> None:
    with pytest.raises(KeyError):
        tool_for_action("rm_rf")

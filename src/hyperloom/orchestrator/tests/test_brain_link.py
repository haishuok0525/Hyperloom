# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A deciding brain call is lent only the queries the body answers, and is held to one decision."""

from __future__ import annotations

import pytest

from hyperloom.brain_api import BrainContractError, BrainTurn, ToolCall, TurnFailed
from hyperloom.orchestrator.brain_link import DecisionPort, one_decision


async def test_a_decision_port_answers_its_queries_and_refuses_every_other_call() -> None:
    async def _materializing(args):
        return {"materializing": True, "asked": dict(args)}

    port = DecisionPort("a re-author", {"query.candidate_materializing": _materializing})
    answer = await port.call(ToolCall("query.candidate_materializing", {"x": 1}))
    assert answer == {"materializing": True, "asked": {"x": 1}}
    with pytest.raises(TurnFailed, match="a re-author cannot call 'query.config_levers'"):
        await port.call(ToolCall("query.config_levers", {}))


def test_a_turn_is_held_to_at_most_one_allowed_decision() -> None:
    submit = ToolCall("review.submit", {})
    assert one_decision(BrainTurn(), ("review.submit",), what="a patch") is None
    assert one_decision(BrainTurn(tool_calls=(submit,)), ("review.submit",), what="a patch") is submit
    with pytest.raises(BrainContractError, match="decided a patch"):
        one_decision(BrainTurn(tool_calls=(submit, submit)), ("review.submit",), what="a patch")
    with pytest.raises(BrainContractError, match="decided a patch"):
        one_decision(BrainTurn(tool_calls=(ToolCall("review.skip", {}),)), ("review.submit",), what="a patch")

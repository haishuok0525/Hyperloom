# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0 honours the contract and, at a call site that has not moved, decides nothing and keeps its state."""

from __future__ import annotations

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain_api import (
    CALL_SITE_METHODS,
    BrainTurn,
    CallSite,
    StateSnapshot,
    TickBatch,
    TurnFailed,
    validate_brain,
)


class _NoModels:
    llm: dict = {}

    async def call(self, call):
        raise AssertionError(f"unexpected tool call {call.name}")


def test_v0_honours_the_contract() -> None:
    assert validate_brain(create_brain({})).name == "v0"


def test_only_orchestration_drives_and_only_the_critic_judges() -> None:
    roles = create_brain({}).role_intents
    assert set(roles) == {"orchestration", "critic"}
    assert set(roles["critic"]) == {"send_message", "alert", "review_verdict"}
    assert "review_verdict" not in roles["orchestration"]
    assert {"request", "delegate", "propose_action"} <= set(roles["orchestration"])


@pytest.mark.parametrize("site", [CallSite.LLM_TURN, CallSite.SETTLEMENT, CallSite.PHASE_PUMP, CallSite.SCORE])
async def test_without_a_critic_model_every_call_site_returns_no_calls_and_hands_the_state_back(site) -> None:
    brain = create_brain({})
    batch = TickBatch(
        call_site=site, tick=1, now_unix=0.0, events=(), snapshot=StateSnapshot("PRELUDE"), brain_state=b"kept"
    )
    turn = await getattr(brain, CALL_SITE_METHODS[site])(batch, _NoModels())
    assert turn == BrainTurn(brain_state=b"kept")


async def test_a_brief_call_without_a_request_fails_the_turn() -> None:
    batch = TickBatch(call_site=CallSite.BRIEF, tick=0, now_unix=0.0, events=(), snapshot=StateSnapshot(""))
    with pytest.raises(TurnFailed, match="specialist_brief"):
        await create_brain({}).on_brief(batch, _NoModels())

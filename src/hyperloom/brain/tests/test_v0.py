# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0 honours the contract and, at a call site that has not moved, decides nothing and keeps its state."""

from __future__ import annotations

import pytest

from hyperloom.brain import create_brain
from hyperloom.brain_api import CALL_SITE_METHODS, BrainTurn, CallSite, StateSnapshot, TickBatch, validate_brain


class _NoModels:
    llm: dict = {}

    async def call(self, call):
        raise AssertionError(f"unexpected tool call {call.name}")


def test_v0_honours_the_contract() -> None:
    assert validate_brain(create_brain({})).name == "v0"


@pytest.mark.parametrize("site", list(CallSite))
async def test_without_a_critic_model_every_call_site_returns_no_calls_and_hands_the_state_back(site) -> None:
    brain = create_brain({})
    batch = TickBatch(
        call_site=site, tick=1, now_unix=0.0, events=(), snapshot=StateSnapshot("PRELUDE"), brain_state=b"kept"
    )
    turn = await getattr(brain, CALL_SITE_METHODS[site])(batch, _NoModels())
    assert turn == BrainTurn(brain_state=b"kept")

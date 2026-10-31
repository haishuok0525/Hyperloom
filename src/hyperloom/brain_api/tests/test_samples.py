# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recorded call-site samples survive a JSON round trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.brain_api import BrainTurn, TickBatch

_SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.mark.parametrize("path", sorted(_SAMPLES.glob("*.json")), ids=lambda p: p.stem)
def test_sample_observation_and_answer_round_trip(path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    batch = TickBatch.from_wire(raw["observation"])
    turn = BrainTurn.from_wire(raw["answer"])
    assert batch.to_wire() == raw["observation"]
    assert turn.to_wire() == raw["answer"]

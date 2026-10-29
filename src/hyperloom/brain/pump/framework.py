# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The FRAMEWORK phase's per-tick work and its order."""

from __future__ import annotations

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

# Snapshot key the body asks under.
FRAMEWORK_PUMP_KEY = "framework_pump"

# Custom-only mode runs the operator's specialist alone.
_CUSTOM_ONLY_STEPS = ("pump.custom_specialist", "pump.bench_untested", "pump.finish_custom_cycle")
# The predictor runs last, so its fault is filed after the phase's own work has run.
_STEPS = (
    "pump.framework_agent",
    "pump.bench_untested",
    "pump.research_scout",
    "pump.config_plateau_recon",
    "pump.stalled_domain",
    "pump.custom_specialist",
    "pump.advisory_plateau",
    "pump.predictor",
)


async def framework_pump(batch: TickBatch, port: Port) -> BrainTurn:
    """Run the FRAMEWORK pump's steps in order; each step sees what the ones before it enqueued."""
    only = batch.snapshot.data[FRAMEWORK_PUMP_KEY]["custom_specialist_only"]
    for step in _CUSTOM_ONLY_STEPS if only else _STEPS:
        await port.call(ToolCall(step, {}))
    return BrainTurn(brain_state=batch.brain_state)

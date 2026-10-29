# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a settled specialist's deliverable comes to: whether the Critic reviews it for integrate_patch, as a source
patch or as a config lever, or whether it goes nowhere."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

# Snapshot keys the body asks under.
PATCH_DELIVERY_KEY = "specialist_patch_delivery"
CONFIG_DELIVERY_KEY = "specialist_config_delivery"


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


def settle_patch_delivery(batch: TickBatch) -> BrainTurn:
    """Review the patches or tuned artifacts a specialist left on disk; note a claim none of whose files exists."""
    delivery: Mapping[str, Any] = batch.snapshot.data[PATCH_DELIVERY_KEY]
    if delivery["patches_on_disk"] or delivery["artifacts_on_disk"]:
        return _turn(batch, ToolCall("review.submit", {}))
    if delivery["claimed"]:
        return _turn(batch, ToolCall("review.skip", {"claimed": list(delivery["claimed"])}))
    return _turn(batch)


async def settle_config_delivery(batch: TickBatch, port: Port) -> BrainTurn:
    """Review the config lever an authoring specialist delivered instead of a patch.

    An ENABLEMENT round is reviewed even without one -- a round that never reaches integrate_patch never reports what
    its boot did -- unless all it asks for is a targeted build. A lever the KB already rejected is not reviewed again.
    """
    delivery: Mapping[str, Any] = batch.snapshot.data[CONFIG_DELIVERY_KEY]
    if not delivery["authoring"] or delivery["patches_written"]:
        return _turn(batch)
    levers = (await port.call(ToolCall("query.config_levers", {})))["levers"]
    enablement = delivery["enablement"]
    if enablement and delivery["build_requested"] and not levers and not delivery["setup_or_artifacts"]:
        return _turn(batch)
    if not (levers or enablement) or not delivery["specialist_task_id"]:
        return _turn(batch)
    if levers and (await port.call(ToolCall("query.config_lever_known_bad", {"levers": levers})))["known_bad"]:
        return _turn(batch)
    return _turn(batch, ToolCall("review.submit", {}))

# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Forced stalled-domain dispatch, the operator-defined custom specialist, and free-form wave splits."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.brain_api import BrainTurn, Port, TickBatch, ToolCall

STALLED_DOMAIN_KEY = "stalled_domain"
CUSTOM_SPECIALIST_KEY = "custom_specialist"
WAVE_SPLIT_KEY = "specialist_wave"

_FRAMEWORK_PHASE = "FRAMEWORK_AGENT"


def _turn(batch: TickBatch, *calls: ToolCall) -> BrainTurn:
    return BrainTurn(tool_calls=calls, brain_state=batch.brain_state)


async def stalled_domain(batch: TickBatch, port: Port) -> BrainTurn:
    """Force at most one domain specialist for a stalled anchor that still has an open gap."""
    request: Mapping[str, Any] = batch.snapshot.data[STALLED_DOMAIN_KEY]
    if str(request["phase"] or "").strip().upper() != _FRAMEWORK_PHASE:
        return _turn(batch)
    for candidate in request["candidates"]:
        if candidate["busy"] or not candidate["gap_canonical_id"] or not candidate["domain"]:
            continue
        exists = (await port.call(ToolCall("query.task_idempotency", {"key": candidate["idempotency_key"]})))["exists"]
        if exists:
            continue
        refuse = (
            await port.call(
                ToolCall(
                    "query.patch_preflight",
                    {
                        "domain": candidate["domain"],
                        "tags": candidate["tags"],
                        "gap_canonical_id": candidate["gap_canonical_id"],
                        "idempotency_key": candidate["idempotency_key"],
                    },
                )
            )
        )["refuse"]
        if refuse:
            continue
        return _turn(
            batch,
            ToolCall(
                "specialist.force_stalled",
                {
                    "domain": candidate["domain"],
                    "tags": candidate["tags"],
                    "gap_canonical_id": candidate["gap_canonical_id"],
                    "idempotency_key": candidate["idempotency_key"],
                    "reason": candidate["reason"],
                },
            ),
        )
    return _turn(batch)


async def custom_specialist(batch: TickBatch, port: Port) -> BrainTurn:
    """Dispatch the operator-defined specialist once per macro-cycle in FRAMEWORK_AGENT."""
    request: Mapping[str, Any] = batch.snapshot.data[CUSTOM_SPECIALIST_KEY]
    if not request["prompt"] or str(request["phase"] or "").strip().upper() != _FRAMEWORK_PHASE:
        return _turn(batch)
    if request["key"] == request["refused_key"]:
        return _turn(batch)
    if (await port.call(ToolCall("query.task_idempotency", {"key": request["key"]})))["exists"]:
        return _turn(batch)
    gap = await port.call(ToolCall("query.custom_specialist_gap", {}))
    args = {
        "key": request["key"],
        "gap_canonical_id": gap["gap_canonical_id"],
        "no_lease_extension": bool(request["custom_only"]),
        "mode_research": bool(gap.get("research_reason")),
        "research_reason": gap.get("research_reason") or "",
        "gap_symptom": gap.get("gap_symptom") or "",
        "gap_layer": gap.get("gap_layer") or "",
    }
    return _turn(batch, ToolCall("specialist.ensure_custom", args))


async def specialist_wave(batch: TickBatch, _port: Port) -> BrainTurn:
    """Expand a free-form specialist wave into one research-mode sub-task per entry (D-03)."""
    request: Mapping[str, Any] = batch.snapshot.data[WAVE_SPLIT_KEY]
    tasks = []
    for idx, task in enumerate(request["tasks"]):
        entry: dict[str, Any] = {"index": idx, "raw": task, "mode": "research"}
        if isinstance(task, dict):
            summary = str(task.get("task_summary") or "").strip()
            if summary:
                entry["task_summary"] = summary
            for carry in ("mode", "bench", "model", "priority", "timeout_minutes", "max_turns"):
                if carry in task:
                    entry[carry] = task[carry]
            entry.setdefault("mode", "research")
        tasks.append(entry)
    if not tasks:
        return _turn(batch)
    return _turn(
        batch,
        ToolCall(
            "specialist.wave_split",
            {"base_key": request["base_key"], "shared": request["shared"], "tasks": tasks},
        ),
    )

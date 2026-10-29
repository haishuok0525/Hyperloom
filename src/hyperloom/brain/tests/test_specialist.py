# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist brief is written from the request alone: what the body knows arrives as facts."""

from __future__ import annotations

from hyperloom.brain import create_brain
from hyperloom.brain.specialist import BRIEF_KEY, write_brief
from hyperloom.brain_api import CallSite, StateSnapshot, TickBatch


_SCHEMAS = {
    "explore_variant": {"example": {"name": "<variant>"}, "contract": ["- explore variant contract"]},
    "upstream_candidate": {"example": {"name": "<candidate>"}, "contract": ["- upstream candidate contract"]},
}


def _request(**facts) -> dict:
    return {
        "inputs": {
            "task_id": "t1",
            "domain": {"key": "serving_specialist", "layer": "serving", "kb_anchor": "kb/serving"},
            "max_turns": 7,
            "framework_source_roots": ["/src/sglang"],
        },
        "facts": {"proposal_schemas": _SCHEMAS, **facts},
    }


class _NoTools:
    llm: dict = {}

    async def call(self, call):
        raise AssertionError(f"unexpected tool call {call.name}")


async def test_the_brief_comes_back_as_the_specialist_run_it_runs_on() -> None:
    request = _request(pr_query_repos=["org/repo"])
    batch = TickBatch(
        call_site=CallSite.BRIEF,
        tick=0,
        now_unix=0.0,
        events=(),
        snapshot=StateSnapshot("", {BRIEF_KEY: request}),
        brain_state=b"kept",
    )
    turn = await create_brain({}).on_brief(batch, _NoTools())
    (call,) = turn.tool_calls
    assert call.name == "specialist.run"
    assert (call.args["system_prompt"], call.args["user_prompt"]) == write_brief(request)
    assert turn.brain_state == b"kept"


def test_the_brief_states_the_facts_the_body_hands_over() -> None:
    system, user = write_brief(
        _request(
            pr_monitor_tools=["mcp__pr_monitor__list_prs", "other__tool"],
            pr_query_repos=["org/repo"],
            pd_topology={"mode": "disaggregated", "prefill_nodes": 2, "transfer_backend": "mooncake"},
            git_checkouts=["/src/sglang"],
        )
    )
    assert "**serving_specialist**" in system
    assert "Hard cap: at most **7** LLM turns." in system
    assert "``list_prs``" in user and "other__tool" not in user
    assert "- org/repo" in user
    assert "## 1a. PD-DISAGGREGATION" in user and "`mooncake`" in user
    assert "- /src/sglang — git checkout" in user
    assert "- explore variant contract" in system


def test_without_facts_the_host_dependent_sections_fall_back() -> None:
    _, user = write_brief(_request())
    assert "PD-DISAGGREGATION" not in user
    assert "- /src/sglang — installed package, no git tree" in user

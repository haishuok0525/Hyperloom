# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist: its brief, the dispatch an ask comes to, and the candidates its reply and its edits come to."""

from __future__ import annotations

import json

from hyperloom.brain.specialist import (
    BriefInputs,
    brief_inputs,
    build_specialist_prompts,
    candidates_of,
    gap_domain,
    read_done,
    target_gap,
)
from hyperloom.brain_api import (
    Attempt,
    Candidate,
    ConfigDelta,
    Evidence,
    Gap,
    Outcome,
    PatchFile,
    ProposeAsk,
    SearchState,
    Workload,
)

_SCHEMAS = {
    "explore_variant": {"example": {"name": "<variant>"}, "contract": ["- explore variant contract"]},
    "upstream_candidate": {"example": {"name": "<candidate>"}, "contract": ["- upstream candidate contract"]},
}
_DIFF = "--- a/vllm/x.py\n+++ b/vllm/x.py\n@@ -1 +1 @@\n-a\n+b\n"


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


def _brief(request: dict) -> tuple[str, str]:
    return build_specialist_prompts(BriefInputs.from_request(request))


def test_a_workspace_brief_states_the_facts_the_body_hands_over() -> None:
    system, user = _brief(
        _request(
            pr_monitor_tools=["mcp__pr_monitor__list_prs", "other__tool"],
            pr_query_repos=["org/repo"],
            pd_topology={"mode": "disaggregated", "prefill_nodes": 2, "transfer_backend": "mooncake"},
            git_checkouts=["/src/sglang"],
        )
    )
    assert "**serving_specialist**" in system
    assert "Hard cap: at most **7** LLM turns." in system
    assert "specialist_done.json" in system and "patches_written" in system
    assert "``list_prs``" in user and "other__tool" not in user
    assert "## 1a. PD-DISAGGREGATION" in user and "`mooncake`" in user
    assert "- /src/sglang — git checkout" in user
    assert "- explore variant contract" in system


def _state(**kw) -> SearchState:
    gaps = (
        Gap("gemm.linear", domain="gemm", severity="medium", layer="kernel", symptom="gemm at 40%"),
        Gap("host.step_overhead", domain="host", severity="high", layer="host", symptom="host is 30%"),
        Gap("attention.decode", domain="attention", severity="high", layer="kernel", symptom="attn at 20%"),
    )
    values = dict(
        workload=Workload("Qwen/Qwen3-0.6B", "vllm", framework_version="0.33.0", gpu_type="MI300X", tp=1, conc=64),
        baseline_tput=1000.0,
        current_tput=1100.0,
        keep_threshold_pct=1.0,
        gaps=gaps,
        evidence=Evidence(roofline={"bound": "memory"}, runtime_findings="kv cache 30% used"),
    )
    values.update(kw)
    return SearchState(**values)


def test_the_dispatch_is_anchored_at_the_asks_gap_else_the_first_most_severe() -> None:
    state = _state()
    assert target_gap(state, ProposeAsk(task="t")).canonical_id == "host.step_overhead"
    own = Gap("x", layer="framework")
    assert target_gap(state, ProposeAsk(task="t", gap=own)) is own
    assert target_gap(_state(gaps=()), ProposeAsk(task="t")) is None


def test_a_gap_points_at_its_domain_by_key_then_anchor_then_layer() -> None:
    assert gap_domain(Gap("a", domain="comm_specialist")) == "comm_specialist"
    assert gap_domain(Gap("a", domain="compiler")) == "compiler_specialist"
    assert gap_domain(Gap("a", domain="attention", layer="kernel")) == "kernel_switch_specialist"
    assert gap_domain(Gap("a", layer="Host")) == "system_specialist"
    assert gap_domain(Gap("a", domain="attention")) == ""
    assert gap_domain(None) == ""


def test_a_freeform_ask_is_briefed_on_its_task_with_the_focus_of_the_gap_it_is_anchored_at() -> None:
    stack = (Candidate("chunked", ConfigDelta("--max-num-batched-tokens 4096")),)
    attempts = (
        Attempt(stack[0], Outcome.KEEP, gain_pct=10.0),
        Attempt(Candidate("graphs", ConfigDelta("--enforce-eager")), Outcome.REVERT, gain_pct=-3.0, note="slower"),
    )
    inp = brief_inputs(_state(stack=stack, attempts=attempts), ProposeAsk(task="Optimise Qwen3", patches=True))
    assert (inp.domain.key, inp.scope, inp.mode) == ("freeform_specialist", "freeform", "patch")
    assert inp.facts.focus_tag_domains == ("system_specialist",)
    assert inp.gap_canonical_id == "host.step_overhead"
    system, user = build_specialist_prompts(inp)
    assert "> Optimise Qwen3" in system
    assert "### Domain focus — system_specialist" in system
    assert "Exit — final reply" in system
    assert "specialist_done.json" not in system and "patches_written" not in system
    assert "heartbeat" not in system and "inbox.json" not in system
    assert "``patch_reason``" in system
    assert "`host.step_overhead`" in user
    assert "Your working directory is the installed vllm 0.33.0 source" in user
    assert "(unavailable: pr_monitor disabled)" in user
    assert "graphs (--enforce-eager) [revert] gain=-3.00% — slower" in user
    assert "chunked (+10.00%)" in user
    assert "- validated cumulative gain: +10.00%" in user


def test_a_domain_ask_runs_that_domain_and_one_without_patches_is_read_only() -> None:
    inp = brief_inputs(_state(), ProposeAsk(domain="serving_specialist", max_turns=9))
    assert (inp.domain.key, inp.scope, inp.mode, inp.max_turns) == ("serving_specialist", "domain", "research", 9)
    system, _ = build_specialist_prompts(inp)
    assert "Read-only dispatch" in system and "``patch_reason``" not in system
    assert brief_inputs(_state(), ProposeAsk(domain="research_scout_specialist")) is None


def _reply(payload: dict) -> str:
    return "Done.\n```json\n" + json.dumps({"intent_type": "specialist_done", "payload": payload}) + "\n```\n"


def test_the_reply_comes_to_the_edit_first_then_the_valid_variants_in_order() -> None:
    done = read_done(
        _reply(
            {
                "proposal_set": [
                    {"name": "graphs", "extra_args": "--cuda-graph-sizes 64", "kb_evidence": ["kb1"], "reason": "r1"},
                    {"name": "graphs", "extra_envs": {"VLLM_X": 1}},
                    {"name": "empty"},
                    "junk",
                    {"name": "bad mode", "extra_args": "--a", "args_mode": "merge"},
                ],
                "patch_reason": "bridge the predicate",
                "summary": "s",
            }
        )
    )
    patches = (
        PatchFile("vllm/x.py", _DIFF),
        PatchFile("specialist_done.json", "+{}\n"),
        PatchFile("patches/001.patch", "+x\n"),
    )
    got = candidates_of(done, patches, prefix="freeform", max_candidates=6)
    assert [c.name for c in got] == ["freeform-patch", "graphs", "graphs-2", "bad-mode"]
    assert [p.path for p in got[0].patches] == ["vllm/x.py"] and got[0].reason == "bridge the predicate"
    assert got[1].config.extra_server_args == "--cuda-graph-sizes 64" and got[1].citations == ("kb1",)
    assert dict(got[2].config.extra_envs) == {"VLLM_X": "1"}
    assert got[3].config.args_mode == "append"
    assert [c.name for c in candidates_of(done, patches, prefix="freeform", max_candidates=2)] == [
        "freeform-patch",
        "graphs",
    ]
    assert read_done("no payload here") is None
    assert candidates_of(None, (), prefix="p", max_candidates=3) == ()

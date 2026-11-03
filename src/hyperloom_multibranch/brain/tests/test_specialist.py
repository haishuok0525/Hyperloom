# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0's specialist voice: the texts the body sent before the brain was cut out, citing the facts it is handed."""

from __future__ import annotations

from dataclasses import replace

from hyperloom_multibranch.brain.specialist import RECENT_ENTRIES, SpecialistPrompts
from hyperloom_multibranch.contracts.brain_brief import (
    SPECIALIST_LAUNCH_KEY,
    SPECIALIST_ROLE_KEY,
    Arrival,
    FakeGain,
    LaunchAsk,
    LogLine,
    RoleAsk,
    SpecialistFacts,
)
from hyperloom_multibranch.contracts.branch_log import BranchLogType
from hyperloom_multibranch.scheduling.brain_link import specialist_facts

#: The specialist prompt versions the body recorded at the baseline commit, role then launch.
BASELINE_PROMPT_VERSIONS = ("b68bf1487f2c", "8b56e0a39cf0")
FACTS = SpecialistFacts(
    notice="[notice]",
    reroot="[reroot]",
    case_script="case.sh",
    bash_safety="Never kill what you did not start.",
    native_refused=("CMakeLists.txt",),
    fake_gains=(FakeGain("A cache that serves a stale answer.", "stale_cache"),),
    directions=("Alpha: first.", "Beta: second."),
)


def _line(n: int, kind: BranchLogType = BranchLogType.ATTEMPT, **fields: object) -> LogLine:
    return LogLine(f"e{n}", kind, f"entry {n}", **fields)


def test_the_voice_is_the_one_the_body_used_at_the_baseline() -> None:
    versions = SpecialistPrompts(specialist_facts()).versions()

    assert (versions[SPECIALIST_ROLE_KEY], versions[SPECIALIST_LAUNCH_KEY]) == BASELINE_PROMPT_VERSIONS


def test_the_role_cites_the_facts_it_is_handed() -> None:
    role = SpecialistPrompts(FACTS).role(RoleAsk("qwen3-8b  on vllm", native={"vllm": "csrc"}))

    assert "optimizes one workload: qwen3-8b on vllm." in role
    assert 'start with "[notice]"' in role and 'starts with "[reroot]"' in role
    assert "Write case.sh," in role and "- Never kill what you did not start." in role
    assert "- A cache that serves a stale answer. Refused as stale_cache." in role
    assert "Alpha; Beta" in role
    assert "vllm's C++/HIP sources under csrc" in role and "A change to CMakeLists.txt is refused" in role


def test_each_version_follows_the_facts_its_text_cites() -> None:
    base = SpecialistPrompts(FACTS).versions()
    renoticed = SpecialistPrompts(replace(FACTS, notice="[runtime]")).versions()
    redirected = SpecialistPrompts(replace(FACTS, directions=("Alpha: first.",))).versions()

    assert renoticed[SPECIALIST_ROLE_KEY] != base[SPECIALIST_ROLE_KEY]
    assert renoticed[SPECIALIST_LAUNCH_KEY] == base[SPECIALIST_LAUNCH_KEY]
    assert redirected[SPECIALIST_LAUNCH_KEY] != base[SPECIALIST_LAUNCH_KEY]


def test_a_long_log_is_summed_up_past_restart_notices_and_superseded_entries() -> None:
    log = (
        _line(0),
        _line(1, supersedes="e0"),
        _line(2, BranchLogType.RESULT, restart=True),
        *(_line(n, BranchLogType.PLAN) for n in range(3, RECENT_ENTRIES + 3)),
    )
    ask = LaunchAsk("b01", "c0", "c9", 2, "/wt/b01", direction="Alpha: first.", log=log)

    launch = SpecialistPrompts(FACTS).launch(ask)

    summary = launch.split("\n\n")[0]
    assert summary.startswith(f"Your branch so far, from its whole log of {len(log)} entries:")
    assert "Tried, the latest 1 of 1:\n- [e1] attempt: entry 1" in summary
    assert "Results" not in summary and "[e0]" not in summary
    assert f"Next, by your latest plan:\n- [e{RECENT_ENTRIES + 2}] plan" in summary
    assert "Your branch's direction, to work on first:\nAlpha: first." in launch
    assert launch.endswith("\n") and "Your previous session ended" not in launch


def test_a_wake_lists_the_arrivals_then_the_notices() -> None:
    woken = SpecialistPrompts(FACTS).wake([Arrival("result", "r7", "r7 has its result")], [_line(9)])

    assert woken.index("- result r7: r7 has its result") < woken.index("- [e9] attempt: entry 9")

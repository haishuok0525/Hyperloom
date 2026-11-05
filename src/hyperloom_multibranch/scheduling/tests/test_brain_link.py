# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Loading the brain by name: the default, the environment's choice, and the brains this body cannot run."""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from hyperloom_multibranch.brain import BrainV0
from hyperloom_multibranch.contracts.brain import (
    BRAIN_CONTRACT_VERSION,
    BrainSetup,
    CallSite,
    JudgeObservation,
    JudgeRound,
)
from hyperloom_multibranch.contracts.llm import Completion
from hyperloom_multibranch.scheduling.brain_link import (
    BRAIN_ENV,
    JUDGE_FACTS,
    BrainLoadError,
    load_brain,
    specialist_facts,
)

MODULE = "hyperloom_multibranch_test_brains"
#: The repository's declaration of where the brain is, which an outer loop's gate reads.
BRAIN_SCOPE = Path(__file__).resolve().parents[4] / "brain_scope.toml"


@dataclass(frozen=True)
class _Brain:
    setup: BrainSetup
    contract_version: str = "0.9"
    call_sites: frozenset[CallSite] = frozenset({CallSite.JUDGE_ROUND, CallSite.BRANCH_BRIEF})
    name: str = "test"

    @property
    def specialist(self) -> object:
        return object()

    def versions(self, *, plays_playbook: bool) -> Mapping[str, str]:
        return {}

    async def judge_round(self, observation: JudgeObservation, llm: Completion) -> JudgeRound:
        return JudgeRound(None)


@pytest.fixture(autouse=True)
def _brains(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType(MODULE)
    module.Fits = _Brain
    module.NextMajor = lambda setup: _Brain(setup, contract_version="1.0")
    module.NoJudge = lambda setup: _Brain(setup, call_sites=frozenset({CallSite.BRANCH_BRIEF}))
    module.NoBrief = lambda setup: _Brain(setup, call_sites=frozenset({CallSite.JUDGE_ROUND}))
    module.NotCallable = 3
    monkeypatch.setitem(sys.modules, MODULE, module)
    monkeypatch.delenv(BRAIN_ENV, raising=False)


def test_the_default_brain_is_v0_built_for_the_body_numbers() -> None:
    brain = load_brain()

    assert isinstance(brain, BrainV0)
    assert brain.setup == BrainSetup(judge=JUDGE_FACTS, specialist=specialist_facts())


def test_the_environment_names_another_brain_of_the_same_major_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BRAIN_ENV, f"{MODULE}:Fits")

    brain = load_brain()

    assert isinstance(brain, _Brain) and brain.setup.judge == JUDGE_FACTS


@pytest.mark.parametrize(
    ("spec", "why"),
    [
        ("no_such_module_xyz:Brain", "cannot be imported"),
        (f"{MODULE}:Missing", "cannot be imported"),
        (f"{MODULE}", "is not module:factory"),
        (f"{MODULE}:NotCallable", "is not callable"),
        (f"{MODULE}:NextMajor", "speaks contract 1.0"),
        (f"{MODULE}:NoJudge", "does not answer ['judge_round']"),
        (f"{MODULE}:NoBrief", "does not answer ['branch_brief']"),
    ],
)
def test_a_brain_this_body_cannot_run_is_refused(spec: str, why: str) -> None:
    with pytest.raises(BrainLoadError, match=why.replace("[", r"\[").replace("]", r"\]")):
        load_brain(spec)


def test_the_declared_brain_scope_names_this_contract_and_the_brain_package() -> None:
    declared = BRAIN_SCOPE.read_text(encoding="utf-8")

    assert f'contract_version = "{BRAIN_CONTRACT_VERSION}"' in declared
    assert 'brain = ["src/hyperloom_multibranch/brain/"]' in declared
    assert '"src/hyperloom_multibranch/contracts/brain.py"' in declared

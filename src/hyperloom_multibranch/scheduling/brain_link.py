# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Loading the brain a session runs: by name, never by import, so that any brain written against the contract fits.

:data:`BRAIN_ENV` names it as ``module:factory``, :data:`DEFAULT_BRAIN` when unset; the factory is called with the
:class:`~hyperloom_multibranch.contracts.brain.BrainSetup` of the body's numbers (:data:`JUDGE_FACTS`) and words
(:func:`specialist_facts`). A brain that cannot be imported, built, or was written against another major contract
version raises :class:`BrainLoadError`, and so does one that does not answer the judge round and the branch brief,
which every session asks.
"""

from __future__ import annotations

import importlib
import os

from hyperloom.measure.runner.native.recipes import NATIVE_BUILD_REFUSE
from hyperloom.orchestrator.prompts.specialist_prompt_builder import BASH_KILL_SAFETY_PREAMBLE
from hyperloom_multibranch.blackboard import STALL_TESTS
from hyperloom_multibranch.blackboard.rsi.frontier import SETTLED_TESTS
from hyperloom_multibranch.contracts.brain import BRAIN_CONTRACT_VERSION, Brain, BrainSetup, CallSite, JudgeFacts
from hyperloom_multibranch.contracts.brain_brief import FakeGain, SpecialistFacts
from hyperloom_multibranch.rulegate.fix_case import CASE_SCRIPT
from hyperloom_multibranch.scheduling.branches import REPAIR_BRANCH_SECONDS, REPAIR_PROTECTED_SECONDS
from hyperloom_multibranch.scheduling.events import MB_BIG_GAIN_BARS, MB_BOTTLENECK_SHIFT_PP
from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.following import REROOT
from hyperloom_multibranch.specialist.judging import NOTICE

BRAIN_ENV = "HYPERLOOM_MB_BRAIN"
DEFAULT_BRAIN = "hyperloom_multibranch.brain:BrainV0"
JUDGE_FACTS = JudgeFacts(
    stall_tests=STALL_TESTS,
    grace_tests=SETTLED_TESTS,
    repair_protected_seconds=REPAIR_PROTECTED_SECONDS,
    repair_branch_seconds=REPAIR_BRANCH_SECONDS,
    big_gain_bars=MB_BIG_GAIN_BARS,
    bottleneck_shift_pp=MB_BOTTLENECK_SHIFT_PP,
)
#: The call sites every brain must answer.
REQUIRED_SITES = frozenset({CallSite.JUDGE_ROUND, CallSite.BRANCH_BRIEF})


def specialist_facts() -> SpecialistFacts:
    """The body's words a brain's specialist prompts cite."""

    # Imported here: the catalog loads main's flow runtime, which importing the multi-branch command must not.
    from hyperloom.measure.fake_gain import FAKE_GAINS

    return SpecialistFacts(
        notice=NOTICE,
        reroot=REROOT,
        case_script=CASE_SCRIPT,
        bash_safety=BASH_KILL_SAFETY_PREAMBLE,
        native_refused=tuple(NATIVE_BUILD_REFUSE),
        fake_gains=tuple(FakeGain(row.shows_as, row.refusal_code) for row in FAKE_GAINS),
        directions=DIRECTIONS,
    )


class BrainLoadError(RuntimeError):
    """The named brain cannot run this body."""


def _major(version: str) -> str:
    return version.split(".", 1)[0]


def _factory(spec: str) -> object:
    module_name, _, attr = spec.partition(":")
    try:
        return getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise BrainLoadError(f"brain {spec!r} cannot be imported: {exc}") from exc


def load_brain(spec: str | None = None) -> Brain:
    """The brain ``spec`` names, else :data:`BRAIN_ENV`'s, else :data:`DEFAULT_BRAIN`, built for this body."""

    name = (spec or os.environ.get(BRAIN_ENV, "") or DEFAULT_BRAIN).strip()
    if ":" not in name:
        raise BrainLoadError(f"brain {name!r} is not module:factory")
    factory = _factory(name)
    if not callable(factory):
        raise BrainLoadError(f"brain {name!r} is not callable")
    brain: Brain = factory(BrainSetup(judge=JUDGE_FACTS, specialist=specialist_facts()))
    if _major(brain.contract_version) != _major(BRAIN_CONTRACT_VERSION):
        raise BrainLoadError(
            f"brain {name!r} speaks contract {brain.contract_version}, the body {BRAIN_CONTRACT_VERSION}"
        )
    missing = REQUIRED_SITES - brain.call_sites
    if missing:
        raise BrainLoadError(f"brain {name!r} does not answer {sorted(site.value for site in missing)}")
    return brain


__all__ = [
    "BRAIN_ENV",
    "DEFAULT_BRAIN",
    "JUDGE_FACTS",
    "REQUIRED_SITES",
    "BrainLoadError",
    "load_brain",
    "specialist_facts",
]

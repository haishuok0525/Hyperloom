# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The body asks the brain for each specialist's brief, handing it the dispatch's inputs and the body's facts."""

from __future__ import annotations

import dataclasses
import uuid

import pytest

from hyperloom.brain.specialist import inputs as brain_inputs
from hyperloom.brain_api import BrainContractError
from hyperloom.inference_optimizer.tests._brief import brief_inputs, build_specialist_prompts
from hyperloom.orchestrator.prompts.specialist_prompt_builder import SpecialistPromptInputs, brief_from_brain
from hyperloom.orchestrator.specialists import profile
from hyperloom.orchestrator.specialists.domains import get_domain


def _inputs(**kwargs) -> SpecialistPromptInputs:
    return SpecialistPromptInputs(task_id="t", domain=get_domain("serving_specialist"), max_turns=4, **kwargs)


def test_the_brain_sees_every_input_the_body_gathers() -> None:
    body = [(f.name, f.default) for f in dataclasses.fields(SpecialistPromptInputs)]
    brain = [(f.name, f.default) for f in dataclasses.fields(brain_inputs.BriefInputs) if f.name != "facts"]
    assert [name for name, _ in brain] == [name for name, _ in body]
    assert [d for name, d in brain if name != "max_turns"] == [d for name, d in body if name != "max_turns"]
    assert brain_inputs.MODE_PATCH == profile.MODE_PATCH


def test_tuple_inputs_cross_the_wire_as_tuples() -> None:
    inp = brief_inputs(_inputs(allocated_gpu_ids=(0, 1), extra_focus_tags=("comm",)))
    assert inp.allocated_gpu_ids == (0, 1)
    assert inp.extra_focus_tags == ("comm",)
    assert inp.domain == brain_inputs.DomainCard(
        key="serving_specialist",
        layer=inp.domain.layer,
        kb_anchor=inp.domain.kb_anchor,
        description=inp.domain.description,
    )


def test_the_mandate_is_built_only_for_the_enablement_domain() -> None:
    assert brief_inputs(_inputs()).facts.enablement_mandate == ""


def test_a_selfcheck_for_an_unregistered_framework_fails_as_before() -> None:
    with pytest.raises(KeyError, match="no-such-framework"):
        build_specialist_prompts(
            _inputs(framework="no-such-framework", workspace_path="/ws/worktree", worktree_package_dir="pkg")
        )


async def test_the_body_runs_the_brief_the_brain_writes() -> None:
    inp = _inputs(notes="from orchestration", gap_canonical_id="gap-1")
    assert await brief_from_brain(inp) == build_specialist_prompts(inp)


_SILENT_BRAIN = """
from hyperloom.brain_api import BrainTurn

class Brain:
    name = "silent"
    contract_version = "1.0"
    role_intents = {}
    patch_approval_verdicts = ("advise", "approve")

    async def on_llm_turn(self, batch, port):
        return BrainTurn()

    async def on_settlement(self, batch, port):
        return BrainTurn()

    async def on_phase_pump(self, batch, port):
        return BrainTurn()

    async def on_brief(self, batch, port):
        return BrainTurn()

def create_brain(config):
    return Brain()
"""


async def test_a_brain_that_answers_a_brief_without_a_specialist_run_breaks_the_contract(tmp_path, monkeypatch) -> None:
    from hyperloom.plugs.builtin import builtin_registry
    from hyperloom.plugs.host import PlugHost, activated
    from hyperloom.plugs.manifest import ManifestSources, resolve_manifest

    module = f"silent_brain_{uuid.uuid4().hex}"
    (tmp_path / f"{module}.py").write_text(_SILENT_BRAIN, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = builtin_registry()
    sources = ManifestSources(cli_plugs=("brain=v0",), cli_configs=(f"brain.module={module}",))
    host = PlugHost(resolve_manifest(registry, sources), registry, session_dir=None, env={})
    host.start()
    with activated(host):
        with pytest.raises(BrainContractError, match="silent"):
            await brief_from_brain(_inputs())

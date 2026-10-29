# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent role definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir
from hyperloom.inference_optimizer.protocol.intent import IntentType
from hyperloom.orchestrator.brain_link import resolve_brain


@dataclass(frozen=True)
class AgentRole:
    """Static role record. Backend instances are created elsewhere."""

    name: str
    #: What the role may emit; the brain decides it, and PolicyGate refuses the rest.
    allowed_intents: frozenset[IntentType]
    can_delegate_side_effects: bool = False
    no_tools: bool = False

    @property
    def system_prompt_path(self) -> Path:
        """Path to this role's system prompt markdown file."""
        return asset_system_prompts_dir() / f"{self.name}.md"

    def load_system_prompt(self) -> str:
        """Read and return this role's system prompt text."""
        return self.system_prompt_path.read_text(encoding="utf-8")


def _brain_intents(role: str) -> frozenset[IntentType]:
    """The intents the brain lets ``role``'s model emit."""
    return frozenset(IntentType(intent) for intent in resolve_brain().role_intents.get(role, ()))


def default_role_registry() -> dict[str, AgentRole]:
    """Return the canonical orchestration and critic role registry."""
    return {
        "orchestration": AgentRole(
            name="orchestration",
            allowed_intents=_brain_intents("orchestration"),
            can_delegate_side_effects=True,
        ),
        "critic": AgentRole(
            name="critic",
            allowed_intents=_brain_intents("critic"),
            can_delegate_side_effects=False,
            no_tools=True,
        ),
    }


__all__ = [
    "AgentRole",
    "default_role_registry",
]

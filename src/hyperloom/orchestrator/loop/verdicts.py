# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A Critic ruling as the router carries it out, and the advisory fields a review verdict carries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.brain_api import VerdictOutcome

# Advisory fields carried on a Critic ``review_verdict`` payload beyond the bare verdict/reasoning.
_VERDICT_ADVISORY_LIST_KEYS: tuple[str, ...] = (
    "required_evidence",
    "risks",
    "notes",
    "kb_evidence",
    "packet_evidence",
)
_VERDICT_ADVISORY_TEXT_KEYS: tuple[str, ...] = (
    "advice_text",
    "alternative_action",
)


@dataclass(frozen=True)
class CriticRuling:
    """The Critic's collapsed ruling on one proposal, with what it wrote alongside."""

    verdict: str
    reasoning: str
    authored_verdict: str = ""
    advisory: dict[str, Any] | None = None
    approved_variant_names: set[str] | None = None
    payload: Mapping[str, Any] | None = None
    variant_verdicts: list[dict[str, Any]] | None = None
    #: What the brain settled the proposal to under ``verdict``.
    outcome: VerdictOutcome = VerdictOutcome.NONE

    @property
    def authored(self) -> str:
        """The verdict the Critic wrote before any hold to a cited rule; defaults to ``verdict``."""
        return str(self.authored_verdict or self.verdict).strip()

    @property
    def effective(self) -> str:
        """The verdict the loop acts on."""
        return str(self.verdict or "").strip()


def serialize_verdict_advisory(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the advisory field set from a ``review_verdict`` payload."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _VERDICT_ADVISORY_LIST_KEYS:
        raw = payload.get(key)
        if isinstance(raw, (list, tuple)):
            items = [item for item in raw if item not in (None, "")]
        elif raw in (None, ""):
            items = []
        else:
            items = [raw]
        if items:
            out[key] = list(items)
    for key in _VERDICT_ADVISORY_TEXT_KEYS:
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            out[key] = raw
    return out

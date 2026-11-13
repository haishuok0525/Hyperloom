# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist: the brief it runs on, the run on the agent port, and the candidates it delivers."""

from __future__ import annotations

from .brief import build_specialist_prompts
from .deliver import candidates_of, read_done
from .domains import DOMAINS, FREEFORM, gap_domain
from .inputs import BriefFacts, BriefInputs, DomainCard
from .run import brief_inputs, run_specialist, target_gap

__all__ = [
    "DOMAINS",
    "FREEFORM",
    "BriefFacts",
    "BriefInputs",
    "DomainCard",
    "brief_inputs",
    "build_specialist_prompts",
    "candidates_of",
    "gap_domain",
    "read_done",
    "run_specialist",
    "target_gap",
]

# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Advisory proposal scoring: CallSite.SCORE."""

from __future__ import annotations

from hyperloom.brain.score.prompt import (
    MAX_PROPOSALS_SCORED,
    SCORING_INSTRUCTIONS,
    ScoringProposal,
    build_scoring_prompt,
    clip,
    coerce_score,
    normalise_model_scores,
    prepare_scoring_proposals,
)
from hyperloom.brain.score.turn import PROPOSAL_SCORE_KEY, score_proposals

__all__ = [
    "MAX_PROPOSALS_SCORED",
    "PROPOSAL_SCORE_KEY",
    "SCORING_INSTRUCTIONS",
    "ScoringProposal",
    "build_scoring_prompt",
    "clip",
    "coerce_score",
    "normalise_model_scores",
    "prepare_scoring_proposals",
    "score_proposals",
]

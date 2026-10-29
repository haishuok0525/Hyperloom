# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Critic's judgement: what each turn's proposals are held to, what the model is asked, how its reply reads,
which of its verdicts the KB keeps, and what each verdict settles to."""

from __future__ import annotations

from hyperloom.brain.critic.review import CRITIC_ROLE, kb_write_targets, review_turn
from hyperloom.brain.critic.ruling import PATCH_APPROVAL_VERDICTS, SETTLEMENT_KEY, settle_verdict, verdict_outcome

__all__ = [
    "CRITIC_ROLE",
    "PATCH_APPROVAL_VERDICTS",
    "SETTLEMENT_KEY",
    "kb_write_targets",
    "review_turn",
    "settle_verdict",
    "verdict_outcome",
]

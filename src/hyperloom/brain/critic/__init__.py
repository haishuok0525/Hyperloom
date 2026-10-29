# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Critic's judgement: what each turn's proposals are held to, what the model is asked, how its reply reads, and
which of its verdicts the KB keeps."""

from __future__ import annotations

from hyperloom.brain.critic.review import CRITIC_ROLE, kb_write_targets, review_turn

__all__ = ["CRITIC_ROLE", "kb_write_targets", "review_turn"]

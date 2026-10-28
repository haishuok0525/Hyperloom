# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The Critic's judgement: what each turn's proposals are held to, what the model is asked, and how its reply reads."""

from __future__ import annotations

from hyperloom.brain.critic.review import CRITIC_ROLE, review_turn

__all__ = ["CRITIC_ROLE", "review_turn"]

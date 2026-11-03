# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The multi-branch brain: the search decisions behind :mod:`hyperloom_multibranch.contracts.brain`.

It imports the contracts and nothing of the body; the body loads it by name
(:func:`hyperloom_multibranch.scheduling.brain_link.load_brain`), never by import.
"""

from hyperloom_multibranch.brain.v0 import BODY_BASELINE_COMMIT, BrainV0

__all__ = ["BODY_BASELINE_COMMIT", "BrainV0"]

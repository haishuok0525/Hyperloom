# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Brain v0's specialist voice: the role, the launch prompt and the messages between a specialist's turns."""

from hyperloom_multibranch.brain.specialist.prompts import (
    ENTRY_CHARS,
    RECENT_ENTRIES,
    SUMMARY_ENTRIES,
    SUMMARY_ENTRY_CHARS,
    VERSION_HEX,
    SpecialistPrompts,
)

__all__ = [
    "ENTRY_CHARS",
    "RECENT_ENTRIES",
    "SUMMARY_ENTRIES",
    "SUMMARY_ENTRY_CHARS",
    "VERSION_HEX",
    "SpecialistPrompts",
]

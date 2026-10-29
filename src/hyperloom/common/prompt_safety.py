# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared prompt helpers: structure safety, and fitting a growing section to its character budget."""

from __future__ import annotations

import re


def defang_prompt_structure(text: str) -> str:
    """Neutralise sequences that could escape a quoted context in an LLM prompt."""
    out = str(text or "")
    out = out.replace("```", "`\u200b``").replace("~~~", "~\u200b~~")
    out = out.replace("data:", "data\u200b:").replace("DATA:", "DATA\u200b:")
    return out.replace("<", "\u2039").replace(">", "\u203a")


def flatten_for_prompt(text: str) -> str:
    """Fold untrusted multi-line text onto a single prompt line."""
    flat = re.sub(r"[\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029]", "\u23ce", str(text or ""))
    if flat.startswith("="):
        flat = "\u2261" + flat[1:]
    return defang_prompt_structure(flat)


def fit_to_budget(lines: list[str], budget_chars: int) -> list[str]:
    """Return the leading ``lines`` that fit in ``budget_chars``.

    The first line is always kept, clipped to the budget, so a backlog led by
    one oversized entry still drains.
    """
    if not lines:
        return []
    kept = [lines[0][:budget_chars]]
    used = len(kept[0])
    for line in lines[1:]:
        used += len(line) + 1
        if used > budget_chars:
            break
        kept.append(line)
    return kept

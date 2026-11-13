# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist delivers through its reply, and the candidates it comes to.

The ``proposal_set`` item schema is the body's explore-variant schema; the body parses it from a workspace, the
brain from the reply.
"""

from __future__ import annotations

import posixpath
from typing import Any, Mapping, Sequence

from hyperloom.brain_api import Candidate, ConfigDelta, PatchFile, WireError
from hyperloom.brain_api.state import ARGS_MODES
from hyperloom_common.jsonio import extract_last_json_with_key

EXPLORE_VARIANT_EXAMPLE: dict[str, Any] = {
    "name": "<unique-in-round>",
    "extra_args": "--example-flag value",
    "extra_envs": {"EXAMPLE_ENV": "1"},
    "remove_args": ["--harmful-base-flag"],
    "unset_envs": ["HARMFUL_BASE_ENV"],
    "args_mode": "append",
    "reason": "why this might help the gap",
    "atomic": False,
    "kb_evidence": [],
    "pr_evidence": [],
    "source_evidence": [],
    "experience_citations": [],
}
EXPLORE_VARIANT_CONTRACT: tuple[str, ...] = (
    (
        "- ``proposal_set`` items reuse the explore variant schema: "
        "``extra_args`` / ``extra_envs`` add or override knobs, "
        "``remove_args`` removes inherited server flags before appending, "
        "``unset_envs`` removes inherited env vars before applying "
        "``extra_envs``, and ``args_mode='replace'`` runs without "
        "inherited server args. Use removal fields when a user/base "
        "knob may be harmful; do not simulate deletion by adding an "
        "unrelated flag."
    ),
    (
        "- ``atomic`` (bool, default false): set ``true`` when this "
        "proposal's ``extra_args`` / ``extra_envs`` are a **coupled set "
        "that only works together** and MUST be benched as one variant "
        "(e.g. enabling MTP/speculative decoding REQUIRES a paired "
        "``--gpu-memory-utilization`` reduction so the draft model has "
        "headroom — split them and each half OOMs or shows no gain). "
        "Put every co-required flag in THIS one entry; do not scatter a "
        "coupling across several proposals."
    ),
)

#: ``BriefFacts.proposal_schemas`` of a dispatch whose items the brain parses.
PROPOSAL_SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    "explore_variant": {"example": EXPLORE_VARIANT_EXAMPLE, "contract": list(EXPLORE_VARIANT_CONTRACT)},
}

_EVIDENCE_KEYS = ("kb_evidence", "pr_evidence", "source_evidence", "experience_citations")
# Files a specialist writes for a workspace dispatcher; in the tree it runs in they are leftovers, not source.
_LEFTOVER_NAMES = ("specialist_done", "heartbeat", "inbox")
_LEFTOVER_SUFFIXES = (".patch", ".diff", ".orig", ".rej")


def read_done(text: str) -> Mapping[str, Any] | None:
    """The ``specialist_done`` payload a reply ends with, or ``None`` when it carries none."""
    return extract_last_json_with_key(text, "proposal_set")


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("id") or item.get("url") or item.get("kb_id") or ""
        if isinstance(item, (str, int)) and str(item).strip():
            out.append(str(item).strip())
    return out


def _citations(entry: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(c for key in _EVIDENCE_KEYS for c in _strings(entry.get(key))))


def config_candidate(entry: Any, name: str) -> Candidate | None:
    """The candidate one ``proposal_set`` item stands for, or ``None`` when it is malformed or changes nothing."""
    if not isinstance(entry, Mapping):
        return None
    envs = entry.get("extra_envs") or {}
    mode = str(entry.get("args_mode") or "append")
    try:
        config = ConfigDelta(
            extra_server_args=str(entry.get("extra_args") or entry.get("extra_server_args") or "").strip(),
            extra_envs={str(k): str(v) for k, v in envs.items()} if isinstance(envs, Mapping) else {},
            remove_args=tuple(_strings(entry.get("remove_args"))),
            unset_envs=tuple(_strings(entry.get("unset_envs"))),
            args_mode=mode if mode in ARGS_MODES else "append",
        )
        return Candidate(name, config=config, reason=str(entry.get("reason") or ""), citations=_citations(entry))
    except WireError:
        return None


def source_patches(patches: Sequence[PatchFile]) -> tuple[PatchFile, ...]:
    """``patches`` without the workspace files a specialist may have left in the tree."""
    kept = []
    for patch in patches:
        base = posixpath.basename(patch.path)
        if base.startswith(_LEFTOVER_NAMES) or base.endswith(_LEFTOVER_SUFFIXES):
            continue
        if patch.path.split("/", 1)[0] == "patches":
            continue
        kept.append(patch)
    return tuple(kept)


def _unique(name: str, taken: set[str]) -> str:
    stem = "".join(c if c.isalnum() or c in "-_." else "-" for c in name.strip()).strip("-") or "variant"
    out, n = stem, 1
    while out in taken:
        n += 1
        out = f"{stem}-{n}"
    taken.add(out)
    return out


def candidates_of(
    done: Mapping[str, Any] | None,
    patches: Sequence[PatchFile],
    *,
    prefix: str,
    max_candidates: int,
) -> tuple[Candidate, ...]:
    """The candidates of a specialist run, best first: its source edit as one candidate, then its config variants
    in its own order, at most ``max_candidates``. Names are unique, the patch's is ``<prefix>-patch``."""
    done = done or {}
    taken: set[str] = set()
    out: list[Candidate] = []
    edits = source_patches(patches)
    if edits:
        reason = str(done.get("patch_reason") or done.get("summary") or "")
        out.append(
            Candidate(_unique(f"{prefix}-patch", taken), patches=edits, reason=reason, citations=_citations(done))
        )
    items = done.get("proposal_set")
    for entry in items if isinstance(items, list) else []:
        if len(out) >= max_candidates:
            break
        raw = str(entry.get("name") or "") if isinstance(entry, Mapping) else ""
        candidate = config_candidate(entry, _unique(raw or f"{prefix}-variant", taken))
        if candidate is not None:
            out.append(candidate)
    return tuple(out[:max_candidates])

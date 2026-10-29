# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the brain is told about a settled specialist's deliverable, and the config its review carries once the brain
sends it to the Critic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.coerce import first_str, to_mapping
from hyperloom.inference_optimizer.grid_server_args import dedupe_extra_server_args, merge_server_args

from ..specialists.profile import is_authoring_specialist

# Snapshot keys the brain decides a settled specialist's follow-ups under.
PATCH_DELIVERY_KEY = "specialist_patch_delivery"
CONFIG_DELIVERY_KEY = "specialist_config_delivery"
REAUTHOR_KEY = "critic_reauthor"
REAUTHOR_DECISIONS = ("reauthor.dispatch", "reauthor.decline", "reauthor.exhausted")
# The terminal progress row, ``(status, provenance)``, each re-author close stamps.
REAUTHOR_CLOSES: Mapping[str, tuple[str, str]] = {
    "reauthor.decline": ("needs_review_no_evidence", "critic"),
    "reauthor.exhausted": ("reauthor_cap", "pump"),
}


def specialist_resolve_bases(session_dir: Path | str | None, sid: str) -> list[Path]:
    """The roots a specialist's claimed files resolve under: its worktree, then its run directory."""
    from hyperloom.inference_optimizer.session.session_paths import runs_dir

    if session_dir is None:
        return []
    spec_root = runs_dir(Path(session_dir), "specialist", sid)
    return [spec_root / "worktree", spec_root]


def config_delivery(sid: str, spec_params: Mapping[str, Any], done_payload: dict[str, Any]) -> dict[str, Any]:
    """What the brain decides a specialist's config-lever deliverable on."""
    patches = done_payload.get("patches_written") or []
    build_request = done_payload.get("needs_targeted_build")
    return {
        "specialist_task_id": sid,
        "authoring": is_authoring_specialist(dict(spec_params)),
        # A patch deliverable is handled by the patch autosubmit bridge.
        "patches_written": isinstance(patches, list) and bool(patches),
        "enablement": bool(spec_params.get("enablement")),
        "build_requested": isinstance(build_request, dict) and bool(build_request),
        "setup_or_artifacts": bool(done_payload.get("setup_commands")) or bool(done_payload.get("artifacts_written")),
    }


def patch_deliverable(
    companion_levers: Mapping[str, Any], established: Mapping[str, Any] | None, *, authoring: bool
) -> dict[str, Any]:
    """The config a specialist's patches are reviewed with: the lever riding with them, on top of what enablement
    has ``established`` when this is an enablement round, and the authoring flag."""
    round_args = first_str(companion_levers.get("extra_server_args"))
    round_envs = to_mapping(companion_levers.get("extra_envs"))
    if established is not None:
        # A flag the architecture requires stays required once a later round moves on to a different blocker, so
        # the round launches from everything enablement has established, with its own values winning a collision.
        round_envs = {**{str(k): str(v) for k, v in to_mapping(established.get("extra_envs")).items()}, **round_envs}
        round_args = dedupe_extra_server_args(
            merge_server_args(first_str(established.get("extra_server_args")), round_args)
        )
    deliverable: dict[str, Any] = {}
    if round_args or round_envs:
        deliverable["extra_server_args"] = round_args
        deliverable["extra_envs"] = round_envs
    if authoring:
        deliverable["framework_agent_authoring"] = True
    return deliverable

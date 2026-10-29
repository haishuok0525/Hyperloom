# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for the CLI front door, run in-process via :func:`runtime.cli.main`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.agents.critic.runtime.cli import main


def _write(path: Path, body) -> Path:
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _isolate_session_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("CRITIC_SESSION_MEMORY_DIR", str(tmp_path / "sm"))
    monkeypatch.setenv("CRITIC_KB_CLIENT_MODE", "inmemory")
    monkeypatch.setenv("KB_DEAD_LETTER_DIR", str(tmp_path / "dlq"))
    yield


def _inbox_request(tmp_path: Path, session_id: str, msg_id: str) -> Path:
    """A coordinator-inbox request carrying one baseline proposal."""
    return _write(
        tmp_path / "req.json",
        {
            "session_id": session_id,
            "raw_prompt": (
                "=== Shared session state ===\n"
                "model=qwen3-14b framework=sglang workload=decode precision=fp8\n"
                "=== Inbox for critic ===\n"
                f"  seq=1 msg_id={msg_id} from=orchestration topic=proposal payload={{'action_name': 'baseline'}}\n"
            ),
        },
    )


def test_prepare_and_commit_review_for_coordinator_inbox(tmp_path, capsys):
    request = _inbox_request(tmp_path, "sess_cli_e2e", "cli01")
    rc = main(["prepare-review", "--request", str(request), "--out", str(tmp_path / "judge.json")])
    assert rc == 0
    judge = json.loads((tmp_path / "judge.json").read_text("utf-8"))
    assert judge["proposals"][0]["msg_id"] == "cli01"

    review = _write(
        tmp_path / "review.json",
        {
            "review_verdicts": [
                {
                    "target_proposal_msg_id": "cli01",
                    "verdict": "approve",
                    "reasoning": "ok",
                }
            ]
        },
    )
    rc = main(
        [
            "commit-review",
            "--request",
            str(request),
            "--review",
            str(review),
            "--out",
            str(tmp_path / "emit.json"),
        ]
    )
    assert rc == 0
    emit = json.loads((tmp_path / "emit.json").read_text("utf-8"))
    assert emit["intent_envelope"]["intents"][0]["payload"]["verdict"] == "approve"


@pytest.mark.parametrize(("kb_writes", "written"), [([], 0), (["--kb-write", "cli02"], 1)])
def test_commit_review_keeps_a_lesson_only_for_a_verdict_it_is_told_to(tmp_path, kb_writes, written):
    request = _inbox_request(tmp_path, "sess_cli_kb", "cli02")
    verdict = {"target_proposal_msg_id": "cli02", "verdict": "reject", "reasoning": "r", "persist_to_kb": True}
    review = _write(tmp_path / "review.json", {"review_verdicts": [verdict]})
    out = tmp_path / "emit.json"
    rc = main(["commit-review", "--request", str(request), "--review", str(review), *kb_writes, "--out", str(out)])
    assert rc == 0
    assert len(json.loads(out.read_text("utf-8"))["kb_writes"]) == written


def test_invalid_request_returns_exit_code_2(tmp_path):
    bad = _write(tmp_path / "bad.json", {"session_id": "s"})
    rc = main(["prepare-review", "--request", str(bad)])
    assert rc == 2


@pytest.mark.parametrize(
    "command",
    ["init-session", "close-session", "list-priors", "write-verdict", "write-kb-drafts", "add-contradiction"],
)
def test_removed_subcommands_are_rejected(command):
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2


def test_replay_dead_letter_empty_queue(tmp_path):
    out_path = tmp_path / "replay.json"
    rc = main(
        [
            "replay-dead-letter",
            "--dir",
            str(tmp_path / "dlq_replay"),
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    summary = json.loads(out_path.read_text("utf-8"))
    assert isinstance(summary, dict)


def test_replay_dead_letter_dispatches_queued_records(tmp_path):
    from hyperloom.agents.critic.runtime.cli import _replay_dispatch
    from hyperloom.agents.critic.runtime.dead_letter import DeadLetter
    from hyperloom.agents.critic.runtime.errors import RuntimeAdapterError
    from hyperloom.agents.critic.runtime.in_memory_kb_client import InMemoryKBClient

    dlq_dir = tmp_path / "dlq_dispatch"
    dlq = DeadLetter(root=dlq_dir)
    common = {"attempts": 1, "last_error": "boom"}
    # Endpoints stored in the DLQ use filesystem-safe names (no "/").
    dlq.append("upsert", {"id": "kb_x", "kind": "pitfall", "scope": {}}, **common)

    out_path = tmp_path / "replay2.json"
    rc = main(
        [
            "replay-dead-letter",
            "--dir",
            str(dlq_dir),
            "--keep-on-success",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    summary = json.loads(out_path.read_text("utf-8"))
    assert isinstance(summary, dict)

    client = InMemoryKBClient()
    with pytest.raises(RuntimeAdapterError):
        _replay_dispatch(client, "nope", {})


def test_commit_review_rejects_non_object_review(tmp_path):
    request = _write(tmp_path / "req.json", {"raw_prompt": ""})
    review = _write(tmp_path / "review.json", ["not", "a", "dict"])
    rc = main(["commit-review", "--request", str(request), "--review", str(review)])
    assert rc == 2


def test_resolve_kb_client_live_requires_url(monkeypatch):
    from hyperloom.agents.critic.runtime.cli import _resolve_kb_client
    from hyperloom.agents.critic.runtime.errors import RuntimeAdapterError

    monkeypatch.setenv("CRITIC_KB_CLIENT_MODE", "live")
    monkeypatch.delenv("KB_BASE_URL", raising=False)
    with pytest.raises(RuntimeAdapterError, match="KB_BASE_URL"):
        _resolve_kb_client()

    monkeypatch.setenv("KB_BASE_URL", "http://kb.invalid")
    client = _resolve_kb_client()
    assert client is not None

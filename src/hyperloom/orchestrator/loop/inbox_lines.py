# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One compact, high-signal line per inbox message, as the prompts and the inbox context tools render it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hyperloom.common.coerce import to_str_list
from hyperloom.common.prompt_safety import defang_prompt_structure as _defang_prompt_structure
from hyperloom.common.prompt_safety import flatten_for_prompt as _flatten_for_inbox

from ..mcp_tool_names import qualified_context_tool_name
from ..state.failure_evidence import UNMEASURED_OUTCOMES, render_failure_line
from .verdicts import serialize_verdict_advisory

if TYPE_CHECKING:
    from ..bus.message_bus import Message

# Result keys surfaced in delegated_result inbox line; first match wins per group.
_OUTCOME_GAIN_KEYS: tuple[str, ...] = (
    "validated_gain_pct",
    "gain_pct",
    "predicted_gain_pct",
    "delta_pct",
)
_OUTCOME_TPUT_KEYS: tuple[str, ...] = (
    "tokens_per_s",
    "tput",
    "throughput",
    "tput_tok_s",
)
_OUTCOME_STATUS_KEYS: tuple[str, ...] = ("status", "verdict", "outcome", "runner_status")
# Notes rendered per inbox line.
_OUTCOME_NOTES_MAX: int = 3


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    """Return ``d[k]`` for the first ``k`` in ``keys`` present + non-None."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _defang_alert_payload(value: Any) -> Any:
    """Recursively defang string leaves of an alert payload (keys untouched)."""
    if isinstance(value, str):
        return _defang_prompt_structure(value)
    if isinstance(value, dict):
        return {k: _defang_alert_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_defang_alert_payload(v) for v in value]
    return value


def _variant_failure_rows(result: dict[str, Any], max_variant_rows: int) -> list[str]:
    """Indented rows for the unmeasured per-variant outcomes of a delegated result."""
    pvos = result.get("per_variant_outcomes")
    if not isinstance(pvos, list):
        return []
    failures = [v for v in pvos if isinstance(v, dict) and str(v.get("outcome") or "").upper() in UNMEASURED_OUTCOMES]
    rows: list[str] = []
    for vo in failures[:max_variant_rows]:
        row = dict(vo)
        row["error_excerpt"] = _flatten_for_inbox(vo.get("error_excerpt") or vo.get("reason") or "")
        rows.append("  failure: " + render_failure_line(row, excerpt_chars=120))
    elided = len(failures) - max_variant_rows
    if elided > 0:
        pull = qualified_context_tool_name("get_variant_failures")
        rows.append(f"  (+{elided} more failures; pull {pull})")
    return rows


def _format_delegated_result(head: str, payload: dict[str, Any], max_variant_rows: int) -> str:
    """Render a delegated_result entry: one header line plus a row per unmeasured variant."""
    raw_result = payload.get("result")
    result: dict[str, Any] = raw_result if isinstance(raw_result, dict) else {}
    # Only a specialist result carries this key, so its presence is the test.
    done: dict[str, Any] = result.get("specialist_done") or {}
    parts = [head, f"kind={payload.get('kind')!r}", f"state={payload.get('state')!r}"]
    for label, value, quoted in (
        ("status", _first_present(result, _OUTCOME_STATUS_KEYS), True),
        ("kept", result.get("kept"), True),
        ("gain", _first_present(result, _OUTCOME_GAIN_KEYS), False),
        ("tput", _first_present(result, _OUTCOME_TPUT_KEYS), False),
    ):
        if value is not None:
            parts.append(f"{label}={value!r}" if quoted else f"{label}={value}")
    summary = str(done.get("summary") or "").strip()
    if summary:
        parts.append(f"summary={summary[:400]!r}")
    if done.get("confidence") is not None:
        parts.append(f"confidence={done['confidence']}")
    for label, key in (("findings", "new_findings"), ("questions", "residual_questions")):
        # A model writes these, so a scalar can arrive where the schema asks for a list: it counts as one.
        items = to_str_list(done.get(key))
        if items:
            parts.append(f"{label}={len(items)}")
    # Executors that never raise report the failure inside the result envelope, leaving the top-level error None.
    error = payload.get("error") or result.get("error")
    if error:
        parts.append(f"error={str(error)[:200]!r}")
    # patch_safety_numeric is the Critic's artifact, not a lever here.
    notes = [str(n) for n in result.get("notes") or () if n and not str(n).startswith("patch_safety_numeric:")]
    if notes:
        shown = "; ".join(notes[:_OUTCOME_NOTES_MAX])
        parts.append(f"notes={shown[:300]!r}")
    header_line = " ".join(parts)
    if max_variant_rows <= 0:
        return header_line
    return "\n".join([header_line, *_variant_failure_rows(result, max_variant_rows)])


def _format_review_verdict(head: str, payload: dict[str, Any]) -> str:
    """Render a Critic verdict with the counts of the advisory it carries."""
    parts = [
        f"{head} target={payload.get('target_proposal_msg_id')!r} "
        f"verdict={payload.get('verdict')!r} "
        f"reasoning={str(payload.get('reasoning') or '')[:140]!r}"
    ]
    advisory = serialize_verdict_advisory(payload)
    required_evidence = advisory.get("required_evidence")
    if required_evidence:
        shown = "; ".join(str(item) for item in required_evidence[:3])
        parts.append(f"required_evidence[{len(required_evidence)}]={shown[:140]!r}")
    risks = advisory.get("risks")
    if risks:
        parts.append(f"risks={len(risks)}")
    advice_text = advisory.get("advice_text")
    if advice_text:
        parts.append(f"advice={advice_text[:140]!r}")
    return " ".join(parts)


def _format_inbox_event(m: "Message", *, max_variant_rows: int = 3) -> str:
    """Render one inbox ``Message`` as a compact, high-signal line."""
    topic = (m.topic or "").strip()
    payload = m.payload if isinstance(m.payload, dict) else {}
    # Canonical inbox header ordering that downstream parsers anchor on.
    msg_id = f"msg_id={m.msg_id} " if getattr(m, "msg_id", None) else ""
    head = f"seq={m.seq} {msg_id}from={m.from_agent} topic={topic}"

    if topic == "delegated_result":
        return _format_delegated_result(head, payload, max_variant_rows)
    if topic in ("policy_denial", "denial") or (topic == "observation" and payload.get("kind") == "policy_denial"):
        return (
            f"{head} action={payload.get('action_name')!r} "
            f"rule={payload.get('rule')!r} "
            f"hint={str(payload.get('hint') or '')[:140]!r}"
        )
    if topic == "review_verdict":
        return _format_review_verdict(head, payload)
    if topic == "observation" and payload.get("kind") is not None:
        return f"{head} kind={payload.get('kind')!r} payload={payload}"
    if topic == "alert":
        # Alert payloads can embed attacker-influenceable server.log excerpts; defang string leaves so a log line
        # can't inject prompt structure.
        return f"{head} payload={_defang_alert_payload(payload)}"

    return f"{head} payload={payload}"

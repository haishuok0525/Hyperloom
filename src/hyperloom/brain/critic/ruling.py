# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What one Critic verdict settles to: held to the rule it cites, a per-variant map collapsed to one ruling, and what
its proposal comes to.

The advisory rules are the body's: it hands over their reason codes, and says whether they speak about the proposal's
kind. Holding a reject to the verdict its cited rule declared is the Critic's own discipline, and is decided here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from hyperloom.brain_api import BrainTurn, TickBatch, ToolCall, VerdictOutcome
from hyperloom.common.framework_arm import is_upstream_pr_prescreen

#: Snapshot key the body hands a Critic verdict's settlement under.
SETTLEMENT_KEY = "review_verdict_settlement"

_ADVISE_VERDICT = "advise"
# The verdict that ends a proposal's life; its counterpart ``advise`` lets the proposal through.
_REJECT_VERDICT = "reject"
_NEEDS_REVIEW_VERDICT = "needs_review"
_INTEGRATE_PATCH = "integrate_patch"


@dataclass(frozen=True)
class AdvisoryRules:
    """The advisory-only rules a review answers to: the reason codes whose rule asked for ``advise``, not ``reject``,
    and whether those rules speak about the reviewed proposal's kind."""

    reason_codes: frozenset[str]
    govern: bool


# The fields a Critic states its grounds in: ``reasoning`` on a single verdict, ``rationale`` on one ``verdict_map``
# entry — the per-variant shape PolicyGate documents and every fixture uses.
_VERDICT_PROSE_KEYS: tuple[str, ...] = ("reasoning", "rationale")

# What a citation looks like: the code opens the verdict's grounds and a colon introduces the finding, the shape the
# field verdict used -- ``"specialist_quantitative_claim_violation: the proposal payload carries the forbidden
# predicted_gain_pct field."`` Nothing may precede the code but whitespace or a backtick, and only the opening line of
# each prose field is read.
_CITATION_OPENER: str = r"[ \t]*`?"


def _opening_prose_lines(entry: dict[str, Any]) -> list[str]:
    """Return the opening line of each prose field ``entry`` states grounds in."""
    openings: list[str] = []
    for key in _VERDICT_PROSE_KEYS:
        for line in str(entry.get(key) or "").splitlines():
            if line.strip():
                openings.append(line)
                break
    return openings


def cited_advisory_reason_code(entry: dict[str, Any], reason_codes: frozenset[str]) -> str:
    """Return the advisory-only rule ``entry`` cites, from the field or its prose."""
    explicit = str(entry.get("failure_reason_code") or "").strip()
    if explicit:
        return explicit if explicit in reason_codes else ""
    # At most one code can open one line, so the sort only fixes the order the candidates are tried in.
    for opening in _opening_prose_lines(entry):
        for code in sorted(reason_codes):
            if re.match(rf"{_CITATION_OPENER}{re.escape(code)}`?[ \t]*:", opening):
                return code
    return ""


# Priority a batch of per-variant verdicts collapses by: one approved variant carries the proposal, otherwise one
# reject sinks it, and advice outranks a request for more review. :func:`collapse_verdict_map` applies this to the
# proceedable subset first so a genuine reject cannot sink siblings that may still run.
_VERDICT_COLLAPSE_ORDER: tuple[str, ...] = ("approve", _REJECT_VERDICT, _ADVISE_VERDICT, _NEEDS_REVIEW_VERDICT)
_PROCEEDABLE_VERDICTS: frozenset[str] = frozenset({"approve", _ADVISE_VERDICT})

#: The verdicts on a patch's review subject that let ``integrate_patch`` apply it: the ones that let a proposal run.
PATCH_APPROVAL_VERDICTS: tuple[str, ...] = tuple(sorted(_PROCEEDABLE_VERDICTS))


def collapse_verdicts(verdicts: Iterable[str]) -> str:
    """Collapse per-variant verdicts into the one the proposal is decided on."""
    present = set(verdicts)
    for candidate in _VERDICT_COLLAPSE_ORDER:
        if candidate in present:
            return candidate
    return _NEEDS_REVIEW_VERDICT


def proceedable_variant_names(held_by_name: Mapping[str, str]) -> set[str]:
    """Return variant names whose held verdict lets them reach a benchmark."""
    return {name for name, verdict in held_by_name.items() if verdict in _PROCEEDABLE_VERDICTS and str(name).strip()}


def collapse_verdict_map(held_by_name: Mapping[str, str]) -> tuple[str, set[str] | None]:
    """Collapse a held ``verdict_map`` and name the variants that may run."""
    proceedable = proceedable_variant_names(held_by_name)
    if proceedable:
        return collapse_verdicts(held_by_name[name] for name in proceedable), proceedable
    return collapse_verdicts(held_by_name.values()), None


def _states_findings(value: Any) -> bool:
    """Return whether a findings field states anything at all."""
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    return bool(value)


def verdict_rests_on_one_ground(entry: dict[str, Any]) -> bool:
    """Return whether ``entry`` refuses for a single reason."""
    if _states_findings(entry.get("required_evidence")):
        return False
    risks = entry.get("risks")
    if not isinstance(risks, (list, tuple)):
        return not _states_findings(risks)
    return len([risk for risk in risks if risk]) <= 1


# The findings a review lists outside its prose: the evidence it still wants and the risks it names.
_VERDICT_FINDING_KEYS: tuple[str, ...] = ("required_evidence", "risks")


def _batch_states_findings(payload: dict[str, Any]) -> bool:
    """Return whether a batch review states a finding of its own."""
    if not isinstance(payload, dict):
        return False
    return any(_states_findings(payload.get(key)) for key in _VERDICT_FINDING_KEYS)


def _inheritable_reason_code(payload: dict[str, Any], reason_codes: frozenset[str]) -> str:
    """Return the payload's declared code, when it cannot soften a variant's reject."""
    code = str(payload.get("failure_reason_code") or "").strip()
    return "" if code in reason_codes else code


def verdict_map_entry_grounds(
    entry: dict[str, Any], payload: dict[str, Any], reason_codes: frozenset[str]
) -> dict[str, Any]:
    """Return the grounds one ``verdict_map`` entry rests on."""
    if not isinstance(entry, dict):
        return {}
    grounds = dict(entry)
    if not isinstance(payload, dict):
        return grounds
    if not grounds.get("failure_reason_code"):
        code = _inheritable_reason_code(payload, reason_codes)
        if code:
            grounds["failure_reason_code"] = code
    return grounds


def _stated_verdict(entry: dict[str, Any]) -> str:
    """Return the verdict ``entry`` states, whatever a hold makes of it."""
    return str(entry.get("verdict") or "").strip()


def verdict_held_to_its_rule(entry: dict[str, Any], rules: AdvisoryRules) -> tuple[str, str]:
    """Return the verdict a ``review_verdict`` entry carries, and why it moved."""
    if not isinstance(entry, dict):
        return "", ""
    verdict = _stated_verdict(entry)
    if verdict != _REJECT_VERDICT:
        return verdict, ""
    if not rules.govern:
        return verdict, ""
    if not verdict_rests_on_one_ground(entry):
        return verdict, ""
    reason_code = cited_advisory_reason_code(entry, rules.reason_codes)
    if reason_code:
        return _ADVISE_VERDICT, reason_code
    return verdict, ""


def verdict_map_entry_held_to_its_rule(
    entry: dict[str, Any], payload: dict[str, Any], rules: AdvisoryRules
) -> tuple[str, str]:
    """Return the verdict one ``verdict_map`` entry carries, and why it moved."""
    grounds = verdict_map_entry_grounds(entry, payload, rules.reason_codes)
    if _batch_states_findings(payload):
        return _stated_verdict(grounds), ""
    return verdict_held_to_its_rule(grounds, rules)


def verdict_outcome(verdict: str, action_name: str, payload: Mapping[str, Any]) -> VerdictOutcome:
    """What a proposal comes to under its collapsed verdict."""
    if verdict in _PROCEEDABLE_VERDICTS:
        return VerdictOutcome.MATERIALIZE
    if verdict == _REJECT_VERDICT and is_upstream_pr_prescreen(action_name, payload):
        return VerdictOutcome.RECORD_DENIAL
    params = payload.get("params") or {}
    if verdict == _REJECT_VERDICT and action_name == _INTEGRATE_PATCH and bool(params.get("enablement")):
        return VerdictOutcome.REARM_ENABLEMENT
    if verdict == _NEEDS_REVIEW_VERDICT:
        return VerdictOutcome.REAUTHOR
    return VerdictOutcome.NONE


def _kept(held: tuple[str, str], holds: list[dict[str, str]], variant: str = "") -> str:
    """The verdict ``held`` leaves, noting the hold in ``holds`` when it moved one."""
    verdict, reason_code = held
    if reason_code:
        holds.append({"variant": variant, "to_verdict": verdict, "failure_reason_code": reason_code})
    return verdict


def settle_verdict(batch: TickBatch) -> BrainTurn:
    """The ruling one Critic verdict settles to, as the one ``verdict.settle`` call the body carries out."""
    request = batch.snapshot.data[SETTLEMENT_KEY]
    review, proposal = request["review_verdict"], request["proposal"]
    rules = AdvisoryRules(frozenset(request["advisory_rules"]["reason_codes"]), request["advisory_rules"]["govern"])
    holds: list[dict[str, str]] = []
    verdict = _kept(verdict_held_to_its_rule(review, rules), holds)
    authored = str(review.get("verdict") or "").strip()
    variants: dict[str, str] = {}
    approved: set[str] | None = None
    verdict_map = review.get("verdict_map")
    if not verdict and isinstance(verdict_map, dict) and verdict_map:
        for name, entry in verdict_map.items():
            variants[str(name)] = _kept(verdict_map_entry_held_to_its_rule(entry, review, rules), holds, str(name))
        verdict, approved = collapse_verdict_map(variants)
        authored = collapse_verdicts(str((entry or {}).get("verdict") or "").strip() for entry in verdict_map.values())
    ruling = {
        "verdict": verdict,
        "authored": authored,
        "variants": variants,
        "approved_variants": None if approved is None else sorted(approved),
        "holds": holds,
        "outcome": verdict_outcome(verdict, proposal["action_name"], proposal["payload"]).value,
    }
    return BrainTurn(tool_calls=(ToolCall("verdict.settle", ruling),), brain_state=batch.brain_state)

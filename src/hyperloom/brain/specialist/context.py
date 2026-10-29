# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The per-task context sections of the user prompt: hardware, budget, gap, KB, evidence, priors, PRs, source trees."""

from __future__ import annotations

import json
from pathlib import PurePath
from typing import Any

from hyperloom.common.prompt_safety import defang_prompt_structure

from .inputs import BriefFacts, BriefInputs
from .focus import _is_agentx


_NONE_PLACEHOLDER = "(none)"
# GPU share below which re-dispatching an op to another backend is not worth the
# equivalence work; keeps per-op glue out of the substitution directive.
_VENDOR_SUBSTITUTION_MIN_GPU_PCT = 5.0


# Section 2 — Hardware context
def _section_hardware(inp: BriefInputs) -> list[str]:
    """Render Section 2 (hardware + workload context) of the prompt.

    Emits GPU type, TP, HBM, peak TFLOPs, and any populated workload
    fields (precision, concurrency, ISL/OSL, max_model_len, arch notes).

    Args:
        inp (BriefInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the hardware-context section.
    """
    rows: list[str] = ["## 2. HARDWARE CONTEXT", ""]
    if inp.gpu_type:
        rows.append(f"- gpu_type: {inp.gpu_type}")
    else:
        rows.append(f"- gpu_type: {_NONE_PLACEHOLDER}")
    if inp.allocated_gpu_ids:
        rows.append("- allocated specialist GPU ids: " + ", ".join(str(g) for g in inp.allocated_gpu_ids))
    if inp.tp > 0:
        rows.append(f"- TP: {inp.tp}")
    else:
        rows.append(f"- TP: {_NONE_PLACEHOLDER}")
    if inp.hbm_gb > 0:
        rows.append(f"- HBM per GPU: {inp.hbm_gb:.1f} GB")
    if inp.peak_tflops > 0:
        rows.append(f"- Peak TFLOPs (declared): {inp.peak_tflops:.1f}")
    # Workload context.
    workload_rows: list[str] = []
    if inp.precision:
        workload_rows.append(f"- precision: {inp.precision}")
    if inp.conc > 0:
        workload_rows.append(f"- concurrency: {inp.conc}")
    if _is_agentx(inp):
        # The corpus fixes the request shape, so ISL/OSL carry no information.
        workload_rows += inp.facts.agentx_workload
    else:
        if inp.isl > 0:
            workload_rows.append(f"- ISL (input seq len): {inp.isl}")
        if inp.osl > 0:
            workload_rows.append(f"- OSL (output seq len): {inp.osl}")
    if inp.max_model_len > 0:
        workload_rows.append(f"- max_model_len: {inp.max_model_len}")
    if workload_rows:
        rows.append("")
        rows.append("Workload:")
        rows.extend(workload_rows)
    if inp.arch_notes:
        rows.append("")
        rows.append(f"Model architecture (advisory): {inp.arch_notes}")
    if inp.target_gap_notes:
        rows.append("")
        rows.append(inp.target_gap_notes)
    return rows


# Section 2a — Execution budget (wall-clock)
def _section_execution_budget(inp: BriefInputs) -> list[str]:
    """Render the wall-clock budget block so the specialist can self-throttle.

    Renders the time left on the dispatch deadline (seconds + minutes) and the
    dispatch start timestamp. Returns ``[]`` when no budget was supplied, so the
    section is omitted entirely rather than emitting a placeholder.

    Args:
        inp: The specialist prompt inputs (reads ``wall_budget_sec`` /
            ``started_at_iso``).

    Returns:
        The rendered execution-budget section lines, or ``[]`` when no budget
        is set.
    """
    if inp.wall_budget_sec <= 0:
        return []
    mins = inp.wall_budget_sec / 60.0
    rows = [
        "## 2a. EXECUTION BUDGET (wall-clock)",
        "",
        f"- Hard wall-clock budget for this entire dispatch: **{inp.wall_budget_sec:.0f}s (~{mins:.0f} min)**.",
    ]
    if inp.started_at_iso:
        rows.append(f"- Dispatch started at: {inp.started_at_iso} (UTC).")
    rows.extend(
        [
            "- The Coordinator hard-kills your subprocess when this budget is "
            + "exhausted — turns are NOT the stop signal. Scope your work to "
            + "reach a deliverable conclusion inside the budget.",
            "- Self-throttle: check elapsed wall-clock with Bash "
            + "(``date -u +%s`` vs the start above), keep your "
            + "``specialist_done.partial.json`` checkpoint current, and write "
            + "the final ``specialist_done.json`` before the budget runs out so "
            + "your best work is never lost to a kill.",
        ]
    )
    return rows


# Section 3 — Gap statement
def _section_gap(inp: BriefInputs) -> list[str]:
    """Render Section 3 (gap statement) of the specialist prompt.

    Emits the canonical gap id, layer, symptom, and most-recent evidence
    JSON, or a ``(none)`` placeholder when no gap is set.

    Args:
        inp (BriefInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the gap-statement section.
    """
    rows = ["## 3. GAP STATEMENT", ""]
    if not inp.gap_canonical_id:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append(f"- gap_canonical_id: `{inp.gap_canonical_id}`")
    if inp.gap_layer:
        rows.append(f"- layer: {inp.gap_layer}")
    if inp.gap_symptom:
        rows.append(f"- symptom: {inp.gap_symptom}")
    if inp.gap_evidence:
        rows.append("")
        rows.append("Most recent evidence:")
        rows.append("```json")
        rows.append(json.dumps(inp.gap_evidence, sort_keys=True, indent=2))
        rows.append("```")
    return rows


# Section 4 — optional KB context
def _is_cold_start(inp: BriefInputs) -> bool:
    """Return True when every prior KB/PR/research source is empty, so the
    cold-start directive is injected in place of the KB block.

    Args:
        inp: The specialist prompt inputs.

    Returns:
        True when every prior KB/PR/research source is empty.
    """
    return (
        not inp.kb_subgraph
        and not inp.experience_kb_block
        and not inp.warm_start_recipe
        and not inp.warm_start_lessons
        and not inp.warm_start_pitfalls
        and not inp.research_hints
    )


def _section_kb_subgraph(inp: BriefInputs) -> list[str]:
    """Build the advisory KB-context section of the specialist prompt.

    Falls back to research hints when the structured KB subgraph is empty.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines rendering the KB subgraph (or hint-based fallback).
    """
    rows = ["## 4. KB CONTEXT (optional, advisory)", ""]
    cold = _is_cold_start(inp)
    if not inp.kb_subgraph:
        if inp.research_hints:
            rows.extend(
                [
                    "Structured KB context is empty for this (model, hardware, domain), but "
                    + "the research scout collected source-backed priors this "
                    + "session. Treat these as your advisory prior (co-equal with "
                    + "RecipeKB priors; the Critic still gates the final answer):",
                    "",
                    inp.research_hints,
                    "",
                    "Anchor proposals on these hints where they fit the gap " + "(Section 3) and hardware (Section 2).",
                ]
            )
            return rows
        if cold:
            # Cold-start directive: fall back to the Section 1 defaults, or exit empty with a rationale.
            rows.extend(
                [
                    "**COLD-START MODE — no priors available.**",
                    "",
                    "All prior sources for this gap are empty:",
                    "",
                    "- KB context: ``(none)`` — no RecipeKB warm-start facts or "
                    + "research hints were available for this "
                    + "(model, hardware, domain) tuple.",
                    "- Warm-start recipe: ``(none)`` (Section 5).",
                    "- Use ``mcp__pr_monitor__*`` tools (Section 6) to query PRs on demand.",
                    "",
                    "**Directive — a coin-flip proposal is worse than none.** "
                    + "Treat the *Winning techniques* + *Pitfalls* bullets in "
                    + "Section 1 as your fallback prior and take the **1–2 "
                    + "most conservative, well-attested defaults** that fit "
                    + "the hardware (Section 2) and the gap symptom "
                    + "(Section 3); flag each ``provenance: "
                    + "domain_focus_default`` and call it an unvalidated "
                    + "fallback in the proposal's ``reason``. If none clears "
                    + "that bar, emit ``proposal_set=[]`` and cite in ``summary`` "
                    + "which you considered and why each was rejected — a "
                    + "bare empty exit with no rationale reads as a tool "
                    + "failure. Do NOT add a ``confidence`` field: "
                    + "self-reported confidence / gain fields are stripped "
                    + "before review. Record in ``residual_questions`` what "
                    + "RecipeKB, research, or ``mcp__pr_monitor__*`` query a "
                    + "future round should pursue.",
                ]
            )
        else:
            rows.extend(
                [
                    _NONE_PLACEHOLDER,
                    "",
                    "(No structured KB context supplied. Use Sections 1, 3, 5, "
                    + "and 6 plus source inspection; record missing RecipeKB / "
                    + "research / PR questions in ``residual_questions`` so a "
                    + "future round can warm richer advisory context.)",
                ]
            )
        return rows
    rows.append("```json")
    rows.append(json.dumps(inp.kb_subgraph, sort_keys=True, separators=(",", ":")))
    rows.append("```")
    return rows


def _section_experience_kb(inp: BriefInputs) -> list[str]:
    """Render the Experience KB section; omitted when this dispatch rendered no Experience.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        Prompt lines carrying the Experience service block, or ``[]``.
    """
    if not inp.experience_kb_block:
        return []
    return [
        "## 4b. EXPERIENCE KB (measured outcomes from earlier sessions)",
        "",
        "Compare each Experience's identity and baseline configuration with Sections 2 and 3 "
        + "before relying on it. When one shaped a proposal, cite it in that proposal's "
        + "``experience_citations``, or for a patch you wrote in the payload's top-level "
        + "``experience_citations``: ``{id, stance, claim}`` with ``stance`` one of ``adopt`` "
        + "(you did its change), ``adapt`` (you did it modified), ``avoid`` (you left it out "
        + "because of its outcome), or ``contrast`` (you chose a different change designed "
        + "against it), and ``claim`` one sentence on why. Only ids shown below are kept.",
        "",
        inp.experience_kb_block,
    ]


def _vendor_substitution_candidates(hot_kernels: Any) -> list[dict[str, Any]]:
    """Select hot ``aten::`` ops worth re-dispatching to a different backend.

    An ``aten::`` name means the op still runs through PyTorch's own dispatch, so
    an alternative backend is available to the call site. A vendor kernel that is
    already in the trace under its own name has nothing left to swap.

    Args:
        hot_kernels: The ``hot_kernels_top15`` rows from the roofline evidence.

    Returns:
        Qualifying rows ordered by descending GPU share.
    """
    if not isinstance(hot_kernels, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in hot_kernels:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name.startswith("aten::"):
            continue
        gpu_pct = entry.get("gpu_pct")
        if not isinstance(gpu_pct, (int, float)) or float(gpu_pct) < _VENDOR_SUBSTITUTION_MIN_GPU_PCT:
            continue
        rows.append(entry)
    rows.sort(key=lambda row: float(row.get("gpu_pct") or 0.0), reverse=True)
    return rows


def _vendor_substitution_directive(hot_kernels: Any) -> list[str]:
    """Render the backend-substitution directive for hot ATen ops (empty when none qualify)."""
    candidates = _vendor_substitution_candidates(hot_kernels)
    if not candidates:
        return []
    rows = [
        "**Backend substitution — these ops still dispatch through PyTorch:**",
        "",
        "| name | gpu_pct | category | call site |",
        "|---|---:|---|---|",
    ]
    for entry in candidates:
        gpu_pct = float(entry.get("gpu_pct") or 0.0)
        rows.append(
            f"| {entry.get('name')} | {gpu_pct:.2f}% | "
            f"{entry.get('kernel_category') or ''} | {entry.get('source_file') or ''} |"
        )
    rows.extend(
        [
            "",
            "You may not rewrite the body of a kernel PyTorch owns. You MAY change "
            "which kernel the **call site** dispatches to, and when the call site is "
            "in the tree you are optimizing that is an ordinary source rewrite — "
            "same class as any other switch you author, behind its own default-off "
            "environment switch, with the original path kept as the fallback.",
            "",
            "Section 7 lists the accelerator libraries installed in this container. "
            "Read them to find an entry point matching the op's semantics and "
            "tensor layout, and verify equivalence on the real shapes before "
            "proposing it. An op holding a large share of device time is worth more "
            "than anything you can win from the glue around it.",
        ]
    )
    return rows


def _section_runtime_findings(inp: BriefInputs) -> list[str]:
    """Render the scanned runtime findings; omitted when none were injected."""
    if not inp.runtime_findings:
        return []
    return [
        "## 4c. RUNTIME FINDINGS (current best server.log, already scanned)",
        "",
        "Do not re-read the server log for these. A disabled or falling-back hot "
        "path is restored first; do not optimize the fallback implementation.",
        "",
        "```text",
        inp.runtime_findings,
        "```",
    ]


def _section_roofline_evidence(inp: BriefInputs) -> list[str]:
    """Render the ROOFLINE EVIDENCE section from ``inp.roofline_evidence``;
    empty evidence renders a heading + ``(none)`` placeholder.

    Args:
        inp: The specialist prompt inputs (reads ``roofline_evidence``).

    Returns:
        The rendered roofline-evidence section lines.
    """
    rows = ["## 4a. ROOFLINE EVIDENCE", ""]
    ev = inp.roofline_evidence
    if not ev:
        rows.append(
            "(none — no fresh roofline snapshot has been recorded yet. "
            "The Coordinator auto-enqueues `roofline` at the end of "
            "PRELUDE and again after every 10% watermark crossing; if "
            "you are seeing this, the snapshot is still in-flight.)"
        )
        return rows

    snap_id = ev.get("roofline_snapshot_id")
    if snap_id is not None:
        rows.append(f"**TraceLens snapshot #{snap_id}**")
        rows.append("")

    summary = ev.get("executive_summary")
    if summary:
        rows.append("**Executive Summary:**")
        for label, key in (
            ("Compute %", "compute_pct"),
            ("Idle %", "idle_pct"),
            ("Exposed Comm %", "comm_pct"),
            ("Exposed memcpy %", "memcpy_pct"),
            ("Top bottleneck", "top_bottleneck"),
        ):
            val = summary.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, (int, float)):
                rows.append(f"- {label}: {float(val):.1f}%")
            else:
                rows.append(f"- {label}: {val}")
        rows.append("")

    hot = ev.get("hot_kernels_top15")
    if hot:
        rows.append("**Top hot kernels (kernel_id | name | gpu_pct | bottleneck | source_file):**")
        rows.append("")
        rows.append("| kernel_id | name | gpu_pct | bottleneck | source_file |")
        rows.append("|---|---|---:|---|---|")
        for k in hot:
            kid = str(k.get("kernel_id") or "")
            name = str(k.get("name") or "")
            gpu_pct = k.get("gpu_pct")
            gpu_pct_str = f"{float(gpu_pct):.2f}%" if isinstance(gpu_pct, (int, float)) else "—"
            bottleneck = str(k.get("bottleneck") or "")
            src = str(k.get("source_file") or "")
            rows.append(f"| `{kid}` | {name} | {gpu_pct_str} | {bottleneck} | {src} |")
        rows.append("")

    directive = _vendor_substitution_directive(hot)
    if directive:
        rows.extend(directive)
        rows.append("")

    analysis_path = ev.get("analysis_md_path")
    if analysis_path:
        rows.append(f"**Full analysis.md path:** `{analysis_path}`")
        rows.append("")
        rows.append(
            "Use the `Read` tool on this path for the full TraceLens "
            "report (~10-20 KB). All section headings are stable: "
            "`## Executive Summary` / `## Top Operations` / "
            "`## Compute Kernel Optimizations` / "
            "`## Kernel Fusion Opportunities` / "
            "`## System-Level Optimizations` / `## Recommendations`."
        )
    return rows


# Section 5 — Recipe summary
def _section_recipe(inp: BriefInputs) -> list[str]:
    """Render Section 5 (warm-start recipe summary) of the prompt.

    Dumps the T0 warm-start recipe as JSON, or a ``(none)`` placeholder
    when no warm-start recipe was supplied.

    Args:
        inp (BriefInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the recipe-summary section.
    """
    rows = ["## 5. WARM-START RECIPE SUMMARY", ""]
    if not inp.warm_start_recipe:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    rows.append("**find-recipe result:**")
    rows.append("```json")
    rows.append(json.dumps(inp.warm_start_recipe, sort_keys=True, separators=(",", ":")))
    rows.append("```")
    return rows


# Section 5b — Related lessons (positive priors from prior KEEPs)
def _section_lessons(inp: BriefInputs) -> list[str]:
    """Render the recipe row's ``lessons`` from prior KEEPs, compactly
    (statement + measured_impact).

    Rows are the flat shape ``Recipe.to_dict`` writes —
    ``{statement, measured_impact, ...}`` — normalised by ``recipe_kb_t0``
    before they land on ``warm_start_lessons``.

    Args:
        inp: The specialist prompt inputs (reads ``warm_start_lessons``).

    Returns:
        The rendered related-lessons section lines.
    """
    rows = ["## 5b. RELATED LESSONS (prior KEEPs on this model+hw)", ""]
    if not inp.warm_start_lessons:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for point in inp.warm_start_lessons:
        # External warm-start data may arrive as a plain string rather than a
        # dict "point"; render the bare statement to tolerate shape drift.
        if isinstance(point, str):
            statement = point.strip()
            if statement:
                rows.append(f"- **{statement}**")
            continue
        if not isinstance(point, dict):
            continue
        statement = str(point.get("statement") or "").strip()
        if not statement:
            continue
        impact_str = _render_measured_impact(point.get("measured_impact"))
        conf = point.get("confidence")
        meta_bits: list[str] = []
        if isinstance(conf, (int, float)) and conf > 0:
            meta_bits.append(f"conf={float(conf):.2f}")
        # validated_count is the strongest cross-session signal; fall back to source_session_id.
        vc = point.get("validated_count")
        if isinstance(vc, int) and vc > 1:
            meta_bits.append(f"validated={vc}")
        recent_ids = point.get("source_session_ids")
        if isinstance(recent_ids, list) and recent_ids:
            meta_bits.append(f"recent={recent_ids[-1]}")
        else:
            src_sid = str(point.get("source_session_id") or "").strip()
            if src_sid:
                meta_bits.append(f"src={src_sid}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        # Version-mismatch annotation; the LLM gets the final call.
        version_note = _format_version_note(inp, point)
        rows.append(f"- **{defang_prompt_structure(statement)}**{meta}{version_note}")
        if impact_str:
            rows.append(f"    impact: {impact_str}")
    if len(rows) == 2:  # only the header + blank line, all lessons filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows


def _format_version_note(
    inp: BriefInputs,
    row: dict[str, Any],
) -> str:
    """Render a ``[from sglang@X.Y, you're on A.B]`` annotation when
    the row's framework_version differs; empty when either side is
    unknown or they match.

    Args:
        inp: The specialist prompt inputs (reads ``framework`` /
            ``framework_version``).
        row: The lesson or pitfall row (reads ``framework_version``).

    Returns:
        The version-mismatch annotation, or "" when unknown or matching.
    """
    lesson_fv = str(row.get("framework_version") or "").strip()
    current_fv = (inp.framework_version or "").strip()
    if not lesson_fv or not current_fv:
        return ""
    if lesson_fv == current_fv:
        return ""
    framework_label = (inp.framework or "framework").strip() or "framework"
    return f" [from {framework_label}@{lesson_fv}, you're on {current_fv}]"


def _render_measured_impact(raw: Any) -> str:
    """Back-compat renderer for a row's ``measured_impact`` (dict, legacy
    string, or other).

    Args:
        raw: The ``measured_impact`` value (dict, string, None, or other).

    Returns:
        A compact human-readable impact string ("" when ``raw`` is None).
    """
    if isinstance(raw, dict):
        parts: list[str] = []
        gain = raw.get("gain_pct")
        if isinstance(gain, (int, float)):
            parts.append(f"+{float(gain):.2f}%")
        tput = raw.get("throughput_after")
        if isinstance(tput, (int, float)):
            parts.append(f"tput={float(tput):.1f}")
        depth = raw.get("stack_depth_at_apply")
        if isinstance(depth, int):
            parts.append(f"depth={depth}")
        when = str(raw.get("measured_at") or "").strip()
        if when:
            parts.append(when[:10])  # keep yyyy-mm-dd for compactness
        return ", ".join(parts)
    if isinstance(raw, str):
        return raw.strip()
    if raw is None:
        return ""
    return str(raw).strip()


# Section 5c — Known pitfalls (anti-priors from prior REVERTs)
def _section_pitfalls(inp: BriefInputs) -> list[str]:
    """Render the recipe row's ``pitfalls`` from prior REVERTs (description +
    severity); framed as forbidden paths, not suggestions.

    Rows are the flat ``{description, severity, ...}`` shape ``Recipe.to_dict``
    writes, normalised by ``recipe_kb_t0``.

    Args:
        inp: The specialist prompt inputs (reads ``warm_start_pitfalls``).

    Returns:
        The rendered known-pitfalls section lines.
    """
    rows = ["## 5c. KNOWN PITFALLS (do NOT repeat — prior REVERTs)", ""]
    if not inp.warm_start_pitfalls:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    for point in inp.warm_start_pitfalls:
        # Tolerate plain-string pitfalls alongside the structured dict "point".
        if isinstance(point, str):
            description = point.strip()
            if description:
                rows.append(f"- **{description}**")
            continue
        if not isinstance(point, dict):
            continue
        description = str(point.get("description") or "").strip()
        if not description:
            continue
        severity = str(point.get("severity") or "").strip()
        conf = point.get("confidence")
        meta_bits: list[str] = []
        if severity:
            meta_bits.append(f"severity={severity}")
        if isinstance(conf, (int, float)) and conf > 0:
            meta_bits.append(f"conf={float(conf):.2f}")
        vc = point.get("validated_count")
        if isinstance(vc, int) and vc > 1:
            meta_bits.append(f"observed={vc}")
        recent_ids = point.get("source_session_ids")
        if isinstance(recent_ids, list) and recent_ids:
            meta_bits.append(f"recent={recent_ids[-1]}")
        else:
            src_sid = str(point.get("source_session_id") or "").strip()
            if src_sid:
                meta_bits.append(f"src={src_sid}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        version_note = _format_version_note(inp, point)
        rows.append(f"- **{description}**{meta}{version_note}")
    if len(rows) == 2:  # only the header + blank line, all pitfalls filtered out
        rows.append(_NONE_PLACEHOLDER)
    return rows


# Section 6 — PR query capability
def _section_pr_feed(inp: BriefInputs) -> list[str]:
    """Render Section 6 (PR query capability) of the specialist prompt.

    When the PR Monitor MCP is available, describes the self-serve query tools
    and lists the global repo allowlist the specialist may query. When
    unavailable, outputs a placeholder.

    Args:
        inp (BriefInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the PR-query capability section.
    """
    rows = ["## 6. PR MONITOR", ""]
    if not inp.pr_monitor_available:
        rows.append("(unavailable: pr_monitor disabled)")
        return rows
    _prefix = "mcp__pr_monitor__"
    tool_shorts = sorted(t[len(_prefix) :] for t in inp.facts.pr_monitor_tools if t.startswith(_prefix))
    rows += [
        "Use ``mcp__pr_monitor__*`` tools to query PRs on demand:",
        " / ".join(f"``{t}``" for t in tool_shorts),
        "",
        "Repos you may query:",
    ]
    for repo in inp.facts.pr_query_repos:
        rows.append(f"- {repo}")
    return rows


# Section 7 — Local source navigation hint
def _source_root_row(root: str, *, worktree_base: str, facts: BriefFacts) -> str:
    """Render one source root, annotated with what a patch against it can do.

    Args:
        root (str): The source root to render.
        worktree_base (str): The tree the specialist's worktree stands for,
            when there is one.
        facts (BriefFacts): Which of the trees are git checkouts.

    Returns:
        str: A markdown list row for the root.
    """
    base = (worktree_base or "").rstrip("/")
    if base and PurePath(root.rstrip("/")).is_relative_to(base):
        if facts.worktree_base_has_git:
            return f"- {root} — git checkout; your worktree was cut from it"
        return f"- {root} — installed package; your worktree holds a git snapshot of it, so edit its files there"
    if root in facts.git_checkouts:
        return f"- {root} — git checkout"
    return f"- {root} — installed package, no git tree: re-author upstream diffs against it, never apply them"


def _resolve_focus_dir(hint: str, session_tree: str, *, tree_is_package: bool) -> str:
    """Return ``hint`` as an absolute path under the session tree.

    Checklist directories are repo-relative (``vllm/model_executor/...``), and
    what that is relative to depends on the tree's shape: a pip-installed tree is
    the package directory itself (``.../dist-packages/vllm/``) and joins onto its
    parent, while a checkout (``/sgl-workspace/vllm/``) holds the package one
    level down and joins directly. The tree decides, not the hint -- matching on
    the hint's leading segment reads a checkout as a package tree whenever the
    two share a name, which is every entry in :data:`_DEFAULT_SOURCE_ROOTS`.

    Args:
        hint (str): The focus directory, repo-relative or absolute.
        session_tree (str): The tree this session is optimising, or ``""``.
        tree_is_package (bool): Whether the tree holds an ``__init__.py``.

    Returns:
        str: The hint unchanged when absolute or without a tree, else the
            resolved absolute form.
    """
    if not session_tree or PurePath(hint).is_absolute():
        return hint
    tree = PurePath(session_tree.rstrip("/"))
    base = tree.parent if tree_is_package else tree
    joined = base / hint.lstrip("/")
    return f"{joined}/" if hint.endswith("/") else str(joined)


def _section_source_hint(inp: BriefInputs) -> list[str]:
    """Render Section 7 (local source navigation hint) of the prompt.

    Leads with the tree this session optimises: the root list cannot express it,
    since its order records only how roots were discovered.

    Args:
        inp (BriefInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the source-hint section.
    """
    rows = ["## 7. LOCAL SOURCE NAVIGATION HINT", ""]
    session_tree = (inp.session_framework_tree or "").strip()
    others = tuple(r for r in inp.framework_source_roots if r.rstrip("/") != session_tree.rstrip("/"))
    if not session_tree and not others and not inp.source_hint_directories:
        rows.append(_NONE_PLACEHOLDER)
        return rows
    if session_tree:
        rows.append("The tree this session is optimising — start here:")
        rows.append(_source_root_row(session_tree, worktree_base=inp.worktree_base, facts=inp.facts))
        rows.append("")
    if others:
        rows.append("Other source trees on this host:" if session_tree else "Source trees on this host:")
        for root in others:
            rows.append(_source_root_row(root, worktree_base=inp.worktree_base, facts=inp.facts))
    if inp.source_hint_directories:
        rows.append("")
        rows.append("Where the evidence points — read these first, then widen:")
        for hint in inp.source_hint_directories:
            rows.append(
                f"- {_resolve_focus_dir(hint, session_tree, tree_is_package=inp.facts.session_tree_is_package)}"
            )
    rows.append("")
    rows.append(
        "These are starting points, not a boundary — read anything on the host "
        "that answers the question. Patches still go through ``integrate_patch`` "
        "(Section 9), whichever tree they name."
    )
    rows.append("")
    rows.append(
        "Use ``WebSearch`` to look up the latest upstream version of the local "
        "repo and compare the implementation you intend to modify against what "
        "is there now. Use ``WebFetch`` to read the relevant file or PR "
        "directly — before authoring a patch, confirm whether the upstream "
        "repo already contains the fix or optimization you are about to write."
    )
    return rows


# Section 1b — Enablement playbook (per-task; enablement specialist only)
def _section_enablement_playbook(inp: BriefInputs) -> list[str]:
    """Render the per-task enablement mandate + ladder book into the user prompt.

    Renders the mandate's ``task_description`` (which embeds the ladder book),
    which the body builds from the verdict the dispatch was decided on and the
    dispatch's own evidence and hands over as ``facts.enablement_mandate``.
    Kept in the user prompt so the cached system prompt stays task-independent.

    Args:
        inp: Assembled prompt inputs for the current dispatch.

    Returns:
        list[str]: The enablement-playbook section lines.
    """
    rows = ["## 1b. ENABLEMENT PLAYBOOK", ""]
    rows.extend(inp.facts.enablement_mandate.splitlines())
    acc_cfg = inp.enablement_accepted_config or {}
    acc_envs = {str(k): str(v) for k, v in (acc_cfg.get("extra_envs") or {}).items()}
    acc_args = str(acc_cfg.get("extra_server_args") or "").strip()
    if acc_envs or acc_args:
        rows.append("")
        rows.append("### Config already in effect")
        rows.append(
            "Prior rounds accepted these and your bench launches with them. Re-propose "
            "one only to override or remove it."
        )
        rows.extend(f"- `export {k}={v}`" for k, v in acc_envs.items())
        if acc_args:
            rows.append(f"- server args: `{acc_args}`")
    return rows


# Section 1a — PD-disaggregation
def _section_pd_disaggregation(inp: BriefInputs) -> list[str]:
    """§1a — PD-disaggregation context (omitted unless pd_mode==disaggregated).

    Surfaces the prefill/decode split so the specialist targets each role's
    distinct bottleneck (prefill: compute / TTFT; decode: memory-bandwidth /
    TPOT) and the KV-transfer path, instead of treating the server as one pool.
    Reads the PD topology the body took from multi-node state; returns ``[]``
    on the single-node / aggregated paths so the section is dropped.

    Args:
        inp (BriefInputs): The assembled prompt inputs.

    Returns:
        list[str]: The PD-disaggregation section lines, or ``[]`` when not
        disaggregated.
    """
    pd = inp.facts.pd_topology
    if not pd:
        return []
    tb = pd.get("transfer_backend") or "the KV transfer backend"
    return [
        "## 1a. PD-DISAGGREGATION (prefill/decode separated)",
        "",
        "This deployment runs **prefill/decode disaggregation**: prefill and "
        "decode execute on SEPARATE GPU nodes, exchanging KV cache via "
        f"`{tb}`. Optimize each role for its OWN bottleneck — do NOT treat the "
        "server as a single pool:",
        "",
        f"- **Prefill** ({pd.get('prefill_nodes')} node(s), tp={pd.get('prefill_tp')}, "
        f"ep={pd.get('prefill_ep')}): compute-bound; drives **TTFT**. Levers: "
        "chunked-prefill size, attention backend for long ISL, MoE dispatch, "
        "prefill token/batch budget.",
        f"- **Decode** ({pd.get('decode_nodes')} node(s), tp={pd.get('decode_tp')}, "
        f"ep={pd.get('decode_ep')}): memory-bandwidth-bound; drives **TPOT/ITL**. "
        "Levers: KV-cache / mem-fraction, max-running-requests, dp-attention, "
        "decode MoE a2a backend.",
        f"- **KV transfer** (`{tb}`): watch bootstrap / transfer stalls; RDMA/IB "
        "device selection affects decode start latency.",
        "- **Balance**: tune the prefill:decode node/TP ratio to the workload shape "
        + "— a saturated role caps end-to-end throughput.",
        "",
        "Per-role GPU telemetry is in the benchmark report's "
        "`gpu_monitor_by_role` (prefill vs decode util / power / VRAM); use it to "
        "confirm which role is the bottleneck BEFORE proposing changes.",
    ]


# Top-level assembler

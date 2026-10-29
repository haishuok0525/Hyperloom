# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialist sub-agent prompt assembler.

Returns ``(system_prompt, user_prompt)``: the system prompt carries the
immutable contract (identity / output protocol / iron rules) so the
backend can cache it across specialists; the user prompt carries per-task
context (hardware / gap / KB / recipe / PR / source hint). Most sections
render a ``(none)`` placeholder when empty; the execution-budget and
PD-disaggregation sections are omitted entirely. Pure: what the body knows
(multi-node state, the enablement mandate, its registries and rules) arrives
as ``inp.facts``.
"""

from __future__ import annotations

import json

from hyperloom.common.coerce import first_text
from hyperloom.common.prompt_safety import defang_prompt_structure

from .inputs import MODE_PATCH, BriefInputs
from .context import (
    _section_enablement_playbook,
    _section_execution_budget,
    _section_experience_kb,
    _section_gap,
    _section_hardware,
    _section_kb_subgraph,
    _section_lessons,
    _section_pd_disaggregation,
    _section_pitfalls,
    _section_pr_feed,
    _section_recipe,
    _section_roofline_evidence,
    _section_runtime_findings,
    _section_source_hint,
)
from .focus import _DOMAIN_FOCUS_TEMPLATES


# Per-task-kind brief appended by _section_mandate; only the per-dispatch dynamic
# context (PR lead, critic feedback, apply errors, residual questions) lives in
# ``inp.notes``. Boilerplate that is the same for every dispatch of a given kind
# lives here so it is maintained in one place.
_TASK_KIND_BRIEFS: dict[str, str] = {
    "framework_authoring": (
        "Read the upstream PR as inspiration, then deliver the best win for this"
        " model / hardware / workload — go beyond the diff where the live source justifies it."
        " Deliverable: a unified-diff patch in your worktree (``patches_written``)"
        " OR a config/env ``proposal_set`` entry when the win is a flag toggle."
        " The Coordinator applies and benches it; you do not benchmark."
    ),
    "framework_local_explore": (
        "No upstream PR was found. Author the best throughput win directly from"
        " the live source + profiling evidence. Read the source trees in Section 7"
        " and the roofline (Section 4a) to locate the hot path."
        " You MAY use WebSearch / WebFetch to compare the local checkout against"
        " the latest upstream code and port a newer optimisation when behind."
        " Deliverable: a unified-diff patch (``patches_written``)"
        " OR a config/env ``proposal_set`` entry. You do not benchmark."
    ),
    "explore_apply_retry": (
        "A previous patch failed to apply against the live source tree."
        " Study the apply errors in the notes, produce a corrected patch."
    ),
}


# Section 1 — Identity & autonomy
def _authors_patches(inp: BriefInputs) -> bool:
    """Whether this dispatch may author patches; research mode gets no worktree."""
    return inp.mode == MODE_PATCH


#: Throughput domains whose focus block offers flags first, so the coda has
#: something to raise the sights from. ``enablement_specialist`` is absent
#: because its ladder book says climb only as far as runnability needs;
#: ``framework_rewrite_specialist`` because it authors source rewrites and
#: nothing else, and already carries its own "Where the wins are" taxonomy;
#: ``speculative_decoding_specialist`` because its focus names no flags at all.
_THROUGHPUT_AUTHORING_DOMAINS = frozenset(
    {
        "serving_specialist",
        "kernel_switch_specialist",
        "comm_specialist",
        "compiler_specialist",
        "system_specialist",
    }
)


def _source_patch_altitude_block(inp: BriefInputs) -> list[str]:
    """Where the durable wins are, for a throughput dispatch that may author code."""
    if not (_authors_patches(inp) and inp.domain.key in _THROUGHPUT_AUTHORING_DOMAINS):
        return []
    return [
        "",
        "### Altitude — where the durable wins are",
        "",
        "The flags above are cheap to validate and their ceiling is fixed by",
        "whoever wired them up; a source change moves that ceiling and keeps",
        "paying in later rounds. When the cheap levers do not close the gap, stop",
        "permuting them and read the source for:",
        "- **An un-bridged capability switch** — a `*_supported()` / `*_enabled()`",
        "  / `is_*()` predicate that takes this model / GPU / precision off a",
        "  faster path. Bridging it creates a lever nobody had.",
        "- **A faster path already present** — reached never, or with arguments",
        "  that disqualify it.",
        "- **A real bug** — a wrong bound, a stale cache key, a fallback that",
        "  fires when it should not. Fixing it is a KEEP like any other.",
        "- **Component usage this architecture argues against** — the order,",
        "  granularity or fusion of the pieces, not their settings.",
        "Deliver a unified diff in your worktree `patches/` dir, small (target ≤5",
        "files) and grounded: a patch you cannot point at a profile line or a",
        "source predicate is budget spent for nothing.",
    ]


def _domain_focus_blocks(inp: BriefInputs) -> list[str]:
    """Render the focus block of the active domain, then of each extra tag on a multi-tag dispatch, closing on the altitude coda that spans them."""
    body: list[str] = []
    rendered: set[str] = set()
    for key in (inp.domain.key, *inp.facts.focus_tag_domains):
        focus = _DOMAIN_FOCUS_TEMPLATES.get(key) if key and key not in rendered else None
        lines = focus(inp) if focus is not None else []
        if not lines:
            continue
        body.extend(["", f"### Domain focus — {key}", "", *lines])
        rendered.add(key)
    body.extend(_source_patch_altitude_block(inp))
    return body


def _section_identity(inp: BriefInputs) -> list[str]:
    """Render Section 1 (identity & autonomy) of the specialist prompt.

    Args:
        inp (BriefInputs): The assembled prompt inputs.

    Returns:
        list[str]: Markdown lines for the identity section.
    """
    if _authors_patches(inp):
        capability_line = (
            "probe the host via Bash, **author source patches into your isolated"
            " worktree**, and use as many of your ``max_turns`` LLM turns as you need"
        )
        deliverable_line = "(Section 8) carrying ``proposal_set`` + ``patches_written``. The hard"
    else:
        capability_line = "probe the host via Bash, and use as many of your ``max_turns`` LLM turns as you need"
        deliverable_line = "(Section 8) carrying ``proposal_set``. The hard"
    if inp.allocated_gpu_ids:
        leaf_examples = "bench N candidates of one lever at once, or read several subsystems"
        leaf_devices = "inherit your VISIBLE_DEVICES (so they share your GPU and cannot oversubscribe), and "
    else:
        leaf_examples = "read several subsystems at once"
        leaf_devices = ""
    fanout_line = (
        f"Fan-out: to parallelize independent single-shot sub-tasks (e.g. {leaf_examples}), "
        'you MAY ``Task(subagent_type="hyperloom-leaf")``. Leaves are single-turn, '
        f"{leaf_devices}cannot fan out further. Use leaves for breadth; do multi-round "
        "depth (e.g. coordinate-descent autotune) yourself."
    )
    body: list[str] = [
        "## 1. IDENTITY & AUTONOMY",
        "",
        f"You are a fully autonomous **{inp.domain.key}** dispatched by the",
        f"Hyperloom Coordinator. Layer: {inp.domain.layer}.",
        f"KB anchor: {inp.domain.kb_anchor}.",
        "",
        f"Description: {inp.domain.description or '(generic)'}",
        "",
        "You operate **autonomously** inside your domain — no per-step approval",
        "is needed. You have full authority to read any code on this host — the",
        "trees in Section 7 are where to start, not where to stop — search any",
        "public GitHub repo or NVIDIA PR,",
        capability_line,
        "to be thorough. Be creative. Investigate deeply. One-turn shortcuts",
        "are discouraged when a real bottleneck is on the table — but stop once",
        "rounds stop yielding new findings; the wall clock is not the only stop",
        "signal. Quality over quantity: **2 proposals is the norm, 4 the hard",
        "cap**. One real beats two padded; an empty ``proposal_set`` with a",
        "clear ``summary`` beats one padded.",
        "",
        "Division of labour: the Coordinator owns the serving GPU, runs the E2E",
        "benchmark, and decides KEEP/REVERT — you do not have to validate final",
        "throughput yourself. Your single deliverable is ONE final ``specialist_done``",
        deliverable_line,
        "capability boundary is fixed by Section 9 Iron Rules; everything inside",
        "it is yours.",
        "",
        fanout_line,
    ]
    body.extend(_domain_focus_blocks(inp))
    if inp.scope == "domains":
        body.extend(_cross_domain_block(inp))
    elif inp.scope == "freeform":
        body.extend(_freeform_block(inp))
    if inp.allocated_gpu_ids:
        body.extend(_gpu_autonomy_block(inp))
    body.extend(_cpu_selfcheck_block(inp))
    if inp.auto_retry_reason.strip():
        body.extend(_auto_retry_note_block(inp))
    return body


def _auto_retry_note_block(inp: BriefInputs) -> list[str]:
    """Heads-up block when this dispatch is a bounded auto-retry of a prior
    transient (timeout / crash / stale-heartbeat) attempt. Advisory only —
    the mandate is unchanged; the note nudges the specialist to scope its work
    so it finishes within budget this time.

    Args:
        inp: The specialist prompt inputs (reads ``auto_retry_reason``).

    Returns:
        The rendered auto-retry notice lines.
    """
    reason = inp.auto_retry_reason.strip()
    return [
        "",
        "### Auto-retry notice",
        "",
        "Your previous attempt on this task did NOT finish cleanly — the "
        f"Coordinator is re-dispatching you. Reason: ``{reason}``.",
        "This is a transient infrastructure failure (a timeout, crash, or "
        + "silent hang), not a rejection of the approach. Scope your "
        + "investigation so you reach a single ``specialist_done`` within "
        + "``max_turns`` this time: prefer fewer, higher-confidence probes, "
        + "emit heartbeats, and avoid long-running shell that risks the same "
        + "timeout.",
    ]


def _gpu_autonomy_block(inp: BriefInputs) -> list[str]:
    """On-GPU autonomy block appended for GPU specialists (those with a card
    allocation). Frames the broad capabilities the specialist has on its own
    leased cards and surfaces the *optional* ``rebench`` helper — none of it is
    a mandate; the Coordinator's ``integrate_patch`` E2E gate stays the single
    authoritative measure of truth.

    Args:
        inp: The specialist prompt inputs.

    Returns:
        The rendered on-GPU autonomy lines.
    """
    cards = ", ".join(str(g) for g in inp.allocated_gpu_ids)
    return [
        "",
        "### On-GPU autonomy (your leased cards)",
        "",
        f"You exclusively own GPU card(s) [{cards}] for this task. On those "
        + "cards you are free to do whatever converges on a benched win:",
        "- For kernel/config autotune, search the installed framework/source "
        + "first for maintained benchmark/tuning entrypoints, config lookup "
        + "paths, and nearby config families; prefer those.",
        "- If the built-in path is missing or incomplete, write a small "
        + "source-derived harness around the framework primitive/config override "
        + "API. Use warmups, true-default/current/candidate baselines, "
        + "median/min-of-reps, and an accuracy guard.",
        "- Write and run arbitrary scripts — autotune harnesses, "
        + "microbenchmarks, profilers (rocprof / torch.profiler / your own "
        + "breakdown).",
        "- Start / restart a real server on your own cards and benchmark it " + "however you see fit.",
        "- Profile freely to get a fresh trace after a change — don't rely only "
        + "on the static roofline snapshot you were handed.",
        "- Tune the framework's config-file levers (e.g. MoE/GEMM/attention "
        + "Triton config JSONs) — a missing/untuned config is often the single "
        + "biggest lever.",
        "- Self-check accuracy (advisory ``max_abs_err`` / gsm8k) when you want "
        + "to — the Coordinator gate stays authoritative, so this is guidance, "
        + "not a requirement.",
        "",
        "Optional helper: a ``rebench`` convenience reuses the real Magpie "
        + "serving + benchmark path on your leased cards, so you can get numbers "
        + "directly comparable to the ``integrate_patch`` gate in one call:",
        "    python -m hyperloom.orchestrator.specialists.rebench \\",
        "        --config <magpie.yaml> --output ./scratch/rebench " + "[--extra-args '<server args>']",
        "  It prints a JSON result with ``output_throughput``. It is OPTIONAL "
        + "— you may instead write your own bench/autotune script. Throughput "
        + "does NOT have to come from rebench.",
    ]


def _cpu_selfcheck_block(inp: BriefInputs) -> list[str]:
    """Optional ``selfcheck`` helper for a patch specialist with a worktree and no GPU."""
    if inp.allocated_gpu_ids or inp.mode != MODE_PATCH or not inp.worktree_package_dir:
        return []
    package = inp.facts.python_packages[inp.framework.strip().lower()]
    if package is None:
        return []
    return [
        "",
        "Optional helper: ``selfcheck`` installs your worktree's package into a private venv,",
        "imports it, byte-compiles the files you changed and runs any pytest targets you pass:",
        "    python -m hyperloom.orchestrator.specialists.selfcheck \\",
        f"        --worktree {inp.workspace_path} --package-dir {inp.worktree_package_dir} "
        f"--package {package} [--pytest <target>]",
        "  It prints a JSON result. A framework with compiled extensions (e.g. vLLM) rebuilds",
        "  them on install, which can take well over your wall budget.",
    ]


def _freeform_block(inp: BriefInputs) -> list[str]:
    """Free-form mandate appended when ``scope == 'freeform'``. The
    specialist is NOT bound
    to the domain catalogue — the Orchestration ``task_description`` is the
    whole mandate. The single deliverable is still ONE ``specialist_done``.

    Args:
        inp: The specialist prompt inputs (reads ``task_description``).

    Returns:
        The rendered free-form mandate lines.
    """
    desc = (inp.task_description or "").strip() or "(no task description provided)"
    if _authors_patches(inp):
        reach = "upstream PRs, host probing, source patches"
        deliverable = "``proposal_set`` + ``patches_written``"
    else:
        reach = "upstream PRs, host probing"
        deliverable = "``proposal_set``"
    return [
        "",
        "### Free-form mandate (scope = freeform)",
        "",
        "You are dispatched as a **free-form** specialist: you are NOT bound to "
        + "the domain catalogue above. The Orchestration mandate below is your "
        + f"whole task — investigate it wherever it leads (framework internals, {reach}).",
        "",
        "Mandate from Orchestration:",
        "",
        f"> {desc}",
        "",
        "Set ``scope='freeform'`` on each proposal. Your single deliverable is "
        + f"still ONE ``specialist_done`` carrying {deliverable}. Never "
        + "self-report numeric speedups — the Coordinator measures gain.",
    ]


def _cross_domain_block(inp: BriefInputs) -> list[str]:
    """Cross-domain mandate appended when ``scope == 'domains'``. The
    single deliverable is still
    ONE ``specialist_done``; the difference is the patch may span every domain
    in scope and the Critic will hold it to the cross-domain rules.

    Args:
        inp: The specialist prompt inputs (reads ``extra_focus_tags`` /
            ``domain``).

    Returns:
        The rendered cross-domain mandate lines.
    """
    tags = ", ".join(inp.extra_focus_tags) if inp.extra_focus_tags else inp.domain.key
    return [
        "",
        "### Cross-domain mandate (scope = domains)",
        "",
        f"You are dispatched as a **cross-domain** specialist over: {tags}.",
        "You may author a single coherent patch that spans these domains "
        + "together when (and only when) the change must happen jointly — a "
        + "combination no single-domain specialist could surface from within "
        + "its own boundary.",
        "",
        "In your ``specialist_done`` you MUST justify the combination:",
        "- give an independent rationale for the change **within each domain** " + "in scope;",
        "- name the **coupling points** (why these changes must land together) "
        + "and at least one **side effect** of the combination;",
        "- show this is genuine cross-domain synthesis, not a concatenation of "
        + "two independent single-domain edits (that is an explore grid combo, "
        + "not a cross-domain patch).",
        "Set ``scope='domains'`` on the proposal so the Critic attaches the "
        + "cross-domain review rules. Never self-report numeric speedups — the "
        + "Coordinator measures gain.",
    ]


# Section 0 — Mandate (deliverable contract + run status)
def _section_mandate(inp: BriefInputs) -> list[str]:
    """Render §0: deliverable contract, run status, and task-kind brief."""
    # Deliverable line based on scope × mode.
    if inp.scope == "freeform":
        anchor = inp.task_description.split("\n")[0].strip()[:120]
        deliverable = "freeform investigation — see task description"
    else:
        anchor = inp.gap_canonical_id
        if not _authors_patches(inp):
            deliverable = "findings and up to 6 ranked config variants (read-only; no patch)"
        elif inp.scope == "domains":
            deliverable = "a coupled patch spanning multiple domains + up to 6 ranked config variants"
        else:
            deliverable = "a source patch and/or up to 6 ranked config variants addressing the gap below"

    rows: list[str] = [
        "## 0. MANDATE",
        "",
        f"- deliverable: {deliverable}",
    ]
    if anchor:
        rows.append(f"- anchor: `{anchor}`")
    rows.extend(_mandate_run_status(inp))

    rows.extend(
        [
            "",
            "Judged by: the Coordinator benches your proposals end-to-end against",
            "the sealed baseline and decides KEEP/REVERT; the accuracy gate runs",
            "alongside. You are not asked to prove the number.",
        ]
    )

    brief = _TASK_KIND_BRIEFS.get(inp.task_kind.strip(), "")
    if brief:
        rows.extend(["", brief])

    if inp.pr_lead:
        title = defang_prompt_structure(first_text(inp.pr_lead.get("title")))
        url = first_text(inp.pr_lead.get("url"))
        diff_url = first_text(inp.pr_lead.get("diff_url"))
        rows.append("")
        if title:
            rows.append(f"PR lead: {title}")
        if url:
            rows.append(f"PR url: {url}")
        if diff_url:
            rows.append(f"Diff: {diff_url} (fetch with WebFetch)")

    if inp.prior_attempts:
        rows.extend(
            [
                "",
                "Already tried this session — avoid the same or equivalent change:",
            ]
        )
        for att in inp.prior_attempts[:20]:
            ref = first_text(att.get("ref"))
            if not ref:
                continue
            status = first_text(att.get("status"), "?")
            gain = att.get("gain_pct")
            gain_str = f" gain={float(gain):+.2f}%" if isinstance(gain, (int, float)) else ""
            why = first_text(att.get("why"))
            why_str = f" — {why}" if why else ""
            rows.append(f"  - {ref} [{status}]{gain_str}{why_str}")

    return rows


def _mandate_run_status(inp: BriefInputs) -> list[str]:
    """Render §0's read-only run status; empty until a baseline or a KEEP threshold exists."""
    if not (inp.baseline_tput > 0 or inp.keep_threshold_pct > 0):
        return []
    rows = ["", "Run status (read-only context; do NOT re-state these as your own measurements):"]
    if inp.baseline_tput > 0:
        rows.append(f"- baseline: {inp.facts.baseline_metric}")
    if inp.current_tput > 0:
        rows.append(f"- current best: {inp.facts.current_metric}")
    if inp.cumulative_gain_validated != 0:
        rows.append(f"- validated cumulative gain: {inp.cumulative_gain_validated:+.2f}%")
    if inp.keep_threshold_pct > 0:
        rows.append(f"- KEEP threshold this cycle: {inp.keep_threshold_pct:.2f}%")
    if inp.applied_stack:
        stack_items = ", ".join(
            f"{e.get('variant_name', '?')} ({e.get('gain_pct', 0):+.2f}%)" for e in inp.applied_stack[:6]
        )
        rows.append(f"- applied stack: {stack_items}")
    return rows


# Section 8 — Output protocol
def _section_output_protocol(inp: BriefInputs) -> list[str]:
    """Render Section 8 (output protocol) of the specialist prompt."""
    workspace = inp.workspace_path or "<workspace>"
    authors_patches = _authors_patches(inp)
    kind = "upstream_candidate" if inp.domain.key == "candidate_discovery_specialist" else "explore_variant"
    schema = inp.facts.proposal_schemas[kind]
    proposal_example, proposal_contract = schema["example"], tuple(schema["contract"])

    exit_lines = [
        "**Exit — file write:** write the ``specialist_done`` payload (schema below) to",
        f"``{workspace}/specialist_done.json`` as your **absolute last action**.",
        "The dispatcher polls for that file as the exit signal; stop after writing.",
    ]

    if authors_patches:
        patch_fields = [
            "- ``deliverable`` declares what the round produced and is",
            "  REQUIRED whenever you changed anything: ``{tree_id, targets,",
            "  patches, artifacts, envs, server_args, setup_commands}``.",
            "  ``targets`` lists every file you edited, relative to your",
            "  worktree — the harvest is scoped to it, so a file you changed",
            "  and did not declare is not shipped. Do NOT put hashes in it;",
            "  the harness computes them where your work was validated.",
            "- ``patches_written`` lists paths (relative to your",
            "  workspace or worktree) of any unified-diff patch files you",
            "  authored this round. Empty list = no patches; downstream",
            "  ``integrate_patch`` action skips when empty.",
            "- ``artifacts_written`` lists any non-diff tuned artifacts to install",
            "  (e.g. an autotuned config JSON) as objects ``{source, target, kind,",
            "  description}``: ``source`` is a path inside your worktree, ``target``",
            "  is the install path — PREFER a framework-relative path; an absolute",
            "  path is accepted only if it resolves inside an allowlisted framework",
            "  root. ``integrate_patch`` backs up the target, installs the artifact,",
            "  runs the same E2E gate, and restores the backup on REVERT. A non-diff",
            "  tuned artifact is a FULL result — keep ``proposal_set`` non-empty or",
            "  list the artifact in ``artifacts_written``.",
            "- ``resolves_finding`` (optional string ``rule_id:subject``): set it only",
            f"  when the patch fixes a ``{'`` / ``'.join(sorted(inp.facts.correctness_fix_rules))}`` entry from section 4c,",
            "  copied verbatim. The gate then verifies the entry is gone from the new log and",
            f"  accuracy passes, and may KEEP a drop of up to {inp.facts.correctness_fix_max_drop_pct:.0f}% on the graded axis",
            "  (throughput, or p50 interactivity with the other interactivity guards still held)",
            "  within the latency budget.",
        ]
        no_output = "  AND no ``patches_written``/``artifacts_written``; in that case"
    else:
        patch_fields = []
        no_output = "  and no findings; in that case"
    no_proposal_rule = [
        "- An empty ``proposal_set`` is legitimate ONLY when you have no actionable proposals",
        no_output,
        "  and you must put the reason in ``summary``.",
    ]

    return [
        "## 8. OUTPUT PROTOCOL",
        "",
        *exit_lines,
        "",
        "**Messages from the Orchestrator (check this as you work):** read",
        f"``{workspace}/inbox.json`` whenever you finish a step. It is a JSON",
        "list of ``{from, ts, body}`` entries, absent until the Orchestrator",
        "sends one. It is how the Orchestrator answers a question you raised",
        "or redirects you mid-run — if it tells you the mandate changed,",
        "follow it rather than finishing the original plan. Never write to",
        "this file.",
        "",
        "**Incremental checkpoint (do this throughout the run):** every time",
        "you reach a new finding or finish a candidate, rewrite your",
        "best-so-far payload to",
        f"``{workspace}/specialist_done.partial.json`` (write to",
        f"``{workspace}/specialist_done.partial.json.tmp`` first, then rename",
        "over the partial so a reader never sees a half-written file). This",
        "partial uses the **same payload schema** as the final file but does",
        "**NOT** end the run — keep working. There is a wall-clock budget; if",
        "you are stopped before finishing, whatever is in the partial is",
        "preserved as your result, so keep it current. The Orchestrator also",
        "reads each rewrite while you are still running — it is how you report",
        "direction and raise ``residual_questions`` early enough to get an",
        "answer back through ``inbox.json``. Write the final",
        "``specialist_done.json`` (which ends the run) only once, as your",
        "absolute last action.",
        "",
        "Payload schema (identical for both channels):",
        "",
        "```json",
        json.dumps(
            {
                "intent_type": "specialist_done",
                "payload": {
                    "gap_canonical_id": inp.gap_canonical_id or "<echo from dispatch>",
                    "domain": inp.domain.key,
                    "proposal_set": [proposal_example],
                    **({"patches_written": [], "experience_citations": []} if authors_patches else {}),
                    "summary": "≤ 500 char overview of what you tried this round",
                    "confidence": 0.6,
                    "new_findings": [],
                    "residual_questions": [],
                },
            },
            sort_keys=True,
            indent=2,
        ),
        "```",
        "",
        "Field contract:",
        "",
        *proposal_contract,
        (
            "- ``proposal_set``: **2 entries is the norm, 4 the hard cap.** You "
            "are a curator, not a brainstormer: rank by expected gain x "
            "confidence, drop anything contradicting ``kb_subgraph`` / "
            "``pr_evidence``, and stop at 2. A 3rd or 4th must beat the median "
            "of the first two. Padding is a failure, not thoroughness: each "
            "weak entry costs a Critic reject and a slot on the serial "
            "benchmark queue. One real proposal is a better round than two "
            "padded ones, and an empty ``proposal_set`` with a clear ``summary`` "
            "is better than one."
        ),
        (
            "- The Critic reviews each surviving variant against the KB "
            + "before benchmarking, so a marginal-quality proposal costs you "
            + "a reject (and a pitfall fact that will warn future sessions "
            + "off the same dead-end)."
        ),
        *patch_fields,
        *no_proposal_rule,
        "- ``new_findings`` is a list of learned items. Research scouts must",
        "  emit source-backed ``{what, source, expected_impact, accuracy_risk,",
        "  domain_tags[]}`` records.",
        "- ``residual_questions`` carries to the next specialist round.",
        "",
        "**Heartbeat (Channel B only):** When running in subprocess mode,",
        f"write ``{workspace}/heartbeat.json`` periodically (≤5 min apart)",
        "via Bash so the dispatcher knows you are still alive. Format:",
        '``{"ts": "<iso8601>", "status": "running", "note": "<short>"}``.',
        "Going silent past 5 minutes kills your subprocess.",
        "",
        (f"Hard cap: at most **{inp.max_turns}** LLM turns. Emit specialist_done before the cap."),
    ]


# Section 9 — Iron rules
def _section_iron_rules(inp: BriefInputs) -> list[str]:
    """Render Section 9 (iron rules) of the specialist prompt."""
    workspace = inp.workspace_path or "<runs/specialist/<task_id>/>"
    authors_patches = _authors_patches(inp)
    if inp.allocated_gpu_ids:
        cards = ", ".join(str(g) for g in inp.allocated_gpu_ids)
        gpu_rule = [
            f"1. You EXCLUSIVELY own GPU card(s) [{cards}] for this task. On",
            "   those cards do whatever you want: edit code, build, start/stop",
            "   your own servers, profile, autotune, install tuned artifacts,",
            "   and run real benchmark loops. The ONE thing you must NOT do:",
            "   touch the production serving process or its cards — co-residing",
            "   on them would corrupt both your measurement and production.",
            "   Manage only processes YOU started, by their own PID/PGID.",
        ]
    else:
        last = "   try and optionally author patches." if authors_patches else "   try."
        gpu_rule = [
            "1. You have no GPU allocation for this task, so do not run GPU",
            "   benchmarks or start servers. The ONE hard boundary that always",
            "   holds: never touch the production serving process / its cards /",
            "   port 8888. The Coordinator runs benchmarks; you propose what to",
            last,
        ]
    if authors_patches:
        integration_rule = [
            "2. **You MAY** produce changes for integration, but stage them ONLY",
            f"   inside your own worktree at ``{workspace}/``. Two output kinds:",
            "   - Unified-diff patches: ``git diff > patches/NNN_<slug>.patch``",
            "     from the worktree; list paths in ``patches_written``.",
            "   - Tuned non-diff artifacts (e.g. an autotuned config JSON): write",
            "     under the worktree and list in ``artifacts_written`` as",
            "     ``{source, target, kind, description}``.",
            "   **NEVER** ``git apply`` / ``git commit`` against the shared source",
            "   trees in Section 7 directly — ``integrate_patch`` is the single",
            "   integration point, whichever tree the patch names.",
        ]
    else:
        integration_rule = [
            "2. **Read-only dispatch:** you have no worktree and MUST NOT author",
            "   patches or edit the source trees. Report what you found",
            "   through ``specialist_done``; a patch-capable specialist authors any",
            "   source change you recommend.",
        ]
    return [
        "## 9. IRON RULES (Inv-5.1 / Inv-5.3)",
        "",
        *gpu_rule,
        *integration_rule,
        "3. Only ``specialist_done``, ``send_message``, and ``alert`` are",
        "   accepted intents; all others are dropped.",
        "4. You **MUST** finish within ``max_turns`` LLM turns and end with",
        "   exactly one ``specialist_done`` exit signal. Silence past the cap",
        "   synthesizes an empty done.",
        f"5. Use ``{workspace}/`` for ALL writes. It and ``SESSION_DIR`` are the",
        "   only directories the dispatcher hands you to write; the source trees",
        "   are yours to read.",
        "6. On tool error or no useful action left, emit",
        "   ``specialist_done{proposal_set=[], summary='<why>'}``.",
        f"7. {inp.facts.bash_kill_safety}",
    ]


def build_specialist_prompts(inp: BriefInputs) -> tuple[str, str]:
    """Assemble the full specialist prompt from its section builders.

    The system prompt carries the immutable contract (identity, output
    protocol, iron rules) and the user prompt carries the per-task context
    (hardware, optional PD-disaggregation, execution budget, gap, KB,
    roofline, recipe, lessons, pitfalls, PR feed, source hint and
    orchestration notes). The split lets the LLM backend cache the system
    prompt across specialists.

    Args:
        inp (BriefInputs): The assembled prompt inputs.

    Returns:
        tuple[str, str]: The ``(system_prompt, user_prompt)`` pair.
    """

    system_sections = [
        _section_identity(inp),
        _section_output_protocol(inp),
        _section_iron_rules(inp),
    ]
    if inp.domain.key == "enablement_specialist":
        # Pre-baseline enablement: the perf context (roofline / recipe / lessons /
        # pitfalls / KB subgraph) is noise when the server cannot boot or the
        # baseline fails its accuracy eval. Carry only the failure, the tiered
        # playbook, and the tools to discover + navigate a fix.
        user_sections = [
            _section_mandate(inp),
            _section_hardware(inp),
            _section_pd_disaggregation(inp),  # § 1a (omitted unless disaggregated)
            _section_execution_budget(inp),  # omitted when no budget
            _section_gap(inp),
            _section_enablement_playbook(inp),  # § 1b (mandate + ladder book)
            _section_pr_feed(inp),
            _section_source_hint(inp),
        ]
    elif inp.domain.key == "custom_specialist":
        # The operator focus is the mandate; perf context is left to the operator text.
        # Experience KB stays: the warmup records its block as shown to every FRAMEWORK_AGENT specialist.
        user_sections = [
            _section_mandate(inp),
            _section_hardware(inp),
            _section_pd_disaggregation(inp),
            _section_execution_budget(inp),
            _section_gap(inp),
            _section_experience_kb(inp),
        ]
        if _authors_patches(inp):
            user_sections.append(_section_source_hint(inp))
    else:
        user_sections = [
            _section_mandate(inp),
            _section_hardware(inp),
            _section_pd_disaggregation(inp),  # § 1a (PD-disaggregation; omitted unless disaggregated)
            _section_execution_budget(inp),  # omitted when no budget
            _section_gap(inp),
            _section_kb_subgraph(inp),
            _section_roofline_evidence(inp),
            _section_experience_kb(inp),
            _section_runtime_findings(inp),
            _section_recipe(inp),
            _section_lessons(inp),
            _section_pitfalls(inp),
            _section_pr_feed(inp),
            _section_source_hint(inp),
        ]
    if inp.notes:
        user_sections.append(
            [
                "## 10. NOTES FROM ORCHESTRATION",
                "",
                defang_prompt_structure(inp.notes),
            ]
        )

    def _flatten(sections: list[list[str]]) -> str:
        """Join section line-lists into a single newline-separated string.

        Inserts one blank line between non-empty sections and appends a
        trailing newline.

        Args:
            sections (list[list[str]]): The per-section lists of lines.

        Returns:
            str: The flattened prompt text.
        """
        out: list[str] = []
        for sec in sections:
            if not sec:  # skip omitted sections (e.g. no execution budget)
                continue
            if out:
                out.append("")
            out.extend(sec)
        return "\n".join(out) + "\n"

    return _flatten(system_sections), _flatten(user_sections)


__all__ = [
    "build_specialist_prompts",
]

"""Stage B: mine recurring success patterns from the Stage A summary buffer.

Reads ``summary_buffer.jsonl``, splits into chunks, asks an LLM to propose
candidate experiences from each chunk, then merges and applies the support
threshold ``K`` to produce the experience library plus its catalog index.

See ``docs/cross_task_experience_replay_v2.md`` §3–§4 for the contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from evaluate_agentic_memory_task import OpenAICompatibleEngine

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


CANDIDATE_REQUIRED_FIELDS = (
    "proposed_id",
    "title",
    "applicable_context",
    "action_guidance",
    "supporting_trajectories",
)
CONTEXT_REQUIRED_FIELDS = ("when", "ui_signals", "domain_hint")


V1_SYSTEM_PROMPT = """You are a pattern extractor for a GUI-agent experience
library. You are given a batch of causal summaries distilled from successful
trajectories. Your job is to propose reusable *experience* entries that
capture UI interaction heuristics — patterns that prevent concrete agent
mistakes.

Output strict JSON only (no markdown fences) with this schema:

{
  "candidates": [
    {
      "proposed_id": "short_snake_case_slug",
      "title": "<= 10 word title describing the UI pattern",
      "applicable_context": {
        "when": "one sentence describing the triggering UI situation",
        "ui_signals": ["concrete visual or structural cues"],
        "domain_hint": "generic | travel | shopping | info | entertainment | service"
      },
      "action_guidance": "one short paragraph explaining the heuristic AND the failure mode it prevents",
      "action_templates": ["<= 2 short templates, optional, empty list if none"],
      "supporting_trajectories": ["T012", "T037", "T089"],
      "confidence": 0.0
    }
  ]
}

WHAT COUNTS AS AN EXPERIENCE (accept these):

  Each accepted pattern is paired with a plausible agent MISTAKE it
  prevents. If you cannot name such a mistake, the pattern is not an
  experience.

  1. "When a modal/dialog is visible, keep actions inside the modal until
     it closes" — prevents: agent clicks on the background page and the
     modal silently stays open.
  2. "After changing a filter, date, picker, or passenger value, look for
     an explicit Apply / Done / Update / Search / Continue action before
     moving on" — prevents: agent assumes the value auto-committed and
     advances with stale state.

     IMPORTANT causal direction: the trigger is a filter/value CHANGE
     that has just happened; the rule is to find the CONFIRMATION action
     next. Do NOT phrase this as "apply filters before searching" — that
     reverses the direction and collapses into a task skeleton.
  3. "Prefer a semantic clickable ancestor (button, link) over a
     decorative child (svg, icon, span)" — prevents: click lands on a
     non-interactive node and nothing happens.
  4. "When a subflow is active, continue within it before jumping back to
     top-level navigation" — prevents: branch drift that breaks progress.
  5. "When multiple inputs must be filled, complete all required fields
     before triggering the final submit/search" — prevents: submit fires
     with incomplete state and the site shows a validation error the
     agent misreads.

WHAT DOES NOT COUNT (reject these, even if they appear often):

  - "Search and select an item" / "Filter and apply criteria" /
    "Confirm and proceed" / "Add to cart" / "Navigate to section" /
    "Initiate search" / "Select options from menu"
  - These are TASK SKELETONS. They describe WHAT the task does, not HOW
    to interact with the UI correctly. A model that knows "search then
    select" still fails because it doesn't know *which specific
    interaction pattern* to use at each step.

  - "Apply filters before searching" / "Apply filters to narrow results" /
    "Set criteria before submitting"
  - These invert the correct causal direction and turn the real pattern
    (confirmation AFTER a value change) into a task-skeleton prescription.
    If the real signal in the summaries is filter/picker confirmation,
    phrase it as accept example #2.

HARD TEST — apply to every candidate before emitting:

  Ask: "What concrete mistake would a plausible agent make if it did NOT
  know this rule?"

  - If the answer is "the task just wouldn't happen" (e.g. "you can't
    search without typing into the search box, duh") — REJECT.
  - If the answer is a specific, nameable failure mode (clicked wrong
    node, skipped confirmation, drifted branch, missed required field) —
    ACCEPT.

Other rules:

- A candidate MUST be supported by at least 2 trajectories in this
  chunk. Downstream code applies a stricter K=3 threshold after
  cross-chunk consolidation.
- supporting_trajectories MUST use exact T### handles from input. Do not
  hallucinate.
- Prefer evidence from `critical_turning_points` and skills marked
  `decisive`. Skills marked `redundant` are NOT evidence.
- action_templates capped at 2 entries. Templates are action shapes,
  not coordinates.
- Titles must be short, generic, NO site names, NO task-skeleton phrases.
- If no genuinely transferable UI heuristic exists in this batch, return
  {"candidates": []}. Empty output is better than task-skeleton output.
"""


V1_CONSOLIDATOR_SYSTEM_PROMPT = """You are consolidating candidate experience
entries that were extracted independently from different chunks of a
trajectory summary buffer. Many candidates describe the SAME underlying
UI interaction pattern with different wording.

Your job:
  1. Group candidates that describe the same underlying pattern.
  2. For each group, emit ONE canonical entry with unioned
     supporting_trajectories and the best-phrased title/guidance.
  3. DROP any group whose final pattern is a task skeleton (e.g.
     "search and select", "filter and apply", "confirm and proceed",
     "add to cart", "navigate to section"). These are task
     descriptions, not UI interaction heuristics.

     Also DROP patterns that invert the causal direction of a real
     heuristic. Specifically: "apply filters before searching" /
     "set criteria before submitting" are inverted phrasings of the
     real pattern "after changing a filter/picker value, find the
     explicit Apply/Done/Search confirmation before moving on". If you
     see an inverted phrasing but the underlying signal is genuine
     post-change confirmation, REPHRASE to the correct direction
     instead of dropping.
  4. DROP any group that, after unioning, still has fewer than 3
     supporting_trajectories. Support must be the unioned set of
     unique T### handles.

For each kept entry, run the HARD TEST: name the concrete agent mistake
the pattern prevents. If you cannot, drop it.

Output strict JSON only (no markdown fences):

{
  "consolidated": [
    {
      "proposed_id": "short_snake_case_slug",
      "title": "<= 10 word UI-pattern title",
      "applicable_context": {
        "when": "triggering UI situation",
        "ui_signals": ["..."],
        "domain_hint": "generic | travel | shopping | info | entertainment | service"
      },
      "action_guidance": "heuristic + the failure mode it prevents",
      "action_templates": ["<= 2 templates, optional"],
      "supporting_trajectories": ["T012", "T037", "T089"],
      "prevents_mistake": "concrete failure mode this rule prevents",
      "merged_from": ["exp_xxx_candidate_id_1", "exp_xxx_candidate_id_2"]
    }
  ]
}

Rules:
- supporting_trajectories MUST be the union of handles from the source
  candidates. Do not invent handles.
- merged_from lists the source proposed_id values (for traceability).
- prevents_mistake is REQUIRED and must name a concrete agent mistake.
  If no concrete mistake applies, drop the entry.
"""


V2_SYSTEM_PROMPT = """You are a pattern extractor for a GUI-agent experience
library. You are given a batch of experience-oriented turning-point summaries
distilled from successful trajectories. Your job is to propose reusable
experience entries that capture concrete GUI state-transition rules.

Output strict JSON only (no markdown fences) with this schema:

{
  "candidates": [
    {
      "proposed_id": "short_snake_case_slug",
      "title": "<= 10 word title describing the GUI rule",
      "applicable_context": {
        "when": "one sentence describing the triggering UI situation",
        "ui_signals": ["concrete visible cues from the active state"],
        "domain_hint": "generic | travel | shopping | info | entertainment | service"
      },
      "action_guidance": "one short paragraph explaining what to do and why now",
      "action_templates": ["<= 2 short templates, optional, empty list if none"],
      "supporting_trajectories": ["T012", "T037", "T089"],
      "trigger_ui_state": "short phrase for the pre-state",
      "forbidden_alternative": "tempting wrong action to avoid",
      "expected_postcondition": "what should become committed after the action",
      "confidence": 0.0
    }
  ]
}

Selection rules:
- Prefer rules grounded in pre_state, commit_signal, post_state, and
  failure_if_skipped from the summaries.
- A rule MUST prevent a concrete GUI mistake like leaving a picker without
  confirming, clicking outside an active modal, or advancing while state is
  still pending.
- Reject pure task skeletons and generic workflow summaries.
- Keep the causal direction correct: trigger on the UI state that is active
  now, then recommend the next interaction that resolves it.
- supporting_trajectories MUST use exact T### handles from input.
- A candidate MUST be supported by at least 2 trajectories in this chunk.
- If no genuinely reusable GUI rule exists, return {"candidates": []}.
"""


V2_CONSOLIDATOR_SYSTEM_PROMPT = """You are consolidating candidate GUI-rule
experiences extracted from different chunks of an experience summary buffer.
Many candidates describe the same underlying state-transition rule with
different wording.

Your job:
  1. Merge semantically duplicate GUI rules.
  2. Keep the strongest trigger wording and clearest failure mode.
  3. DROP any candidate that is only a task skeleton or generic workflow tip.
  4. DROP any group whose unioned supporting_trajectories has fewer than 3
     unique T### handles.

Output strict JSON only (no markdown fences):

{
  "consolidated": [
    {
      "proposed_id": "short_snake_case_slug",
      "title": "<= 10 word GUI-rule title",
      "applicable_context": {
        "when": "triggering UI situation",
        "ui_signals": ["..."],
        "domain_hint": "generic | travel | shopping | info | entertainment | service"
      },
      "action_guidance": "what to do and why",
      "action_templates": ["<= 2 templates, optional"],
      "supporting_trajectories": ["T012", "T037", "T089"],
      "prevents_mistake": "concrete agent mistake this rule prevents",
      "trigger_ui_state": "short phrase for the pre-state",
      "forbidden_alternative": "tempting wrong action to avoid",
      "expected_postcondition": "what should become committed after the action",
      "merged_from": ["exp_xxx_candidate_id_1", "exp_xxx_candidate_id_2"]
    }
  ]
}

Rules:
- supporting_trajectories MUST be the union of source handles.
- merged_from lists the source proposed_id values.
- prevents_mistake is REQUIRED and must name a concrete failure mode.
- Keep the rule at the interaction-pattern level, not the site-specific task level.
"""


USER_PROMPT_TEMPLATE = """Batch of {n} causal summaries.

Each summary is rendered compactly. T### is the trajectory handle — use it
verbatim in supporting_trajectories.

Summaries:
{summaries_block}

Emit the candidates JSON now."""


CONSOLIDATOR_USER_TEMPLATE = """Candidate experiences extracted from {n_chunks} chunks.

Candidates (rendered compactly):
{candidates_block}

Consolidate these into the canonical list. Apply K>=3 on the unioned
supporting_trajectories. Drop task skeletons. Output the JSON."""


def _compact_encode_v1(idx: int, record: dict) -> str:
    s = record["summary"]
    handle = f"T{idx:03d}"
    domain = record.get("domain", "")
    subdomain = record.get("subdomain", "")
    goal = s.get("goal", "")
    patterns = s.get("tool_usage_patterns") or []
    patterns_str = "; ".join(patterns) if patterns else "(none)"

    effectiveness = s.get("skill_effectiveness") or {}
    decisive = [skill for skill, label in effectiveness.items() if label == "decisive"]
    necessary = [skill for skill, label in effectiveness.items() if label == "necessary"]

    turning_lines: list[str] = []
    for tp in s.get("critical_turning_points") or []:
        step = tp.get("step", "?")
        decision = tp.get("decision", "")
        reason = tp.get("reason", "")
        turning_lines.append(f"    step {step}: {decision} — {reason}")

    lines = [
        f"[{handle}] domain={domain}/{subdomain}",
        f"  goal: {goal}",
        f"  patterns: {patterns_str}",
        f"  decisive: {'; '.join(decisive) if decisive else '(none)'}",
        f"  necessary: {'; '.join(necessary) if necessary else '(none)'}",
    ]
    if turning_lines:
        lines.append("  turning_points:")
        lines.extend(turning_lines)
    else:
        lines.append("  turning_points: (none)")
    return "\n".join(lines)


def _compact_encode_v2(idx: int, record: dict) -> str:
    s = record["summary"]
    handle = f"T{idx:03d}"
    domain = record.get("domain", "")
    subdomain = record.get("subdomain", "")
    goal = s.get("goal", "")
    task_shape = s.get("task_shape", "")
    lines = [
        f"[{handle}] domain={domain}/{subdomain}",
        f"  goal: {goal}",
        f"  task_shape: {task_shape}",
    ]
    turning_points = s.get("turning_points") or []
    if not turning_points:
        lines.append("  turning_points: (none)")
        return "\n".join(lines)

    lines.append("  turning_points:")
    for tp in turning_points:
        step = tp.get("step", "?")
        pre = tp.get("pre_state") or {}
        action = tp.get("action") or {}
        post = tp.get("post_state") or {}
        commit_signal = tp.get("commit_signal") or []
        lines.extend([
            f"    step {step}: pattern={tp.get('generalizable_pattern', '')}",
            (
                "      pre_state: "
                f"ui_context={pre.get('ui_context', '')}; "
                f"active_subflow={pre.get('active_subflow', '')}; "
                f"recent_user_commit={pre.get('recent_user_commit', '')}; "
                f"pending_commit={pre.get('pending_commit', False)}"
            ),
            (
                "      action: "
                f"{action.get('op', '')} target={action.get('target', '')} "
                f"role={action.get('target_role', '')}"
            ),
            f"      commit_signal: {'; '.join(commit_signal) if commit_signal else '(none)'}",
            (
                "      post_state: "
                f"state_change={post.get('state_change', '')}; "
                f"subflow_status={post.get('subflow_status', '')}"
            ),
            f"      failure_if_skipped: {tp.get('failure_if_skipped', '')}",
        ])
    rejected = s.get("rejected_branches") or []
    lines.append(f"  rejected_branches: {'; '.join(rejected) if rejected else '(none)'}")
    return "\n".join(lines)


def _compact_encode(idx: int, record: dict, experience_version: str) -> str:
    if experience_version == "v2":
        return _compact_encode_v2(idx, record)
    return _compact_encode_v1(idx, record)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start : end + 1])


def _validate_candidate(candidate: dict, valid_handles: set[str]) -> list[str]:
    errors: list[str] = []
    for field in CANDIDATE_REQUIRED_FIELDS:
        if field not in candidate:
            errors.append(f"missing: {field}")
    if errors:
        return errors

    context = candidate["applicable_context"]
    if not isinstance(context, dict):
        errors.append("applicable_context must be object")
    else:
        for key in CONTEXT_REQUIRED_FIELDS:
            if key not in context:
                errors.append(f"applicable_context.{key} missing")

    supports = candidate["supporting_trajectories"]
    if not isinstance(supports, list) or not supports:
        errors.append("supporting_trajectories must be non-empty list")
    else:
        unknown = [h for h in supports if h not in valid_handles]
        if unknown:
            errors.append(f"unknown handles: {unknown[:3]}")

    templates = candidate.get("action_templates") or []
    if not isinstance(templates, list):
        errors.append("action_templates must be list")
    elif len(templates) > 2:
        errors.append("action_templates must have <= 2 entries")

    for key in ("title", "action_guidance"):
        value = candidate.get(key, "")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be non-empty string")

    return errors


def _propose_from_chunk(
    engine: OpenAICompatibleEngine,
    chunk: list[tuple[int, dict]],
    handle_to_annotation: dict[str, str],
    max_tokens: int,
    experience_version: str,
) -> list[dict]:
    summaries_block = "\n\n".join(_compact_encode(idx, rec, experience_version) for idx, rec in chunk)
    user_prompt = USER_PROMPT_TEMPLATE.format(n=len(chunk), summaries_block=summaries_block)
    system_prompt = V2_SYSTEM_PROMPT if experience_version == "v2" else V1_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = engine.chat(messages, max_tokens=max_tokens)
    parsed = _extract_json_object(raw)
    candidates = parsed.get("candidates") or []
    if not isinstance(candidates, list):
        raise ValueError("`candidates` field must be a list")

    valid_handles = {f"T{idx:03d}" for idx, _ in chunk}
    cleaned: list[dict] = []
    for candidate in candidates:
        errors = _validate_candidate(candidate, valid_handles)
        if errors:
            logger.warning("dropping candidate %r: %s", candidate.get("proposed_id"), errors)
            continue
        candidate["supporting_annotations"] = [
            handle_to_annotation[h] for h in candidate["supporting_trajectories"]
        ]
        cleaned.append(candidate)
    return cleaned


_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_title(title: str) -> str:
    t = _NORM_RE.sub(" ", title.lower())
    return " ".join(t.split())


def _compact_candidate(candidate: dict) -> str:
    supports = ", ".join(candidate["supporting_trajectories"])
    templates = candidate.get("action_templates") or []
    lines = [
        f"[{candidate['proposed_id']}] title: {candidate['title']}",
        f"  when: {candidate['applicable_context']['when']}",
        f"  guidance: {candidate['action_guidance']}",
        f"  support ({len(candidate['supporting_trajectories'])}): {supports}",
    ]
    if templates:
        lines.append(f"  templates: {templates}")
    if candidate.get("trigger_ui_state"):
        lines.append(f"  trigger_ui_state: {candidate['trigger_ui_state']}")
    if candidate.get("forbidden_alternative"):
        lines.append(f"  forbidden_alternative: {candidate['forbidden_alternative']}")
    if candidate.get("expected_postcondition"):
        lines.append(f"  expected_postcondition: {candidate['expected_postcondition']}")
    return "\n".join(lines)


def _consolidate_candidates(
    engine: OpenAICompatibleEngine,
    candidates: list[dict],
    n_chunks: int,
    handle_to_annotation: dict[str, str],
    max_tokens: int,
    experience_version: str,
) -> list[dict]:
    if not candidates:
        return []

    valid_handles = set(handle_to_annotation.keys())
    id_to_candidate = {c["proposed_id"]: c for c in candidates}

    candidates_block = "\n\n".join(_compact_candidate(c) for c in candidates)
    user_prompt = CONSOLIDATOR_USER_TEMPLATE.format(
        n_chunks=n_chunks, candidates_block=candidates_block
    )
    system_prompt = (
        V2_CONSOLIDATOR_SYSTEM_PROMPT if experience_version == "v2"
        else V1_CONSOLIDATOR_SYSTEM_PROMPT
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = engine.chat(messages, max_tokens=max_tokens)
    parsed = _extract_json_object(raw)
    consolidated_raw = parsed.get("consolidated") or []
    if not isinstance(consolidated_raw, list):
        raise ValueError("`consolidated` must be a list")

    results: list[dict] = []
    for entry in consolidated_raw:
        errors = _validate_candidate(entry, valid_handles)
        if not entry.get("prevents_mistake"):
            errors.append("missing: prevents_mistake")
        if errors:
            logger.warning(
                "dropping consolidated entry %r: %s", entry.get("proposed_id"), errors
            )
            continue

        supports = sorted(set(entry["supporting_trajectories"]))
        annotations = [handle_to_annotation[h] for h in supports]

        merged_from = entry.get("merged_from") or []
        for source_id in merged_from:
            source = id_to_candidate.get(source_id)
            if source:
                annotations.extend(source["supporting_annotations"])
        annotations = sorted(set(annotations))

        templates = list(entry.get("action_templates") or [])[:2]

        results.append({
            "proposed_id": entry["proposed_id"],
            "title": entry["title"],
            "applicable_context": entry["applicable_context"],
            "action_guidance": entry["action_guidance"],
            "action_templates": templates,
            "supporting_annotations": annotations,
            "prevents_mistake": entry["prevents_mistake"],
            "trigger_ui_state": entry.get("trigger_ui_state", ""),
            "forbidden_alternative": entry.get("forbidden_alternative", ""),
            "expected_postcondition": entry.get("expected_postcondition", ""),
            "merged_from": merged_from,
            "confidence": float(entry.get("confidence") or 0.0),
        })
    return results


def _assign_experience_ids(candidates: list[dict], existing: dict[str, dict]) -> list[dict]:
    results: list[dict] = []
    used_ids = set(existing.keys())
    for candidate in candidates:
        title_key = _normalize_title(candidate["title"])
        matched_id = None
        for exp_id, record in existing.items():
            if _normalize_title(record["title"]) == title_key:
                matched_id = exp_id
                break
        if matched_id:
            experience_id = matched_id
        else:
            slug = re.sub(r"[^a-z0-9]+", "_", candidate["proposed_id"].lower()).strip("_")
            base = f"exp_{slug}" if slug else "exp_pattern"
            experience_id = base
            suffix = 2
            while experience_id in used_ids:
                experience_id = f"{base}_{suffix}"
                suffix += 1
            used_ids.add(experience_id)
        candidate["experience_id"] = experience_id
        results.append(candidate)
    return results


def _finalize_library_record(
    candidate: dict,
    existing: dict | None,
    timestamp: str,
) -> dict:
    support_set = set(candidate["supporting_annotations"])
    if existing:
        support_set |= set(existing.get("evidence", {}).get("supporting_trajectories", []))

    record = {
        "experience_id": candidate["experience_id"],
        "title": candidate["title"],
        "applicable_context": candidate["applicable_context"],
        "action_guidance": candidate["action_guidance"],
        "action_templates": candidate.get("action_templates") or [],
        "prevents_mistake": candidate.get("prevents_mistake", ""),
        "trigger_ui_state": candidate.get("trigger_ui_state", ""),
        "forbidden_alternative": candidate.get("forbidden_alternative", ""),
        "expected_postcondition": candidate.get("expected_postcondition", ""),
        "evidence": {
            "support_count": len(support_set),
            "supporting_trajectories": sorted(support_set),
        },
        "confidence": candidate.get("confidence") or 0.0,
        "last_updated": timestamp,
    }
    if existing:
        record["created_at"] = existing.get("created_at", timestamp)
    else:
        record["created_at"] = timestamp
    return record


def _load_library(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    library: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("experience_id"):
                library[record["experience_id"]] = record
    return library


def _write_library(path: Path, library: dict[str, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in library.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _build_catalog(library: dict[str, dict], cap: int) -> list[dict]:
    entries = []
    for record in library.values():
        entries.append({
            "id": record["experience_id"],
            "title": record["title"],
            "trigger": record["applicable_context"]["when"],
            "_support_count": record["evidence"]["support_count"],
            "_last_updated": record.get("last_updated", ""),
        })
    entries.sort(key=lambda e: (e["_support_count"], e["_last_updated"]), reverse=True)
    entries = entries[:cap]
    for entry in entries:
        entry.pop("_support_count", None)
        entry.pop("_last_updated", None)
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage B: mine recurring patterns from Stage A summaries."
    )
    parser.add_argument(
        "--buffer",
        default="outputs/cross_task_experience/summary_buffer.jsonl",
    )
    parser.add_argument(
        "--library",
        default="outputs/cross_task_experience/experience_library.jsonl",
    )
    parser.add_argument(
        "--catalog",
        default="outputs/cross_task_experience/catalog.json",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--rate-limit", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument(
        "--experience-version",
        default="v1",
        choices=["v1", "v2"],
        help="Pattern extraction schema/prompt version.",
    )
    parser.add_argument("--support-threshold", type=int, default=3, help="K in §3.2")
    parser.add_argument("--catalog-cap", type=int, default=50)

    args = parser.parse_args(argv)

    buffer_path = Path(args.buffer)
    library_path = Path(args.library)
    catalog_path = Path(args.catalog)
    library_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    with buffer_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            summaries.append(json.loads(line))
    logger.info("loaded %d summaries from %s", len(summaries), buffer_path)

    indexed = list(enumerate(summaries, start=1))
    handle_to_annotation = {f"T{idx:03d}": rec["annotation_id"] for idx, rec in indexed}

    engine = OpenAICompatibleEngine(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
    )

    all_candidates: list[dict] = []
    for start in range(0, len(indexed), args.chunk_size):
        chunk = indexed[start : start + args.chunk_size]
        logger.info("proposing from chunk %d-%d (size=%d)", start + 1, start + len(chunk), len(chunk))
        try:
            candidates = _propose_from_chunk(
                engine,
                chunk,
                handle_to_annotation,
                args.max_tokens,
                args.experience_version,
            )
        except Exception as exc:
            logger.warning("chunk %d-%d failed: %s", start + 1, start + len(chunk), exc)
            continue
        logger.info("  -> %d candidates", len(candidates))
        all_candidates.extend(candidates)

    n_chunks = (len(indexed) + args.chunk_size - 1) // args.chunk_size
    logger.info("consolidating %d candidates from %d chunks via LLM", len(all_candidates), n_chunks)
    try:
        consolidated = _consolidate_candidates(
            engine,
            all_candidates,
            n_chunks,
            handle_to_annotation,
            args.max_tokens,
            args.experience_version,
        )
    except Exception as exc:
        logger.warning("consolidator LLM failed (%s); falling back to raw candidates", exc)
        consolidated = all_candidates

    logger.info("consolidated to %d entries", len(consolidated))

    kept = [c for c in consolidated if len(c["supporting_annotations"]) >= args.support_threshold]
    dropped = len(consolidated) - len(kept)
    logger.info("applied K>=%d: kept=%d dropped=%d", args.support_threshold, len(kept), dropped)

    existing = _load_library(library_path)
    kept = _assign_experience_ids(kept, existing)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for candidate in kept:
        existing_record = existing.get(candidate["experience_id"])
        existing[candidate["experience_id"]] = _finalize_library_record(
            candidate, existing_record, timestamp
        )

    _write_library(library_path, existing)
    catalog = _build_catalog(existing, args.catalog_cap)
    with catalog_path.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, ensure_ascii=False, indent=2)

    logger.info(
        "library size=%d catalog size=%d wrote library=%s catalog=%s",
        len(existing), len(catalog), library_path, catalog_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage C: step-time experience selector.

Given task text, the last 3 actions, a current-observation summary, and
the full catalog, pick AT MOST one experience whose trigger clearly
applies to the current step, or return null.

Exposes:
  - select_experience(engine, task, recent_steps, current_obs, catalog, library)
      -> {"experience_id": str|None, "reason": str, "injection": str|None}
  - render_experience_slot(record) -> str

CLI mode runs a dry-run on N random static GUI samples and prints what the
selector picks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from evaluate_agentic_memory_task import OpenAICompatibleEngine
from multimodal_utils import load_multimodal_samples

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


SELECTOR_SYSTEM_PROMPT = """You are an experience selector for a GUI agent.

At each step, you receive the task, the last 3 actions, a summary of the
current observation, and a catalog of reusable UI-interaction experiences.

Your job: pick AT MOST ONE experience whose trigger CLEARLY matches the
current step. If nothing clearly applies, return null. Do not stretch to
fit. A null answer is better than a loose match.

Output strict JSON only (no markdown fences):

  {"experience_id": "<id>" | null, "reason": "<one sentence>"}

Rules:
- Exactly one experience_id or null. Never multiple.
- experience_id must come from the catalog verbatim (or be null).
- reason must cite the concrete trigger signal from the current step.
"""


SELECTOR_USER_TEMPLATE = """Task:
{task}

Recent steps (oldest first, up to 3):
{recent_block}

Current observation summary:
{current_obs}

Experience catalog:
{catalog_block}

Pick AT MOST ONE experience, or return null."""


def _format_catalog(catalog: list[dict]) -> str:
    lines = []
    for entry in catalog:
        lines.append(
            f"- id: {entry['id']}\n"
            f"  title: {entry['title']}\n"
            f"  trigger: {entry['trigger']}"
        )
    return "\n".join(lines)


def _format_recent_steps(recent_steps: list[dict]) -> str:
    if not recent_steps:
        return "(none — this is the first step)"
    lines = []
    for i, step in enumerate(recent_steps, start=1):
        action = step.get("action", "(unknown action)")
        obs = step.get("observation_summary", "")
        suffix = f" → {obs}" if obs else ""
        lines.append(f"  step {i}: {action}{suffix}")
    return "\n".join(lines)


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
        raise ValueError("no JSON object found in selector output")
    return json.loads(text[start : end + 1])


def select_experience(
    engine: OpenAICompatibleEngine,
    task: str,
    recent_steps: list[dict],
    current_obs: str,
    catalog: list[dict],
    library: dict[str, dict],
    max_tokens: int = 200,
) -> dict:
    """Run one selector call. Returns dict with experience_id, reason, injection."""
    if not catalog:
        return {"experience_id": None, "reason": "empty catalog", "injection": None}

    user_prompt = SELECTOR_USER_TEMPLATE.format(
        task=task,
        recent_block=_format_recent_steps(recent_steps[-3:]),
        current_obs=current_obs or "(no current observation summary)",
        catalog_block=_format_catalog(catalog),
    )
    messages = [
        {"role": "system", "content": SELECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = engine.chat(messages, max_tokens=max_tokens)
    parsed = _extract_json_object(raw)

    exp_id = parsed.get("experience_id")
    reason = parsed.get("reason", "") or ""
    if exp_id is None or exp_id == "null":
        return {"experience_id": None, "reason": reason, "injection": None}

    record = library.get(exp_id)
    if record is None:
        logger.warning("selector returned unknown experience_id=%s; treating as null", exp_id)
        return {"experience_id": None, "reason": f"unknown id {exp_id}", "injection": None}

    return {
        "experience_id": exp_id,
        "reason": reason,
        "injection": render_experience_slot(record),
    }


def render_experience_slot(record: dict) -> str:
    """Render an experience for injection into the policy prompt."""
    lines = [
        "[active_experience]",
        f"Title: {record['title']}",
        f"When it applies: {record['applicable_context']['when']}",
        f"Guidance: {record['action_guidance']}",
    ]
    if record.get("trigger_ui_state"):
        lines.append(f"Trigger UI state: {record['trigger_ui_state']}")
    if record.get("forbidden_alternative"):
        lines.append(f"Avoid: {record['forbidden_alternative']}")
    if record.get("expected_postcondition"):
        lines.append(f"Expected after action: {record['expected_postcondition']}")
    templates = record.get("action_templates") or []
    if templates:
        lines.append("Suggested action shapes:")
        for template in templates[:2]:
            lines.append(f"- {template}")
    return "\n".join(lines)


def load_library_by_id(path: Path) -> dict[str, dict]:
    library: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            library[rec["experience_id"]] = rec
    return library


def build_context_for_sample(sample: dict) -> tuple[str, list[dict], str]:
    """Turn one static GUI sample into (task, recent_steps, current_obs)."""
    task = sample.get("confirmed_task", "")
    previous = sample.get("previous_actions", [])[-3:]
    recent_steps = [{"action": repr_text} for repr_text in previous]

    if sample.get("action_space") == "aitw":
        ui_elements = sample.get("ui_elements") or []
        current_obs = (
            f"activity={sample.get('current_activity', '')} "
            f"device={sample.get('device_type', '')} "
            f"ui_hint=[{'; '.join(ui_elements[:8])}]"
        )
        return task, recent_steps, current_obs

    pos = sample.get("pos_candidates") or []
    element_hint = ""
    if pos:
        first = pos[0]
        tag = first.get("tag", "")
        attrs_raw = first.get("attributes")
        if isinstance(attrs_raw, str):
            try:
                attrs = json.loads(attrs_raw)
            except json.JSONDecodeError:
                attrs = {}
        else:
            attrs = attrs_raw if isinstance(attrs_raw, dict) else {}
        parts = []
        if tag:
            parts.append(f"tag={tag}")
        for key in ("class", "aria_label", "name", "text"):
            if attrs.get(key):
                trimmed = str(attrs[key]).strip().replace("\n", " ")
                if len(trimmed) > 50:
                    trimmed = trimmed[:47] + "..."
                parts.append(f'{key}="{trimmed}"')
                if len(parts) >= 4:
                    break
        element_hint = ", ".join(parts)

    current_obs = (
        f"website={sample.get('website', '')} "
        f"domain={sample.get('domain', '')}/{sample.get('subdomain', '')} "
        f"target_hint=[{element_hint}]"
    )
    return task, recent_steps, current_obs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage C selector — dry-run on static GUI datasets.")
    parser.add_argument(
        "--catalog",
        default="outputs/cross_task_experience/catalog_4o_v3.json",
    )
    parser.add_argument(
        "--library",
        default="outputs/cross_task_experience/experience_library_4o_v3.jsonl",
    )
    parser.add_argument("--dataset-path", default="data/multimodal_mind2web")
    parser.add_argument("--dataset-format", default="auto", choices=["auto", "mind2web", "aitw"])
    parser.add_argument(
        "--split",
        default="test_task",
    )
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--rate-limit", type=int, default=30)
    parser.add_argument("--output", default="outputs/cross_task_experience/selector_dryrun.jsonl")

    args = parser.parse_args(argv)

    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    library = load_library_by_id(Path(args.library))
    logger.info("loaded catalog size=%d library size=%d", len(catalog), len(library))

    samples = load_multimodal_samples(
        dataset_path=args.dataset_path,
        split=args.split,
        dataset_format=args.dataset_format,
    )
    rng = random.Random(args.seed)
    sampled = rng.sample(samples, min(args.n_samples, len(samples)))
    logger.info("sampled %d steps from split=%s", len(sampled), args.split)

    engine = OpenAICompatibleEngine(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_counts: dict[str, int] = {}
    null_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx, sample in enumerate(sampled, start=1):
            task, recent_steps, current_obs = build_context_for_sample(sample)
            try:
                result = select_experience(
                    engine, task, recent_steps, current_obs, catalog, library
                )
            except Exception as exc:
                logger.warning("selector failed on sample %d: %s", idx, exc)
                continue

            exp_id = result["experience_id"]
            if exp_id is None:
                null_count += 1
                key = "null"
            else:
                key = exp_id
            selected_counts[key] = selected_counts.get(key, 0) + 1

            record = {
                "annotation_id": sample["annotation_id"],
                "action_uid": sample["action_uid"],
                "target_action_index": sample.get("target_action_index"),
                "task": task,
                "current_obs": current_obs,
                "experience_id": exp_id,
                "reason": result["reason"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("=== dry-run selection distribution ===")
    for key, count in sorted(selected_counts.items(), key=lambda kv: -kv[1]):
        logger.info("  %-40s %d", key, count)
    logger.info("null rate: %d/%d = %.1f%%", null_count, len(sampled), 100 * null_count / len(sampled))
    logger.info("wrote %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

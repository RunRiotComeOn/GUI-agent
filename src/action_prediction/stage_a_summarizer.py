"""Stage A: distill a static GUI trajectory into a causal summary.

One LLM call per annotation. Reads a supported offline static GUI dataset,
groups rows by annotation_id, sorts by target_action_index, and asks the
model to emit a causal summary with the six fields defined in
docs/cross_task_experience_replay_v2.md §2.2.

Output is appended to a JSONL buffer. Re-runs skip annotations already
present in the output file.
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


REQUIRED_FIELDS = (
    "goal",
    "key_trajectory",
    "skill_effectiveness",
    "critical_turning_points",
    "tool_usage_patterns",
    "outcome",
)

SKILL_LABELS = {"decisive", "necessary", "redundant"}


SYSTEM_PROMPT = """You are a trajectory summarizer for GUI agents on offline static GUI trajectories.

You receive one successful task trajectory (every step is a ground-truth
target action). Your job is to compress the event stream into an
attributable causal summary — a summary where success is tied to concrete
decision points.

Emit strict JSON only. No markdown fences. No extra commentary.

Required fields:
- goal: the abstracted task intent. Remove the specific site, exact date,
  exact location, person name, passenger count, or other one-off values.
  Keep the intent shape (e.g. "complete a hotel search with date and
  guest constraints on an OTA").
- key_trajectory: ordered list of short strings in the form
  "step N: <skill_name> <what it accomplished>". Only include load-bearing
  steps.
- skill_effectiveness: object mapping <skill_name> -> one of
  "decisive" | "necessary" | "redundant". Use EXACTLY these three labels.
  Choose semantic skill names (e.g. "click_confirm_after_filter"),
  not raw action types.
- critical_turning_points: list of objects
  {"step": <int>, "decision": <str>, "reason": <str>}.
  Include ONLY steps where the trajectory branched or committed state
  (e.g. modal opened, filter confirmed, checkout advanced). Every entry
  MUST carry a non-empty reason. If there are no genuine turning points,
  return an empty list.
- tool_usage_patterns: list of short recurring shapes like
  "fill -> confirm -> advance".
- outcome: one sentence attributing success to the concrete turning points
  above.

Grounding rules:
- Do not invent steps or UI that are not in the input.
- Do not include site-specific trivia ("this website's header is blue").
- Do not copy raw DOM ids or full element attributes into the summary.
"""


USER_PROMPT_TEMPLATE = """Task:
{task}

Website: {website}
Domain: {domain}
Subdomain: {subdomain}

Steps:
{steps_block}

Emit the causal summary JSON now."""


def _load_stage_a_samples(
    dataset_path: str,
    split: str,
    annotation_ids: list[str] | None = None,
    websites: list[str] | None = None,
    domains: list[str] | None = None,
    dataset_format: str = "auto",
) -> list[dict]:
    return load_multimodal_samples(
        dataset_path=dataset_path,
        split=split,
        annotation_ids=annotation_ids,
        websites=websites,
        domains=domains,
        dataset_format=dataset_format,
    )


def _element_hint(sample: dict) -> str:
    pos = sample.get("pos_candidates") or []
    if not pos:
        return ""
    first = pos[0]
    tag = first.get("tag", "")
    attrs_raw = first.get("attributes")
    if isinstance(attrs_raw, str):
        try:
            attrs = json.loads(attrs_raw)
        except json.JSONDecodeError:
            attrs = {}
    elif isinstance(attrs_raw, dict):
        attrs = attrs_raw
    else:
        attrs = {}
    hint_parts = []
    if tag:
        hint_parts.append(f"tag={tag}")
    for key in ("class", "text", "aria_label", "name", "id"):
        value = attrs.get(key)
        if value:
            trimmed = str(value).strip().replace("\n", " ")
            if len(trimmed) > 60:
                trimmed = trimmed[:57] + "..."
            hint_parts.append(f'{key}="{trimmed}"')
            if len(hint_parts) >= 4:
                break
    return ", ".join(hint_parts)


def _format_steps_block(trajectory_samples: list[dict]) -> str:
    lines = []
    for sample in trajectory_samples:
        idx = int(sample.get("target_action_index", -1)) + 1
        action_repr = sample.get("target_action_reprs") or "(missing action_repr)"
        operation = sample.get("operation") or {}
        op = operation.get("op", "")
        value = operation.get("value", "") or ""
        value_display = value if len(str(value)) <= 40 else f"{str(value)[:37]}..."
        element_hint = _element_hint(sample)
        if sample.get("action_space") == "aitw":
            ui_hint = "; ".join((sample.get("ui_elements") or [])[:4])
            element_hint = ui_hint
        lines.append(
            f"{idx}: {action_repr} | op={op} | value={value_display!r} | {element_hint}"
        )
    return "\n".join(lines)


def _group_by_annotation(samples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample["annotation_id"], []).append(sample)
    for annotation_id, items in grouped.items():
        items.sort(key=lambda item: int(item.get("target_action_index", 0)))
    return grouped


def _load_existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in existing buffer: %s", line[:120])
                continue
            annotation_id = record.get("annotation_id")
            if annotation_id:
                seen.add(annotation_id)
    return seen


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


def _validate_summary(summary: dict) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in summary:
            errors.append(f"missing field: {field}")
    if errors:
        return errors

    if not isinstance(summary["goal"], str) or not summary["goal"].strip():
        errors.append("goal must be a non-empty string")

    if not isinstance(summary["key_trajectory"], list):
        errors.append("key_trajectory must be a list")

    effectiveness = summary["skill_effectiveness"]
    if not isinstance(effectiveness, dict):
        errors.append("skill_effectiveness must be an object")
    else:
        for skill, label in effectiveness.items():
            if label not in SKILL_LABELS:
                errors.append(
                    f"skill_effectiveness[{skill!r}]={label!r} not in {sorted(SKILL_LABELS)}"
                )

    turning_points = summary["critical_turning_points"]
    if not isinstance(turning_points, list):
        errors.append("critical_turning_points must be a list")
    else:
        for i, entry in enumerate(turning_points):
            if not isinstance(entry, dict):
                errors.append(f"critical_turning_points[{i}] must be an object")
                continue
            for key in ("step", "decision", "reason"):
                if key not in entry or entry[key] in (None, "", []):
                    errors.append(f"critical_turning_points[{i}].{key} missing or empty")

    if not isinstance(summary["tool_usage_patterns"], list):
        errors.append("tool_usage_patterns must be a list")

    if not isinstance(summary["outcome"], str) or not summary["outcome"].strip():
        errors.append("outcome must be a non-empty string")

    return errors


def _summarize_one(
    engine: OpenAICompatibleEngine,
    annotation_id: str,
    trajectory_samples: list[dict],
    max_tokens: int,
) -> dict:
    first = trajectory_samples[0]
    user_prompt = USER_PROMPT_TEMPLATE.format(
        task=first.get("confirmed_task", ""),
        website=first.get("website", ""),
        domain=first.get("domain", ""),
        subdomain=first.get("subdomain", ""),
        steps_block=_format_steps_block(trajectory_samples),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    raw = engine.chat(messages, max_tokens=max_tokens)
    summary = _extract_json_object(raw)
    errors = _validate_summary(summary)
    if errors:
        raise ValueError(f"summary validation failed: {errors}")

    return {
        "annotation_id": annotation_id,
        "dataset_format": first.get("dataset_format", "unknown"),
        "source_dataset": first.get("dataset_format", "unknown"),
        "task": first.get("confirmed_task", ""),
        "website": first.get("website", ""),
        "domain": first.get("domain", ""),
        "subdomain": first.get("subdomain", ""),
        "num_steps": len(trajectory_samples),
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage A: distill offline static GUI trajectories into causal summaries."
    )
    parser.add_argument("--dataset-path", default="data/multimodal_mind2web")
    parser.add_argument("--dataset-format", default="auto", choices=["auto", "mind2web", "aitw"])
    parser.add_argument(
        "--split",
        default="test_task",
    )
    parser.add_argument(
        "--output",
        default="outputs/cross_task_experience/summary_buffer.jsonl",
        help="JSONL buffer. Re-runs skip annotation_ids already present.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--rate-limit", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=None, help="Max annotations to summarize.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample this many annotations before summarization.",
    )
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--annotation-id", action="append", default=[])
    parser.add_argument("--website", action="append", default=[])
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Skip trajectories whose step count exceeds this bound.",
    )

    args = parser.parse_args(argv)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading samples from %s split=%s", args.dataset_path, args.split)
    samples = _load_stage_a_samples(
        dataset_path=args.dataset_path,
        split=args.split,
        annotation_ids=args.annotation_id,
        websites=args.website,
        domains=args.domain,
        dataset_format=args.dataset_format,
    )
    grouped = _group_by_annotation(samples)
    logger.info("Loaded %d samples across %d annotations.", len(samples), len(grouped))

    existing_ids = _load_existing_ids(output_path)
    if existing_ids:
        logger.info("Found %d annotations already in %s; will skip.", len(existing_ids), output_path)

    engine = OpenAICompatibleEngine(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
        temperature=args.temperature,
    )

    todo = [aid for aid in grouped if aid not in existing_ids]
    if args.sample_size is not None and len(todo) > args.sample_size:
        rng = random.Random(args.sample_seed)
        todo = rng.sample(todo, args.sample_size)
        logger.info(
            "Randomly sampled %d annotations with seed=%d.",
            len(todo),
            args.sample_seed,
        )
    if args.limit is not None:
        todo = todo[: args.limit]
    logger.info("%d annotations to summarize (after skip/limit).", len(todo))

    successes = 0
    failures = 0
    with output_path.open("a", encoding="utf-8") as out:
        for idx, annotation_id in enumerate(todo, start=1):
            trajectory = grouped[annotation_id]
            if len(trajectory) > args.max_steps:
                logger.info(
                    "[%d/%d] skipping annotation %s: %d steps > max_steps=%d",
                    idx, len(todo), annotation_id, len(trajectory), args.max_steps,
                )
                continue
            try:
                record = _summarize_one(engine, annotation_id, trajectory, args.max_tokens)
            except Exception as exc:
                failures += 1
                logger.warning(
                    "[%d/%d] annotation %s failed: %s", idx, len(todo), annotation_id, exc,
                )
                continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            successes += 1
            if idx % 10 == 0 or idx == len(todo):
                logger.info(
                    "[%d/%d] ok=%d fail=%d latest=%s",
                    idx, len(todo), successes, failures, annotation_id,
                )

    logger.info("Done. wrote=%d failed=%d output=%s", successes, failures, output_path)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

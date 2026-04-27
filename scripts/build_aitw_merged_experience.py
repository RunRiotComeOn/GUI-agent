#!/usr/bin/env python3
"""Build a merged Mind2Web + AITW experience library.

Workflow:
1. Seed a merged summary buffer from an existing Mind2Web summary buffer.
2. Append Stage A summaries from a sampled subset of AITW tasks.
3. Seed a merged library from an existing Mind2Web experience library.
4. Run Stage B consolidation over the merged summary buffer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _dedupe_summary_buffer(path: Path) -> int:
    if not path.exists():
        return 0
    seen: set[tuple[str, str]] = set()
    kept: list[str] = []
    removed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            key = (
                str(record.get("dataset_format") or record.get("source_dataset") or "unknown"),
                str(record.get("annotation_id") or ""),
            )
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept.append(raw)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for line in kept:
            handle.write(line + "\n")
    tmp.replace(path)
    return removed


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged Mind2Web + AITW experience artifacts.")
    parser.add_argument("--aitw-dataset-path", required=True)
    parser.add_argument("--aitw-split", default="all")
    parser.add_argument("--aitw-sample-size", type=int, default=1500)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument(
        "--base-summary-buffer",
        default="outputs/cross_task_experience/summary_buffer_all_tasks.jsonl",
    )
    parser.add_argument(
        "--fallback-base-summary-buffer",
        default="outputs/cross_task_experience/summary_buffer.jsonl",
    )
    parser.add_argument(
        "--merged-summary-buffer",
        default="outputs/cross_task_experience/summary_buffer_mind2web_aitw_1500.jsonl",
    )
    parser.add_argument(
        "--base-library",
        default="outputs/cross_task_experience/experience_library_4o_v3.jsonl",
    )
    parser.add_argument(
        "--merged-library",
        default="outputs/cross_task_experience/experience_library_mind2web_aitw_1500.jsonl",
    )
    parser.add_argument(
        "--merged-catalog",
        default="outputs/cross_task_experience/catalog_mind2web_aitw_1500.json",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--stage-a-rate-limit", type=int, default=30)
    parser.add_argument("--stage-b-rate-limit", type=int, default=20)
    parser.add_argument("--stage-a-max-tokens", type=int, default=1200)
    parser.add_argument("--stage-b-max-tokens", type=int, default=3000)
    parser.add_argument(
        "--experience-version",
        default="v1",
        choices=["v1", "v2"],
        help="Stage A/B experience extraction version.",
    )
    parser.add_argument("--stage-a-max-steps", type=int, default=30)
    parser.add_argument("--stage-b-chunk-size", type=int, default=50)
    parser.add_argument("--support-threshold", type=int, default=3)
    parser.add_argument("--catalog-cap", type=int, default=50)
    parser.add_argument(
        "--refresh-merged-buffer",
        action="store_true",
        help="Recreate the merged summary buffer from the base summary buffer before appending AITW.",
    )
    parser.add_argument(
        "--refresh-merged-library",
        action="store_true",
        help="Recreate the merged experience library from the base Mind2Web library before Stage B.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = args.api_key
    env["OPENAI_BASE_URL"] = args.api_base

    base_summary = Path(args.base_summary_buffer)
    if not base_summary.exists():
        base_summary = Path(args.fallback_base_summary_buffer)
    if not base_summary.exists():
        raise FileNotFoundError("No base Mind2Web summary buffer found.")

    merged_summary = Path(args.merged_summary_buffer)
    merged_library = Path(args.merged_library)
    merged_catalog = Path(args.merged_catalog)
    base_library = Path(args.base_library)

    if args.refresh_merged_buffer and merged_summary.exists():
        merged_summary.unlink()
    if args.refresh_merged_library and merged_library.exists():
        merged_library.unlink()

    _copy_if_missing(base_summary, merged_summary)
    _copy_if_missing(base_library, merged_library)

    removed = _dedupe_summary_buffer(merged_summary)
    if removed:
        print(f"deduped {removed} duplicate summaries from {merged_summary}", flush=True)

    stage_a_cmd = [
        sys.executable,
        "src/action_prediction/stage_a_summarizer.py",
        "--dataset-path",
        args.aitw_dataset_path,
        "--dataset-format",
        "aitw",
        "--split",
        args.aitw_split,
        "--output",
        str(merged_summary),
        "--model",
        args.model,
        "--api-base",
        args.api_base,
        "--api-key",
        args.api_key,
        "--rate-limit",
        str(args.stage_a_rate_limit),
        "--max-tokens",
        str(args.stage_a_max_tokens),
        "--experience-version",
        args.experience_version,
        "--max-steps",
        str(args.stage_a_max_steps),
        "--sample-size",
        str(args.aitw_sample_size),
        "--sample-seed",
        str(args.sample_seed),
    ]
    _run(stage_a_cmd, env)

    removed = _dedupe_summary_buffer(merged_summary)
    if removed:
        print(f"deduped {removed} duplicate summaries from {merged_summary}", flush=True)

    stage_b_cmd = [
        sys.executable,
        "src/action_prediction/stage_b_pattern_extractor.py",
        "--buffer",
        str(merged_summary),
        "--library",
        str(merged_library),
        "--catalog",
        str(merged_catalog),
        "--model",
        args.model,
        "--api-base",
        args.api_base,
        "--api-key",
        args.api_key,
        "--rate-limit",
        str(args.stage_b_rate_limit),
        "--chunk-size",
        str(args.stage_b_chunk_size),
        "--max-tokens",
        str(args.stage_b_max_tokens),
        "--experience-version",
        args.experience_version,
        "--support-threshold",
        str(args.support_threshold),
        "--catalog-cap",
        str(args.catalog_cap),
    ]
    _run(stage_b_cmd, env)

    print("merged summary buffer:", merged_summary)
    print("merged experience library:", merged_library)
    print("merged catalog:", merged_catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

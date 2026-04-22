#!/usr/bin/env python3
"""Sample a minimal local AITW subset for experience extraction.

This script avoids downloading the full AITW dataset. Instead it:
1. Lists public GCS shards for selected subsets.
2. Randomizes shard order with a seed.
3. Downloads only the shards needed to collect N unique episodes.
4. Parses TFRecord+GZIP files and writes a lightweight JSONL dataset that
   keeps only Stage-A-relevant fields and drops screenshot bytes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


def _ensure_tfrecord_import() -> None:
    tmp_pkg = "/tmp/tfrecord_pkg"
    if tmp_pkg not in sys.path and Path(tmp_pkg).exists():
        sys.path.insert(0, tmp_pkg)
    try:
        import tfrecord  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing tfrecord package. Install it into /tmp/tfrecord_pkg first."
        ) from exc


_ensure_tfrecord_import()
from tfrecord.reader import tfrecord_loader  # type: ignore  # noqa: E402


BASE_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
BASE_MEDIA = "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/{name}?generation={generation}&alt=media"


def _api_json(url: str) -> dict:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def list_objects(prefix: str) -> list[dict]:
    objects: list[dict] = []
    page_token: str | None = None
    while True:
        params = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            params["pageToken"] = page_token
        url = BASE_API + "?" + urllib.parse.urlencode(params)
        payload = _api_json(url)
        for item in payload.get("items", []):
            name = item["name"]
            if name.endswith("/"):
                continue
            objects.append(item)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return objects


def _download_object(item: dict, dest_root: Path) -> Path:
    name = item["name"]
    rel = name.removeprefix("android-in-the-wild/")
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item.get("size", "0"))
    if dest.exists() and dest.stat().st_size == expected:
        return dest

    quoted = urllib.parse.quote(name, safe="")
    url = BASE_MEDIA.format(name=quoted, generation=item["generation"])
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    actual = tmp.stat().st_size
    if expected and actual != expected:
        raise RuntimeError(f"Size mismatch for {rel}: expected {expected}, got {actual}")
    tmp.replace(dest)
    return dest


def _decode_scalar(value):
    if hasattr(value, "shape") and getattr(value, "shape", None) == (1,):
        try:
            value = value[0]
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        if len(value) == 1:
            return _decode_scalar(value[0])
        return [_decode_scalar(item) for item in value]
    if hasattr(value, "tolist"):
        value = value.tolist()
        if isinstance(value, list):
            return [_decode_scalar(item) for item in value]
    return value


def _extract_episode_numeric_id(raw_episode_id: str) -> int | None:
    prefix = raw_episode_id.split(".", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return None


def _normalize_record(example: dict, subset: str) -> dict:
    ui_text = _decode_scalar(example.get("image/ui_annotations_text", []))
    if not isinstance(ui_text, list):
        ui_text = [ui_text] if ui_text else []
    ui_types = _decode_scalar(example.get("image/ui_annotations_ui_types", []))
    if not isinstance(ui_types, list):
        ui_types = [ui_types] if ui_types else []
    positions = _decode_scalar(example.get("image/ui_annotations_positions", []))
    if not isinstance(positions, list):
        positions = [positions] if positions else []
    touch = _decode_scalar(example.get("results/yx_touch"))
    lift = _decode_scalar(example.get("results/yx_lift"))
    if not isinstance(touch, list):
        touch = None
    if not isinstance(lift, list):
        lift = None

    raw_episode = _decode_scalar(example.get("episode_id", ""))
    numeric_episode_id = _extract_episode_numeric_id(raw_episode)
    return {
        "ep_id": raw_episode,
        "episode_numeric_id": numeric_episode_id,
        "episode_length": int(_decode_scalar(example.get("episode_length", 0)) or 0),
        "step_id": int(_decode_scalar(example.get("step_id", 0)) or 0),
        "goal_info": _decode_scalar(example.get("goal_info", "")),
        "current_activity": _decode_scalar(example.get("current_activity", "")),
        "device_type": _decode_scalar(example.get("device_type", "")),
        "android_api_level": int(_decode_scalar(example.get("android_api_level", 0)) or 0),
        "image_ui_annotations_text": ui_text,
        "image_ui_annotations_ui_types": ui_types,
        "image_ui_annotations_positions": positions,
        "results_action_type": int(_decode_scalar(example.get("results/action_type", 0)) or 0),
        "results_type_action": _decode_scalar(example.get("results/type_action", "")),
        "results_yx_touch": touch,
        "results_yx_lift": lift,
        "source_subset": subset,
    }


def load_split_ids(split_path: Path, split_label: str | None) -> set[int] | None:
    if split_label is None:
        return None
    data = json.loads(split_path.read_text(encoding="utf-8"))
    ids = data.get(split_label)
    if ids is None:
        raise ValueError(f"Split label {split_label!r} not found in {split_path}")
    return {int(x) for x in ids}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a minimal AITW subset without downloading full shards.")
    parser.add_argument("--subsets", nargs="+", default=["single", "general"])
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split-file", default="data/android_in_the_wild/splits/standard.json")
    parser.add_argument("--split-label", default="train", help="Set to none to ignore split filtering.")
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="aitw_single_general_standard_train_1500.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split_label = None if str(args.split_label).lower() == "none" else args.split_label
    allowed_episode_ids = load_split_ids(Path(args.split_file), split_label)

    dest_root = Path(args.dest_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    rng = random.Random(args.seed)
    objects: list[tuple[str, dict]] = []
    for subset in args.subsets:
        prefix = f"android-in-the-wild/{subset}/"
        subset_objects = list_objects(prefix)
        rng.shuffle(subset_objects)
        objects.extend((subset, item) for item in subset_objects)
    rng.shuffle(objects)

    selected_episodes: dict[str, list[dict]] = defaultdict(list)
    downloaded_files = 0
    scanned_records = 0

    for subset, item in objects:
        if len(selected_episodes) >= args.sample_size:
            break
        local_path = _download_object(item, dest_root)
        downloaded_files += 1
        print(
            f"[download {downloaded_files}] subset={subset} file={local_path.name} selected={len(selected_episodes)}",
            flush=True,
        )
        try:
            for example in tfrecord_loader(str(local_path), None, compression_type="gzip"):
                scanned_records += 1
                normalized = _normalize_record(example, subset)
                episode_numeric_id = normalized["episode_numeric_id"]
                if allowed_episode_ids is not None and episode_numeric_id not in allowed_episode_ids:
                    continue
                episode_id = normalized["ep_id"]
                if episode_id not in selected_episodes and len(selected_episodes) >= args.sample_size:
                    continue
                selected_episodes[episode_id].append(normalized)
        except Exception as exc:
            print(f"warning: failed to parse {local_path}: {exc}", file=sys.stderr, flush=True)

    episodes = list(selected_episodes.items())
    episodes.sort(key=lambda item: item[0])
    with output_path.open("w", encoding="utf-8") as handle:
        for _, records in episodes:
            records.sort(key=lambda item: item["step_id"])
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "subsets": args.subsets,
        "sample_size_requested": args.sample_size,
        "sample_size_collected": len(episodes),
        "seed": args.seed,
        "split_file": args.split_file,
        "split_label": split_label,
        "downloaded_files": downloaded_files,
        "scanned_records": scanned_records,
        "output_path": str(output_path),
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
from fnmatch import fnmatch
import base64
from io import BytesIO
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def _parse_json_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _normalize_candidates(candidates):
    normalized = []
    for candidate in candidates:
        candidate_obj = _parse_json_field(candidate)
        normalized.append(candidate_obj)
    return normalized


def normalize_multimodal_sample(sample: dict) -> dict:
    normalized = dict(sample)
    normalized["operation"] = _parse_json_field(sample["operation"])
    normalized["pos_candidates"] = _normalize_candidates(sample["pos_candidates"])
    normalized["neg_candidates"] = _normalize_candidates(sample["neg_candidates"])
    action_reprs = sample.get("action_reprs", [])
    target_index = int(sample.get("target_action_index", -1))
    normalized["target_action_index"] = target_index
    normalized["previous_actions"] = action_reprs[:target_index] if target_index >= 0 else []
    return normalized


def attach_candidate_ranks(samples: list[dict], candidate_results: dict | None) -> list[dict]:
    if candidate_results is None:
        return samples

    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]

    for sample in samples:
        sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
        for group_name in ("pos_candidates", "neg_candidates"):
            for candidate in sample[group_name]:
                candidate_id = candidate["backend_node_id"]
                candidate["score"] = candidate_scores[sample_id][candidate_id]
                candidate["rank"] = candidate_ranks[sample_id][candidate_id]
    return samples


def _image_to_base64(image) -> str:
    image = image.convert("RGB")
    max_dim = 1024
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=75, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def image_to_chat_content(image) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{_image_to_base64(image)}",
        },
    }


def _collect_local_files(dataset_path: Path, split: str) -> list[str]:
    if split == "all":
        split_names = ["test_task", "test_website", "test_domain"]
    else:
        split_names = [split]

    files: list[str] = []
    data_dir = dataset_path / "data"
    for split_name in split_names:
        files.extend(str(path) for path in sorted(data_dir.glob(f"{split_name}-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found for split={split!r} under {data_dir}"
        )
    return files


def _match_filters(sample: dict, websites: set[str], domains: set[str]) -> bool:
    if websites and sample["website"] not in websites:
        return False
    if domains and sample["domain"] not in domains:
        return False
    return True


def _match_annotation_filters(
    sample: dict,
    annotation_ids: set[str],
    annotation_patterns: list[str],
    action_uids: set[str],
) -> bool:
    if annotation_ids and sample["annotation_id"] not in annotation_ids:
        return False
    if annotation_patterns and not any(
        fnmatch(sample["annotation_id"], pattern) for pattern in annotation_patterns
    ):
        return False
    if action_uids and sample["action_uid"] not in action_uids:
        return False
    return True


def load_multimodal_samples(
    dataset_path: str,
    split: str,
    limit: int | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    annotation_ids: Iterable[str] | None = None,
    annotation_patterns: Iterable[str] | None = None,
    action_uids: Iterable[str] | None = None,
    websites: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
) -> list[dict]:
    dataset_root = Path(dataset_path)
    data_files = _collect_local_files(dataset_root, split)
    dataset = load_dataset("parquet", data_files=data_files, split="train")

    annotation_id_set = {item for item in (annotation_ids or []) if item}
    annotation_pattern_list = [item for item in (annotation_patterns or []) if item]
    action_uid_set = {item for item in (action_uids or []) if item}
    website_set = {item for item in (websites or []) if item}
    domain_set = {item for item in (domains or []) if item}

    all_samples: list[dict] = []
    for row in dataset:
        all_samples.append(normalize_multimodal_sample(row))

    grouped_samples: dict[str, list[dict]] = {}
    for sample in all_samples:
        grouped_samples.setdefault(sample["annotation_id"], []).append(sample)
    for annotation_id, samples in grouped_samples.items():
        samples.sort(key=lambda item: item["target_action_index"])
        for idx, sample in enumerate(samples):
            sample["previous_step_records"] = [
                {
                    "annotation_id": previous_sample["annotation_id"],
                    "action_uid": previous_sample["action_uid"],
                    "target_action_index": previous_sample["target_action_index"],
                    "action_repr": previous_sample.get("target_action_reprs", ""),
                    "screenshot": previous_sample.get("screenshot"),
                }
                for previous_sample in samples[:idx]
            ]

    selected: list[dict] = []
    for row_idx, sample in enumerate(all_samples):
        if row_idx < start_index:
            continue
        if end_index is not None and row_idx >= end_index:
            break
        if not _match_filters(sample, website_set, domain_set):
            continue
        if not _match_annotation_filters(
            sample,
            annotation_id_set,
            annotation_pattern_list,
            action_uid_set,
        ):
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break
    return selected

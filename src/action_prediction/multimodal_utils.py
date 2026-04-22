import json
from fnmatch import fnmatch
import base64
from io import BytesIO
from pathlib import Path
from typing import Iterable
import math

from datasets import load_dataset
from PIL import Image


MIND2WEB_TEST_SPLITS = ("test_task", "test_website", "test_domain")
AITW_ACTION_TYPE_BY_ID = {
    3: "type",
    4: "dual_point",
    5: "go_back",
    6: "go_home",
    7: "enter",
    8: "task_complete",
    9: "task_impossible",
    10: "dual_point",
}


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
    normalized["dataset_format"] = "mind2web"
    normalized["action_space"] = "mind2web"
    normalized["operation"] = _parse_json_field(sample["operation"])
    normalized["pos_candidates"] = _normalize_candidates(sample["pos_candidates"])
    normalized["neg_candidates"] = _normalize_candidates(sample["neg_candidates"])
    action_reprs = sample.get("action_reprs", [])
    target_index = int(sample.get("target_action_index", -1))
    normalized["target_action_index"] = target_index
    normalized["previous_actions"] = action_reprs[:target_index] if target_index >= 0 else []
    return normalized


def _maybe_float_pair(value) -> list[float] | None:
    if value is None:
        return None
    value = _parse_json_field(value)
    if isinstance(value, dict):
        value = value.get("coordinates") or value.get("yx") or value.get("xy")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _normalize_aitw_image(value):
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"] is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if "path" in value and value["path"]:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        try:
            return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")
        except Exception:
            image_path = Path(value)
            if image_path.exists():
                return Image.open(image_path).convert("RGB")
    return value


def _normalize_aitw_action_type(sample: dict, typed_text: str, touch: list[float] | None, lift: list[float] | None) -> str:
    for key in ("action_type_text", "action_type", "type_text"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower().replace("-", "_").replace(" ", "_")

    raw_type = sample.get("results_action_type")
    if isinstance(raw_type, list) and len(raw_type) == 1:
        raw_type = raw_type[0]
    if isinstance(raw_type, str) and raw_type.strip():
        normalized = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized.isdigit():
            return normalized
        raw_type = int(normalized)
    if isinstance(raw_type, int):
        mapped = AITW_ACTION_TYPE_BY_ID.get(raw_type)
        if mapped is not None:
            return mapped

    if typed_text:
        return "type"
    if touch is not None or lift is not None:
        return "dual_point"
    return "unknown"


def _format_aitw_action_repr(action: dict) -> str:
    action_type = action["action_type"]
    typed_text = action.get("typed_text", "") or ""
    touch = action.get("touch_point")
    lift = action.get("lift_point")
    if action_type == "type":
        return f"TYPE text={typed_text!r}"
    if action_type == "dual_point":
        return f"DUAL_POINT touch={touch} lift={lift}"
    return action_type.upper()


def build_aitw_action_description(action: dict) -> str:
    action_type = (action.get("action_type") or "unknown").lower()
    typed_text = action.get("typed_text", "") or ""
    touch = action.get("touch_point")
    lift = action.get("lift_point")
    if action_type == "type":
        return f"type text={typed_text!r}"
    if action_type == "dual_point":
        return f"dual_point touch={touch} lift={lift}"
    return action_type


def parse_aitw_action_prediction(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    action_type = str(data.get("action_type") or "").strip().lower().replace("-", "_").replace(" ", "_")
    touch_point = _maybe_float_pair(data.get("touch_point"))
    lift_point = _maybe_float_pair(data.get("lift_point"))
    typed_text = str(data.get("typed_text") or "").strip()
    return {
        "action_type": action_type or "unknown",
        "touch_point": touch_point,
        "lift_point": lift_point,
        "typed_text": typed_text,
    }


def is_aitw_tap(action: dict) -> bool:
    if (action.get("action_type") or "").lower() != "dual_point":
        return False
    touch = action.get("touch_point")
    lift = action.get("lift_point")
    if touch is None or lift is None:
        return False
    return math.dist(touch, lift) <= 0.04


def aitw_action_match(predicted: dict, target: dict) -> float:
    pred_type = (predicted.get("action_type") or "").lower()
    target_type = (target.get("action_type") or "").lower()
    if pred_type != target_type:
        return 0.0

    if pred_type == "type":
        pred_text = (predicted.get("typed_text") or "").strip().lower()
        target_text = (target.get("typed_text") or "").strip().lower()
        return 1.0 if pred_text == target_text else 0.0

    if pred_type == "dual_point":
        pred_touch = predicted.get("touch_point")
        pred_lift = predicted.get("lift_point")
        target_touch = target.get("touch_point")
        target_lift = target.get("lift_point")
        if pred_touch is None or pred_lift is None or target_touch is None or target_lift is None:
            return 0.0
        pred_is_tap = is_aitw_tap(predicted)
        target_is_tap = is_aitw_tap(target)
        if pred_is_tap and target_is_tap:
            return 1.0 if math.dist(pred_touch, target_touch) <= 0.14 else 0.0
        if pred_is_tap != target_is_tap:
            return 0.0
        pred_delta_y = abs(pred_touch[0] - pred_lift[0])
        pred_delta_x = abs(pred_touch[1] - pred_lift[1])
        target_delta_y = abs(target_touch[0] - target_lift[0])
        target_delta_x = abs(target_touch[1] - target_lift[1])
        pred_axis = "vertical" if pred_delta_y >= pred_delta_x else "horizontal"
        target_axis = "vertical" if target_delta_y >= target_delta_x else "horizontal"
        return 1.0 if pred_axis == target_axis else 0.0

    return 1.0


def _summarize_ui_annotations(sample: dict, max_items: int = 20) -> list[str]:
    texts = sample.get("image_ui_annotations_text") or sample.get("ui_text") or []
    ui_types = sample.get("image_ui_annotations_ui_types") or sample.get("ui_types") or []
    items: list[str] = []
    for idx, text in enumerate(texts):
        text_value = str(text or "").strip().replace("\n", " ")
        type_value = str(ui_types[idx]).strip() if idx < len(ui_types) else ""
        if not text_value and not type_value:
            continue
        if len(text_value) > 80:
            text_value = text_value[:77] + "..."
        if text_value and type_value:
            items.append(f"{type_value}:{text_value}")
        else:
            items.append(text_value or type_value)
        if len(items) >= max_items:
            break
    return items


def normalize_aitw_sample(sample: dict, source_path: Path | None = None) -> dict:
    typed_text_raw = sample.get("type_text")
    if typed_text_raw is None:
        typed_text_raw = sample.get("typed_text")
    if typed_text_raw is None:
        typed_text_raw = sample.get("results_type_action")
    typed_text = ""
    if isinstance(typed_text_raw, list):
        typed_text = str(typed_text_raw[0] or "").strip() if typed_text_raw else ""
    elif typed_text_raw is not None:
        typed_text = str(typed_text_raw).strip()

    touch_point = _maybe_float_pair(sample.get("touch_coord"))
    if touch_point is None:
        touch_point = _maybe_float_pair(sample.get("results_yx_touch"))
    lift_point = _maybe_float_pair(sample.get("lift_coord"))
    if lift_point is None:
        lift_point = _maybe_float_pair(sample.get("results_yx_lift"))

    action_type = _normalize_aitw_action_type(sample, typed_text, touch_point, lift_point)
    action = {
        "action_type": action_type,
        "touch_point": touch_point,
        "lift_point": lift_point,
        "typed_text": typed_text,
    }
    target_action_repr = _format_aitw_action_repr(action)
    episode_id = str(sample.get("ep_id") or sample.get("episode_id") or sample.get("annotation_id") or "")
    step_id = int(sample.get("step_id", sample.get("target_action_index", 0)))
    current_activity = str(sample.get("current_activity") or sample.get("activity") or "")
    subset_name = ""
    if source_path is not None:
        lower_parts = [part.lower() for part in source_path.parts]
        for candidate in ("general", "googleapps", "install", "single", "webshopping"):
            if candidate in lower_parts:
                subset_name = candidate
                break
    website = current_activity or subset_name or "android"
    ui_elements = _summarize_ui_annotations(sample)
    screenshot = _normalize_aitw_image(sample.get("image"))
    if screenshot is None:
        screenshot = _normalize_aitw_image(sample.get("image_encoded"))

    normalized = {
        "dataset_format": "aitw",
        "action_space": "aitw",
        "annotation_id": episode_id,
        "action_uid": str(sample.get("action_uid") or f"{episode_id}_{step_id}"),
        "confirmed_task": str(sample.get("goal_info") or sample.get("goal") or sample.get("confirmed_task") or ""),
        "website": website,
        "domain": subset_name or "android",
        "subdomain": str(sample.get("device_type") or sample.get("android_api_level") or ""),
        "target_action_index": step_id,
        "target_action_reprs": target_action_repr,
        "operation": {
            "op": action_type.upper(),
            "value": typed_text,
        },
        "ground_truth_action": action,
        "pos_candidates": [],
        "neg_candidates": [],
        "cleaned_html": "",
        "raw_html": "",
        "screenshot": screenshot,
        "ui_elements": ui_elements,
        "ui_annotation_positions": sample.get("image_ui_annotations_positions") or [],
        "current_activity": current_activity,
        "android_api_level": sample.get("android_api_level"),
        "device_type": sample.get("device_type"),
        "episode_length": sample.get("episode_length"),
        "previous_actions": [],
    }
    return normalized


def attach_candidate_ranks(samples: list[dict], candidate_results: dict | None) -> list[dict]:
    if candidate_results is None:
        for sample in samples:
            if sample.get("action_space") != "mind2web":
                continue
            for group_name in ("pos_candidates", "neg_candidates"):
                for idx, candidate in enumerate(sample.get(group_name, [])):
                    candidate.setdefault("score", 0.0)
                    candidate.setdefault("rank", idx)
        return samples

    candidate_scores = candidate_results["scores"]
    candidate_ranks = candidate_results["ranks"]

    for sample in samples:
        if sample.get("action_space") != "mind2web":
            continue
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


def detect_dataset_format(dataset_path: str) -> str:
    dataset_root = Path(dataset_path)
    data_dir = dataset_root / "data"
    if data_dir.exists() and any(data_dir.glob("test_task-*.parquet")):
        return "mind2web"
    if data_dir.exists() and any(data_dir.glob("test_website-*.parquet")):
        return "mind2web"
    lower_path = dataset_root.as_posix().lower()
    if "mind2web" in lower_path:
        return "mind2web"
    if "aitw" in lower_path or "android_in_the_wild" in lower_path:
        return "aitw"
    sample_files = list(dataset_root.rglob("*.parquet"))[:5]
    if not sample_files:
        sample_files = list(dataset_root.rglob("*.json"))[:5] + list(dataset_root.rglob("*.jsonl"))[:5]
    for file_path in sample_files:
        lowered = file_path.as_posix().lower()
        if "mind2web" in lowered:
            return "mind2web"
        if "aitw" in lowered or "android_in_the_wild" in lowered:
            return "aitw"
    return "mind2web"


def _collect_mind2web_files(dataset_path: Path, split: str) -> list[str]:
    split_names = list(MIND2WEB_TEST_SPLITS) if split == "all" else [split]
    files: list[str] = []
    data_dir = dataset_path / "data"
    for split_name in split_names:
        files.extend(str(path) for path in sorted(data_dir.glob(f"{split_name}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={split!r} under {data_dir}")
    return files


def _collect_aitw_files(dataset_path: Path, split: str) -> list[str]:
    all_files = [
        path for path in sorted(dataset_path.rglob("*.parquet"))
        if ".meta." not in path.name
    ]
    if not all_files:
        all_files = [
            path for path in (sorted(dataset_path.rglob("*.json")) + sorted(dataset_path.rglob("*.jsonl")))
            if ".meta." not in path.name
        ]
    if not all_files:
        raise FileNotFoundError(f"No parquet/json/jsonl files found under {dataset_path}")
    if split == "all":
        return [str(path) for path in all_files]

    selected = []
    split_lower = split.lower()
    split_tokens = {split_lower, f"/{split_lower}/", f"{split_lower}-", f"{split_lower}_"}
    for path in all_files:
        normalized = path.as_posix().lower()
        if any(token in normalized for token in split_tokens):
            selected.append(str(path))
    if selected:
        return selected
    return [str(path) for path in all_files]


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
    sample_annotation_id = str(sample.get("annotation_id", ""))
    sample_action_uid = str(sample.get("action_uid", ""))
    if annotation_ids and sample_annotation_id not in annotation_ids:
        return False
    if annotation_patterns and not any(
        fnmatch(sample_annotation_id, pattern) for pattern in annotation_patterns
    ):
        return False
    if action_uids and sample_action_uid not in action_uids:
        return False
    return True


def _build_previous_step_records(samples: list[dict]) -> None:
    grouped_samples: dict[str, list[dict]] = {}
    for sample in samples:
        grouped_samples.setdefault(sample["annotation_id"], []).append(sample)
    for _, episode_samples in grouped_samples.items():
        episode_samples.sort(key=lambda item: int(item.get("target_action_index", -1)))
        history: list[str] = []
        for idx, sample in enumerate(episode_samples):
            sample["previous_actions"] = history[:]
            sample["previous_step_records"] = [
                {
                    "annotation_id": previous_sample["annotation_id"],
                    "action_uid": previous_sample["action_uid"],
                    "target_action_index": previous_sample["target_action_index"],
                    "action_repr": previous_sample.get("target_action_reprs", ""),
                    "screenshot": previous_sample.get("screenshot"),
                }
                for previous_sample in episode_samples[:idx]
            ]
            history.append(sample.get("target_action_reprs", ""))


def _load_dataset_rows(data_files: list[str], file_format: str) -> list[dict]:
    dataset = load_dataset(file_format, data_files=data_files, split="train")
    return [dict(row) for row in dataset]


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
    dataset_format: str = "auto",
) -> list[dict]:
    dataset_root = Path(dataset_path)
    if dataset_format == "auto":
        dataset_format = detect_dataset_format(dataset_path)

    annotation_id_set = {item for item in (annotation_ids or []) if item}
    annotation_pattern_list = [item for item in (annotation_patterns or []) if item]
    action_uid_set = {item for item in (action_uids or []) if item}
    website_set = {item for item in (websites or []) if item}
    domain_set = {item for item in (domains or []) if item}

    all_samples: list[dict] = []
    if dataset_format == "mind2web":
        data_files = _collect_mind2web_files(dataset_root, split)
        for row in _load_dataset_rows(data_files, "parquet"):
            all_samples.append(normalize_multimodal_sample(row))
    elif dataset_format == "aitw":
        data_files = _collect_aitw_files(dataset_root, split)
        for file_path in data_files:
            file_format = "parquet" if file_path.endswith(".parquet") else "json"
            for row in _load_dataset_rows([file_path], file_format):
                all_samples.append(normalize_aitw_sample(row, source_path=Path(file_path)))
    else:
        raise ValueError(f"Unsupported dataset_format={dataset_format!r}")

    _build_previous_step_records(all_samples)

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

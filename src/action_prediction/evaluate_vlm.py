import argparse
import collections
import copy
import json
import logging
import os
import pickle
import random
import time

import requests
from requests import RequestException
from metric import ActionEvaluatorMultiChoice
from multimodal_utils import (
    attach_candidate_ranks,
    image_to_chat_content,
    load_multimodal_samples,
)

from dataloader import format_input_multichoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class StaticDataset:
    def __init__(self, samples, max_context_len=512):
        self.data = samples
        self.max_context_len = max_context_len


class OpenAICompatibleVLMEngine:
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        rate_limit: int = -1,
        temperature: float = 0,
        timeout: int = 300,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_available_time = 0.0
        self.max_retries = 8

    def _wait_for_slot(self) -> None:
        if self.request_interval <= 0:
            return
        now = time.time()
        if now < self.next_available_time:
            time.sleep(self.next_available_time - now)
        self.next_available_time = max(now, self.next_available_time) + self.request_interval

    def generate(self, prompt, max_new_tokens=50, image=None, history_images=None, history_text=None, **kwargs):
        self._wait_for_slot()
        payload_messages = copy.deepcopy(prompt)
        content_blocks = []
        base_text = payload_messages[-1]["content"]
        if history_text:
            base_text = f"{history_text}\n\n{base_text}"
        content_blocks.append({"type": "text", "text": base_text})
        for history_image in history_images or []:
            content_blocks.append(image_to_chat_content(history_image))
        if image is not None:
            content_blocks.append(image_to_chat_content(image))
        if len(content_blocks) > 1:
            payload_messages[-1]["content"] = content_blocks
        else:
            payload_messages[-1]["content"] = base_text

        payload = {
            "model": self.model,
            "messages": payload_messages,
            "max_tokens": max_new_tokens,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        last_error = None
        for attempt_idx in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return [choice["message"]["content"] for choice in data["choices"]]
            except RequestException as exc:
                last_error = exc
                if attempt_idx == self.max_retries - 1:
                    raise
                sleep_seconds = 2 ** attempt_idx
                logger.warning(
                    "OpenAI-compatible request failed on attempt %s/%s: %s. Retrying in %ss.",
                    attempt_idx + 1,
                    self.max_retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)
        raise last_error


class VLMActionEvaluator(ActionEvaluatorMultiChoice):
    @staticmethod
    def build_manual_compressed_history(
        sample,
        history_compression_map,
        history_text_char_budget,
    ):
        sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
        config = history_compression_map.get(sample_id, {})
        previous_steps = sample.get("previous_step_records", [])
        kept_step_indices = config.get("kept_step_indices", [])
        kept_steps = []
        for idx in kept_step_indices:
            if 1 <= idx <= len(previous_steps):
                kept_steps.append(previous_steps[idx - 1])

        summary = config.get("summary", "").strip()
        if not summary:
            return "", kept_steps

        lines = [
            "Compressed full-trajectory history for the same task:",
            f"Task: {sample['confirmed_task']}",
            summary,
            "The kept history images are the visually important states from earlier in the trajectory.",
        ]
        history_text = "\n".join(lines)
        if history_text_char_budget and len(history_text) > history_text_char_budget:
            history_text = history_text[:history_text_char_budget].rstrip() + "\n[history truncated]"
        return history_text, kept_steps

    @staticmethod
    def build_manual_compressed_plus_recent_history(
        sample,
        history_compression_map,
        history_text_char_budget,
        recent_k,
    ):
        sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
        config = history_compression_map.get(sample_id, {})
        previous_steps = sample.get("previous_step_records", [])
        if not previous_steps:
            return "", []

        recent_steps = previous_steps[-recent_k:] if recent_k > 0 else []
        early_steps = previous_steps[:-recent_k] if recent_k > 0 else previous_steps
        kept_step_indices = config.get("kept_step_indices", [])

        early_kept_steps = []
        for idx in kept_step_indices:
            if 1 <= idx <= len(early_steps):
                early_kept_steps.append(early_steps[idx - 1])

        history_images = early_kept_steps + recent_steps

        summary = config.get("summary", "").strip()
        if not summary:
            return "", history_images

        lines = [
            "Compressed early trajectory history for the same task:",
            f"Task: {sample['confirmed_task']}",
            summary,
            f"The final {recent_k} earlier screenshots are preserved in full as the recent local interaction context.",
            "Any additional kept history images are visually important earlier states from the compressed part of the trajectory.",
        ]
        history_text = "\n".join(lines)
        if history_text_char_budget and len(history_text) > history_text_char_budget:
            history_text = history_text[:history_text_char_budget].rstrip() + "\n[history truncated]"
        return history_text, history_images

    @staticmethod
    def select_previous_steps(sample, history_mode, recent_k):
        previous_steps = sample.get("previous_step_records", [])
        if not previous_steps:
            return []
        if history_mode in {
            "recent",
            "recent_anchor_summary",
            "manual_summary_recent",
            "manual_compressed_plus_recent",
        }:
            return previous_steps[-recent_k:] if recent_k > 0 else []
        return previous_steps

    @staticmethod
    def build_history_text(sample, previous_steps, history_text_char_budget):
        if not previous_steps:
            return ""

        lines = [
            "Previous trajectory context for the same task:",
            f"Task: {sample['confirmed_task']}",
            "These are earlier steps that happened before the current decision.",
        ]
        for idx, previous_step in enumerate(previous_steps, start=1):
            action_repr = previous_step.get("action_repr") or "(missing action repr)"
            lines.append(f"Step {idx}: {action_repr}")
        history_text = "\n".join(lines)
        if history_text_char_budget and len(history_text) > history_text_char_budget:
            history_text = history_text[:history_text_char_budget].rstrip() + "\n[history truncated]"
        return history_text

    @staticmethod
    def build_change_summary_text(sample, previous_steps, history_text_char_budget):
        if not previous_steps:
            return ""

        lines = [
            "Trajectory change summary for the same task:",
            f"Task: {sample['confirmed_task']}",
            "The anchor image is the earliest screenshot in the recent history window.",
            "The current image is after the following state changes:",
        ]
        for idx, previous_step in enumerate(previous_steps, start=1):
            action_repr = previous_step.get("action_repr") or "(missing action repr)"
            lines.append(f"Change {idx}: {action_repr}")
        lines.append("Use the anchor image to understand where this local interaction sequence started.")
        history_text = "\n".join(lines)
        if history_text_char_budget and len(history_text) > history_text_char_budget:
            history_text = history_text[:history_text_char_budget].rstrip() + "\n[history truncated]"
        return history_text

    def evaluate_dataset_vlm(
        self,
        dataset,
        model,
        prompt_template,
        top_k=50,
        output_path=None,
        name="default",
        use_image=True,
        history_mode="none",
        history_text_char_budget=24000,
        recent_k=3,
        history_summary_map=None,
        history_compression_map=None,
    ):
        all_element_acc = []
        all_action_f1 = []
        all_step_acc = []
        sample_to_website = {}
        all_final_predictions = []
        all_outputs = []
        for k in [5, 10, 20, 50]:
            recall_at_k = sum(
                1 if any(c["rank"] < k for c in sample["pos_candidates"]) else 0
                for sample in dataset.data
            ) / max(len(dataset.data), 1)
            logger.info("Recall Cap @ %s: %.4f", k, recall_at_k)
        acc = sum(
            1 if any(c["rank"] == 0 for c in sample["pos_candidates"]) else 0
            for sample in dataset.data
        ) / max(len(dataset.data), 1)
        logger.info("Candidate generator acc: %.4f", acc)

        for sample in dataset.data:
            annotation_id = sample["annotation_id"]
            sample_to_website[annotation_id] = sample["website"]

            pos_candidates = [c for c in sample["pos_candidates"] if c["rank"] < top_k]
            pos_ids = [c["backend_node_id"] for c in pos_candidates]
            if len(pos_ids) == 0:
                all_element_acc.append([0, annotation_id])
                all_action_f1.append([0, annotation_id])
                all_step_acc.append([0, annotation_id])
                all_final_predictions.append(
                    [f"{sample['annotation_id']}_{sample['action_uid']}", "", ""]
                )
                all_outputs.append(
                    [f"{sample['annotation_id']}_{sample['action_uid']}", []]
                )
                continue

            _, _, target_out, _ = format_input_multichoice(sample, pos_ids[:1], pos_ids[0])
            _, target_action = self.postprocess_action(target_out)
            neg_candidates = [c for c in sample["neg_candidates"] if c["rank"] < top_k]
            neg_ids = [c["backend_node_id"] for c in neg_candidates]
            all_candidates = pos_ids + neg_ids
            random.shuffle(all_candidates)
            final_prediction = None
            outputs = []

            while len(all_candidates) > 1:
                candidate_ids = all_candidates[:5]
                all_candidates = all_candidates[5:]
                seq_context, seq_in, _, choices = format_input_multichoice(
                    sample, candidate_ids, -1, keep_html_brackets=True
                )
                prompt = copy.deepcopy(prompt_template)
                prompt[-1]["content"] = f"'''\n{seq_context}\n'''\n\n{seq_in}"
                history_images = []
                history_text = ""
                selected_previous_steps = self.select_previous_steps(sample, history_mode, recent_k)
                if history_mode != "none":
                    if history_mode == "manual_compressed_full":
                        history_text, selected_previous_steps = self.build_manual_compressed_history(
                            sample,
                            history_compression_map or {},
                            history_text_char_budget,
                        )
                    elif history_mode == "manual_compressed_plus_recent":
                        history_text, history_images = self.build_manual_compressed_plus_recent_history(
                            sample,
                            history_compression_map or {},
                            history_text_char_budget,
                            recent_k,
                        )
                        selected_previous_steps = history_images
                    elif history_mode == "recent_anchor_summary":
                        history_text = self.build_change_summary_text(
                            sample, selected_previous_steps, history_text_char_budget
                        )
                    elif history_mode == "manual_summary_anchor":
                        sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
                        history_text = history_summary_map.get(sample_id, "")
                    elif history_mode == "manual_summary_recent":
                        sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
                        history_text = history_summary_map.get(sample_id, "")
                    else:
                        history_text = self.build_history_text(
                            sample, selected_previous_steps, history_text_char_budget
                        )
                    if use_image and history_mode == "full":
                        history_images = [
                            step["screenshot"]
                            for step in selected_previous_steps
                            if step.get("screenshot") is not None
                        ]
                    elif use_image and history_mode == "recent_anchor_summary":
                        anchor_candidates = [
                            step["screenshot"]
                            for step in selected_previous_steps
                            if step.get("screenshot") is not None
                        ]
                        history_images = anchor_candidates[:1]
                    elif use_image and history_mode == "manual_summary_anchor":
                        anchor_candidates = [
                            step["screenshot"]
                            for step in selected_previous_steps
                            if step.get("screenshot") is not None
                        ]
                        history_images = anchor_candidates[:1]
                    elif use_image and history_mode == "manual_summary_recent":
                        history_images = [
                            step["screenshot"]
                            for step in selected_previous_steps
                            if step.get("screenshot") is not None
                        ]
                    elif use_image and history_mode == "manual_compressed_full":
                        history_images = [
                            step["screenshot"]
                            for step in selected_previous_steps
                            if step.get("screenshot") is not None
                        ]
                    elif use_image and history_mode == "manual_compressed_plus_recent":
                        history_images = [
                            step["screenshot"]
                            for step in selected_previous_steps
                            if step.get("screenshot") is not None
                        ]
                outputs.append(
                    [
                        candidate_ids,
                        [seq_context, seq_in, choices],
                        {
                            "history_mode": history_mode,
                            "history_text": history_text,
                            "history_image_count": len(history_images),
                            "selected_previous_steps": len(selected_previous_steps),
                        },
                        None,
                    ]
                )
                output = model.generate(
                    prompt=prompt,
                    max_new_tokens=50,
                    image=sample["screenshot"] if use_image else None,
                    history_images=history_images,
                    history_text=history_text,
                )
                outputs[-1][-1] = output[0]
                pred_element, pred_action = self.postprocess_action_llm(output[0])
                if pred_element[0] != "A":
                    pred_element = ord(pred_element[0]) - ord("B")
                    try:
                        pred_element = choices[pred_element][0]
                        all_candidates.append(pred_element)
                        final_prediction = (pred_element, pred_action)
                    except IndexError:
                        logger.info("IndexError for output=%s", output[0])
                        final_prediction = None

            all_outputs.append([f"{sample['annotation_id']}_{sample['action_uid']}", outputs])
            if len(all_candidates) == 0 or final_prediction is None:
                all_element_acc.append([0, annotation_id])
                all_action_f1.append([0, annotation_id])
                all_step_acc.append([0, annotation_id])
                all_final_predictions.append(
                    [f"{sample['annotation_id']}_{sample['action_uid']}", "", ""]
                )
            else:
                all_element_acc.append([1 if final_prediction[0] in pos_ids else 0, annotation_id])
                all_action_f1.append(
                    [self.calculate_f1(final_prediction[1], target_action), annotation_id]
                )
                all_step_acc.append(
                    [
                        1
                        if all_action_f1[-1][0] == 1 and all_element_acc[-1][0] == 1
                        else 0,
                        annotation_id,
                    ]
                )
                all_final_predictions.append(
                    [
                        f"{sample['annotation_id']}_{sample['action_uid']}",
                        final_prediction[0],
                        final_prediction[1],
                    ]
                )

        macro_element_acc = collections.defaultdict(list)
        macro_action_f1 = collections.defaultdict(list)
        macro_step_acc = collections.defaultdict(list)
        for score, annotation_id in all_element_acc:
            macro_element_acc[annotation_id].append(score)
        for score, annotation_id in all_action_f1:
            macro_action_f1[annotation_id].append(score)
        for score, annotation_id in all_step_acc:
            macro_step_acc[annotation_id].append(score)

        error_ratio = collections.defaultdict(int)
        acc_per_website = collections.defaultdict(list)
        for annotation_id, values in macro_step_acc.items():
            acc_per_website[sample_to_website[annotation_id]].append(sum(values) / len(values))
            error_count = len([value for value in values if value == 0])
            if error_count <= 3:
                error_ratio[error_count] += 1
            else:
                error_ratio[">3"] += 1
        if macro_element_acc:
            error_ratio = {k: v / len(macro_element_acc) for k, v in error_ratio.items()}
            acc_per_website = {k: (sum(v) / len(v), len(v)) for k, v in acc_per_website.items()}
            macro_element_acc_value = sum(sum(v) / len(v) for v in macro_element_acc.values()) / len(
                macro_element_acc
            )
            macro_action_f1_value = sum(sum(v) / len(v) for v in macro_action_f1.values()) / len(
                macro_action_f1
            )
            macro_step_acc_value = sum(sum(v) / len(v) for v in macro_step_acc.values()) / len(
                macro_step_acc
            )
        else:
            acc_per_website = {}
            macro_element_acc_value = 0.0
            macro_action_f1_value = 0.0
            macro_step_acc_value = 0.0
            error_ratio = {}

        result = {
            "element_acc": sum(x[0] for x in all_element_acc) / max(len(all_element_acc), 1),
            "action_f1": sum(x[0] for x in all_action_f1) / max(len(all_action_f1), 1),
            "step_acc": sum(x[0] for x in all_step_acc) / max(len(all_step_acc), 1),
            "marco_element_acc": macro_element_acc_value,
            "marco_action_f1": macro_action_f1_value,
            "marco_step_acc": macro_step_acc_value,
            "error_ratio": error_ratio,
            "acc_per_website": acc_per_website,
        }
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/{name}_predictions_top{top_k}.json", "w", encoding="utf-8") as f:
                json.dump(all_final_predictions, f)
            with open(f"{output_path}/{name}_results_top{top_k}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            with open(f"{output_path}/{name}_outputs_top{top_k}.json", "w", encoding="utf-8") as f:
                json.dump(all_outputs, f)
        return result


def load_prompt_template(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_annotation_ids(annotation_id_file: str | None) -> list[str]:
    if not annotation_id_file:
        return []
    with open(annotation_id_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_ids_from_file(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_history_summary_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {item["sample_id"]: item["summary"] for item in data}


def load_history_compression_map(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {item["sample_id"]: item for item in data}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an OpenAI-compatible VLM on static Multimodal-Mind2Web using the official Mind2Web action metric."
    )
    parser.add_argument("--dataset-path", default="data/multimodal_mind2web")
    parser.add_argument("--score-file", default="data/mind2web_aux/scores_all_data.pkl")
    parser.add_argument("--split", default="test_task", choices=["test_task", "test_website", "test_domain", "all"])
    parser.add_argument("--output-dir", default="outputs/mind2web_vlm")
    parser.add_argument("--prompt-file", default="src/action_prediction/llm_prompt.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rate-limit", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--annotation-id", action="append", default=[])
    parser.add_argument("--annotation-id-file")
    parser.add_argument("--annotation-pattern", action="append", default=[])
    parser.add_argument("--action-uid", action="append", default=[])
    parser.add_argument("--action-uid-file")
    parser.add_argument("--website", action="append", default=[])
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--html-only", action="store_true")
    parser.add_argument(
        "--history-mode",
        default="none",
        choices=[
            "none",
            "text_only",
            "full",
            "recent",
            "recent_anchor_summary",
            "manual_summary_anchor",
            "manual_summary_recent",
            "manual_compressed_full",
            "manual_compressed_plus_recent",
        ],
    )
    parser.add_argument("--history-text-char-budget", type=int, default=24000)
    parser.add_argument("--recent-k", type=int, default=3)
    parser.add_argument("--history-summary-file")
    parser.add_argument("--history-compression-file")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    annotation_ids = args.annotation_id + load_annotation_ids(args.annotation_id_file)
    action_uids = args.action_uid + load_ids_from_file(args.action_uid_file)
    history_summary_map = load_history_summary_map(args.history_summary_file)
    history_compression_map = load_history_compression_map(args.history_compression_file)
    samples = load_multimodal_samples(
        dataset_path=args.dataset_path,
        split=args.split,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index,
        annotation_ids=annotation_ids,
        annotation_patterns=args.annotation_pattern,
        action_uids=action_uids,
        websites=args.website,
        domains=args.domain,
    )
    logger.info("Selected %s action samples from split=%s", len(samples), args.split)
    if samples:
        logger.info(
            "First sample: annotation_id=%s action_uid=%s website=%s",
            samples[0]["annotation_id"],
            samples[0]["action_uid"],
            samples[0]["website"],
        )

    if args.dry_run:
        return

    with open(args.score_file, "rb") as f:
        candidate_results = pickle.load(f)
    samples = attach_candidate_ranks(samples, candidate_results)

    prompt_template = load_prompt_template(args.prompt_file)
    dataset = StaticDataset(samples)
    model = OpenAICompatibleVLMEngine(
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
        temperature=args.temperature,
    )
    evaluator = VLMActionEvaluator(tokenizer=None)
    split_name = args.split if args.split != "all" else "mixed_test"
    results = evaluator.evaluate_dataset_vlm(
        dataset=dataset,
        model=model,
        prompt_template=prompt_template,
        top_k=args.top_k,
        output_path=args.output_dir,
        name=split_name,
        use_image=not args.html_only,
        history_mode=args.history_mode,
        history_text_char_budget=args.history_text_char_budget,
        recent_k=args.recent_k,
        history_summary_map=history_summary_map,
        history_compression_map=history_compression_map,
    )
    logger.info("Results: %s", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import collections
import copy
import json
import logging
import os
import pickle
import random
import re
import time

import requests
from dotenv import load_dotenv
from requests import RequestException

from dataloader import format_input_multichoice
from metric import ActionEvaluatorMultiChoice
from multimodal_utils import (
    aitw_action_match,
    attach_candidate_ranks,
    build_aitw_action_description,
    image_to_chat_content,
    load_multimodal_samples,
    parse_aitw_action_prediction,
)

load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


AITW_TASK_AGENT_SYSTEM = """You are the task agent in a GUI agentic system for Android in the Wild.
Predict the next mobile action from the screenshot, task, and rolling memory.

Output strict JSON only:
{
  "action_type": "dual_point" | "type" | "go_back" | "go_home" | "enter" | "task_complete" | "task_impossible",
  "touch_point": [y, x] | null,
  "lift_point": [y, x] | null,
  "typed_text": "<text or empty string>"
}

Rules:
- Coordinates are normalized [0, 1].
- Use `dual_point` for taps and drags/swipes.
- For non-gesture actions set touch_point and lift_point to null.
- Output JSON only. No markdown fences. No explanation.
"""


MEMORY_AGENT_SYSTEM = """You are the memory agent in a GUI agentic system.
Your job is to update rolling interaction memory online.

Rules:
- Compare only the previous screenshot and the current screenshot for the newest step.
- Use the provided action string only as auxiliary context.
- First emit one newest item as either `recent_change` or `recent_change_keyframe`.
- A `recent_change` is for a small local update on the same working screen.
- A `recent_change_keyframe` is for a major state transition such as page jump, modal/dialog open, checkout/payment transition, or a clear interaction-focus shift.
- The memory must preserve the exact interaction primitive, not only a high-level workflow story.
- Always ground the update in the acted-on element and the action type.
- If a value was typed or selected, copy that value in a short normalized form.
- Always state what interaction focus the UI is in after this step, and what the next likely local goal is.
- Interaction focus means the current sub-flow or active workspace, such as `search form`, `guest modal`, `results list`, `truck options`, `location continuation`, `add-ons`, `checkout form`, or `payment`.
- Keep the change description extremely short and grounded in visible UI changes.
- Do not narrate the whole task.
- Do not speculate about hidden state.
- Do not replace precise controls with vague summaries like `continued checkout` or `moved forward` if a more specific element or sub-flow is visible.
- If the UI focus has shifted away from one branch to another, say that explicitly in `focus_after` or `next_goal`.

Output strict JSON only with this schema:
{
  "item_type": "recent_change" | "recent_change_keyframe",
  "change": "<very short visible state change>",
  "element": "<very short acted-on element description>",
  "action_type": "CLICK" | "SELECT" | "TYPE" | "DUAL_POINT" | "GO_BACK" | "GO_HOME" | "ENTER" | "TASK_COMPLETE" | "TASK_IMPOSSIBLE" | "",
  "action_value": "<typed/selected value or empty string>",
  "focus_after": "<current active UI sub-flow after this step>",
  "next_goal": "<the next immediate local target implied by the UI, not the whole task>",
  "keep_image": true | false,
  "action": "<copy the provided action string only if item_type is recent_change_keyframe, else empty string>"
}

Good style examples:
- `"change": "truck options appear with pricing", "element": "Find Your Truck button", "action_type": "CLICK", "focus_after": "truck options", "next_goal": "choose the correct truck card"`
- `"change": "email field becomes filled", "element": "email input", "action_type": "TYPE", "action_value": "jame_jones@hotmail.com", "focus_after": "checkout form", "next_goal": "fill the next required input"`
- `"change": "location step becomes active", "element": "Continue to Location button", "action_type": "CLICK", "focus_after": "location continuation", "next_goal": "continue within location flow, not add-ons"`
- `"change": "screen scrolls down to more settings", "element": "settings list", "action_type": "DUAL_POINT", "focus_after": "settings list", "next_goal": "tap the correct settings row"`
- `"change": "previous screen returns", "element": "system back", "action_type": "GO_BACK", "focus_after": "previous page", "next_goal": "continue from the previous screen"`

Bad style examples:
- `"change": "the user moved further in the flow"`
- `"change": "checkout continues"`
- `"change": "the task is almost done"`

Do not wrap JSON in markdown fences.
"""


MEMORY_AGENT_USER_TEMPLATE = """Task:
{task}

Current memory before this update:
[older_summary]
{older_summary}

[recent_buffer]
{recent_buffer}

Newest action:
{action_repr}

Compare the two screenshots and update memory for only this newest transition."""


SUMMARIZER_SYSTEM = """You are the summarizer for old GUI interaction memory.
You compress older recent-memory items into a single older_summary.

Rules:
- Summarize only the older prefix, not the newest tail.
- Keep it concise and high-level.
- Preserve major page or workflow transitions.
- Merge repeated small local changes aggressively.
- Mention keyframe transitions at a coarse level.
- Do not narrate every step.
- Do not drop the interaction anchors that are necessary for future action grounding.
- Keep explicit references to:
  - key acted-on elements when they define the current branch,
  - exact action types when they matter,
  - important typed/selected values,
  - interaction-focus shifts such as `guest modal -> results`, `truck options -> location continuation`, or `payment options -> personal info form`.
- Prefer short structured prose over vague storytelling.
- If an older prefix establishes that one branch is active and another branch is no longer active, retain that distinction.

Output strict JSON only:
{
  "older_summary": "<compressed older summary>"
}
Do not wrap JSON in markdown fences.
"""


SUMMARIZER_USER_TEMPLATE = """Task:
{task}

Existing older summary:
{older_summary}

Older recent-memory prefix to compress:
{prefix_text}
"""


TASK_AGENT_SYSTEM = """You are the task agent in a GUI agentic system.
You receive the current screenshot, the current task, and rolling memory from a memory agent.
Use the memory only to understand task progress and current interaction state.
Then solve the candidate action selection task.

Follow the answer format expected by the benchmark exactly."""

TASK_AGENT_FORMAT_SUFFIX = """

You must answer in exactly one of these formats:
- Answer: A.
- Answer: B.
  Action: CLICK
- Answer: C.
  Action: SELECT
  Value: <value>
- Answer: D.
  Action: TYPE
  Value: <value>

Rules:
- Always start with `Answer: <letter>.`
- The letter must be one of A, B, C, D, E.
- If the answer is not A, include `Action: ...`
- Include `Value: ...` only when needed.
- Do not explain your choice.
"""


class StaticDataset:
    def __init__(self, samples):
        self.data = samples


class OpenAICompatibleEngine:
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str,
        rate_limit: int = -1,
        temperature: float = 0.0,
        timeout: int = 300,
        max_retries: int = 8,
    ) -> None:
        if not api_key:
            raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_available_time = 0.0

    def _wait_for_slot(self) -> None:
        if self.request_interval <= 0:
            return
        now = time.time()
        if now < self.next_available_time:
            time.sleep(self.next_available_time - now)
        self.next_available_time = max(now, self.next_available_time) + self.request_interval

    def chat(self, messages, max_tokens=300, temperature=None):
        self._wait_for_slot()
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
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
                message = data["choices"][0]["message"]
                content = message.get("content", "")
                if content:
                    return content
                reasoning_content = message.get("reasoning_content", "")
                if reasoning_content:
                    return reasoning_content
                return ""
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

    def chat_with_images(self, system_prompt, user_text, current_image, previous_image=None, max_tokens=300):
        if previous_image is not None:
            content = [image_to_chat_content(previous_image)]
        else:
            content = []
        if current_image is not None:
            content.append(image_to_chat_content(current_image))
        content.append({"type": "text", "text": user_text})
        return self.chat_with_image_list(system_prompt, content, max_tokens=max_tokens)

    def chat_with_image_list(self, system_prompt, content_blocks, max_tokens=300):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_blocks},
        ]
        return self.chat(messages, max_tokens=max_tokens)


def safe_json_load(text: str):
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("` \n")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def fallback_memory_item(action_repr: str):
    return {
        "item_type": "recent_change",
        "change": "the interface updates locally on the same screen.",
        "element": "",
        "action_type": "",
        "action_value": "",
        "focus_after": "same local screen",
        "next_goal": "continue the current local interaction",
        "keep_image": False,
        "action": "",
    }


def fallback_older_summary(older_summary: str, prefix: list[dict]) -> str:
    prefix_lines = [item.get("change", "").strip() for item in prefix if item.get("change")]
    merged = " ".join(prefix_lines[:3]).strip()
    if older_summary and merged:
        return f"{older_summary} {merged}".strip()
    if older_summary:
        return older_summary
    if merged:
        return merged
    return "Earlier interaction history contains several local state updates in the same task flow."


def render_recent_item(item: dict) -> str:
    parts = [f"[{item['item_type']}] {item['change']}"]
    if item.get("element"):
        parts.append(f"element: {item['element']}")
    if item.get("action_type"):
        parts.append(f"action_type: {item['action_type']}")
    if item.get("action_value"):
        parts.append(f"action_value: {item['action_value']}")
    if item.get("focus_after"):
        parts.append(f"focus_after: {item['focus_after']}")
    if item.get("next_goal"):
        parts.append(f"next_goal: {item['next_goal']}")
    if item["item_type"] == "recent_change_keyframe" and item.get("action"):
        parts.append(f"action: {item['action']}")
    return " | ".join(parts)


def parse_task_agent_output(output: str):
    text = (output or "").strip()
    answer_match = re.search(r"Answer:\s*([A-E])\.", text, re.IGNORECASE)
    if answer_match:
        selected_option = answer_match.group(1).upper()
    else:
        line_start_match = re.search(r"(^|\n)\s*([A-E])\.", text)
        selected_option = line_start_match.group(2).upper() if line_start_match else "A"
    action_match = re.search(r"Action:\s*(CLICK|SELECT|TYPE)", text, re.IGNORECASE)
    action = action_match.group(1).upper() if action_match else ""
    value_match = re.search(r"Value:\s*(.*)$", text, re.MULTILINE)
    value = value_match.group(1).strip() if value_match else ""
    return selected_option, (action + " " + value).strip()


def summarize_prefix_if_needed(engine, task, older_summary, recent_buffer, keep_recent_items):
    if len(recent_buffer) <= keep_recent_items:
        return older_summary, recent_buffer, None
    prefix = recent_buffer[:-keep_recent_items]
    tail = recent_buffer[-keep_recent_items:]
    prefix_text = "\n".join(f"- {render_recent_item(item)}" for item in prefix)
    user_text = SUMMARIZER_USER_TEMPLATE.format(
        task=task,
        older_summary=older_summary or "(empty)",
        prefix_text=prefix_text,
    )
    response = ""
    data = None
    for _ in range(2):
        response = engine.chat(
            [
                {"role": "system", "content": SUMMARIZER_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            max_tokens=320,
        )
        try:
            data = safe_json_load(response)
            break
        except json.JSONDecodeError:
            data = None
    if data is None:
        logger.warning("Summarizer returned non-JSON/empty output. Falling back to heuristic summary.")
        summarized = fallback_older_summary(older_summary, prefix)
        return summarized, tail, {
            "prefix_text": prefix_text,
            "response": {"older_summary": summarized, "fallback": True},
        }
    return data.get("older_summary", older_summary), tail, {
        "prefix_text": prefix_text,
        "response": data,
    }


def build_memory_state_for_sample(sample, memory_engine, keep_recent_items=3):
    older_summary = ""
    recent_buffer = []
    traces = []
    previous_steps = sample.get("previous_step_records", [])
    for idx in range(1, len(previous_steps)):
        prev_step = previous_steps[idx - 1]
        curr_step = previous_steps[idx]
        user_text = MEMORY_AGENT_USER_TEMPLATE.format(
            task=sample["confirmed_task"],
            older_summary=older_summary or "(empty)",
            recent_buffer="\n".join(f"- {render_recent_item(item)}" for item in recent_buffer) or "(empty)",
            action_repr=curr_step.get("action_repr", "(missing action repr)"),
        )
        item = None
        response = ""
        for local_try in range(2):
            response = memory_engine.chat_with_images(
                system_prompt=MEMORY_AGENT_SYSTEM,
                user_text=user_text,
                previous_image=prev_step.get("screenshot"),
                current_image=curr_step.get("screenshot"),
                max_tokens=320,
            )
            try:
                item = safe_json_load(response)
                break
            except json.JSONDecodeError:
                item = None
        if item is None:
            logger.warning(
                "Memory agent returned non-JSON/empty output for sample=%s transition=%s action=%s. Falling back.",
                f"{sample['annotation_id']}_{sample['action_uid']}",
                idx + 1,
                curr_step.get("action_repr", ""),
            )
            item = fallback_memory_item(curr_step.get("action_repr", ""))
        memory_item = {
            "item_type": item["item_type"],
            "change": item["change"].strip(),
            "element": (item.get("element") or "").strip(),
            "action_type": (item.get("action_type") or "").strip(),
            "action_value": (item.get("action_value") or "").strip(),
            "focus_after": (item.get("focus_after") or "").strip(),
            "next_goal": (item.get("next_goal") or "").strip(),
            "keep_image": bool(item.get("keep_image", False)),
            "action": (item.get("action") or "").strip(),
            "image_step_index": idx + 1 if bool(item.get("keep_image", False)) else None,
            "image": curr_step.get("screenshot") if bool(item.get("keep_image", False)) else None,
        }
        recent_buffer.append(memory_item)
        summary_trace = None
        older_summary, recent_buffer, summary_trace = summarize_prefix_if_needed(
            engine=memory_engine,
            task=sample["confirmed_task"],
            older_summary=older_summary,
            recent_buffer=recent_buffer,
            keep_recent_items=keep_recent_items,
        )
        traces.append(
            {
                "transition_index": idx + 1,
                "action_repr": curr_step.get("action_repr", ""),
                "memory_item": {
                    "item_type": memory_item["item_type"],
                    "change": memory_item["change"],
                    "element": memory_item["element"],
                    "action_type": memory_item["action_type"],
                    "action_value": memory_item["action_value"],
                    "focus_after": memory_item["focus_after"],
                    "next_goal": memory_item["next_goal"],
                    "keep_image": memory_item["keep_image"],
                    "action": memory_item["action"],
                    "image_step_index": memory_item["image_step_index"],
                },
                "summary_update": summary_trace,
                "older_summary_after": older_summary,
                "recent_buffer_after": [
                    {
                        "item_type": x["item_type"],
                        "change": x["change"],
                        "element": x["element"],
                        "action_type": x["action_type"],
                        "action_value": x["action_value"],
                        "focus_after": x["focus_after"],
                        "next_goal": x["next_goal"],
                        "keep_image": x["keep_image"],
                        "action": x["action"],
                        "image_step_index": x["image_step_index"],
                    }
                    for x in recent_buffer
                ],
            }
        )
    return older_summary, recent_buffer, traces


def build_memory_for_sample(sample, memory_engine, keep_recent_items=3):
    older_summary, recent_buffer, traces = build_memory_state_for_sample(
        sample=sample,
        memory_engine=memory_engine,
        keep_recent_items=keep_recent_items,
    )
    lines = []
    if older_summary:
        lines.append("[older_summary]")
        lines.append(older_summary)
    if recent_buffer:
        lines.append("[recent_buffer]")
        for idx, item in enumerate(recent_buffer, start=1):
            lines.append(f"{idx}. {render_recent_item(item)}")
    memory_text = "\n".join(lines)
    memory_images = [item["image"] for item in recent_buffer if item.get("keep_image") and item.get("image") is not None]
    return memory_text, memory_images, traces


class AgenticMemoryTaskEvaluator(ActionEvaluatorMultiChoice):
    @staticmethod
    def _build_aitw_result_summary(all_step_scores, sample_to_website):
        macro_step_acc = collections.defaultdict(list)
        for score, annotation_id in all_step_scores:
            macro_step_acc[annotation_id].append(score)
        acc_per_website = collections.defaultdict(list)
        for annotation_id, values in macro_step_acc.items():
            acc_per_website[sample_to_website[annotation_id]].append(sum(values) / len(values))
        partial_match = sum(score for score, _ in all_step_scores) / max(len(all_step_scores), 1)
        complete_match = (
            sum(1 for values in macro_step_acc.values() if values and min(values) == 1.0)
            / max(len(macro_step_acc), 1)
        )
        return {
            "partial_match": partial_match,
            "complete_match": complete_match,
            "step_acc": partial_match,
            "episode_count": len(macro_step_acc),
            "acc_per_website": {k: (sum(v) / len(v), len(v)) for k, v in acc_per_website.items()},
        }

    @staticmethod
    def _build_result_summary(
        all_element_acc,
        all_action_f1,
        all_step_acc,
        sample_to_website,
    ):
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
            macro_element_acc_value = sum(sum(v) / len(v) for v in macro_element_acc.values()) / len(macro_element_acc)
            macro_action_f1_value = sum(sum(v) / len(v) for v in macro_action_f1.values()) / len(macro_action_f1)
            macro_step_acc_value = sum(sum(v) / len(v) for v in macro_step_acc.values()) / len(macro_step_acc)
        else:
            macro_element_acc_value = 0.0
            macro_action_f1_value = 0.0
            macro_step_acc_value = 0.0
            error_ratio = {}
            acc_per_website = {}

        return {
            "element_acc": sum(x[0] for x in all_element_acc) / max(len(all_element_acc), 1),
            "action_f1": sum(x[0] for x in all_action_f1) / max(len(all_action_f1), 1),
            "step_acc": sum(x[0] for x in all_step_acc) / max(len(all_step_acc), 1),
            "marco_element_acc": macro_element_acc_value,
            "marco_action_f1": macro_action_f1_value,
            "marco_step_acc": macro_step_acc_value,
            "error_ratio": error_ratio,
            "acc_per_website": acc_per_website,
        }

    @staticmethod
    def _write_outputs(output_path, name, top_k, all_predictions, all_outputs, result):
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/{name}_predictions_top{top_k}.json", "w", encoding="utf-8") as f:
            json.dump(all_predictions, f)
        with open(f"{output_path}/{name}_results_top{top_k}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        with open(f"{output_path}/{name}_outputs_top{top_k}.json", "w", encoding="utf-8") as f:
            json.dump(all_outputs, f)

    def evaluate_dataset(
        self,
        dataset,
        task_engine,
        memory_engine,
        prompt_template,
        top_k=50,
        output_path=None,
        name="default",
        keep_recent_items=3,
    ):
        if dataset.data and dataset.data[0].get("action_space") == "aitw":
            return self.evaluate_dataset_aitw(
                dataset=dataset,
                task_engine=task_engine,
                memory_engine=memory_engine,
                output_path=output_path,
                name=name,
                keep_recent_items=keep_recent_items,
            )
        all_element_acc = []
        all_action_f1 = []
        all_step_acc = []
        sample_to_website = {}
        all_predictions = []
        all_outputs = []

        for sample in dataset.data:
            annotation_id = sample["annotation_id"]
            sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
            sample_to_website[annotation_id] = sample["website"]

            memory_text, memory_images, memory_trace = build_memory_for_sample(
                sample=sample,
                memory_engine=memory_engine,
                keep_recent_items=keep_recent_items,
            )

            pos_candidates = [c for c in sample["pos_candidates"] if c["rank"] < top_k]
            pos_ids = [c["backend_node_id"] for c in pos_candidates]
            if len(pos_ids) == 0:
                all_element_acc.append([0, annotation_id])
                all_action_f1.append([0, annotation_id])
                all_step_acc.append([0, annotation_id])
                all_predictions.append([sample_id, "", ""])
                all_outputs.append([sample_id, {"memory_trace": memory_trace, "rounds": []}])
                if output_path is not None:
                    result = self._build_result_summary(
                        all_element_acc, all_action_f1, all_step_acc, sample_to_website
                    )
                    self._write_outputs(output_path, name, top_k, all_predictions, all_outputs, result)
                continue

            _, _, target_out, _ = format_input_multichoice(sample, pos_ids[:1], pos_ids[0])
            _, target_action = self.postprocess_action(target_out)
            neg_candidates = [c for c in sample["neg_candidates"] if c["rank"] < top_k]
            neg_ids = [c["backend_node_id"] for c in neg_candidates]
            all_candidates = pos_ids + neg_ids
            random.shuffle(all_candidates)
            final_prediction = None
            rounds = []

            while len(all_candidates) > 1:
                candidate_ids = all_candidates[:5]
                all_candidates = all_candidates[5:]
                seq_context, seq_in, _, choices = format_input_multichoice(
                    sample, candidate_ids, -1, keep_html_brackets=True
                )
                prompt = copy.deepcopy(prompt_template)
                prompt[0]["content"] = TASK_AGENT_SYSTEM
                prompt[-1]["content"] = f"'''\n{seq_context}\n'''\n\n{seq_in}{TASK_AGENT_FORMAT_SUFFIX}"
                content_blocks = []
                for memory_image in memory_images:
                    content_blocks.append(image_to_chat_content(memory_image))
                content_blocks.append(image_to_chat_content(sample["screenshot"]))
                content_blocks.append(
                    {
                        "type": "text",
                        "text": f"{memory_text}\n\n{prompt[-1]['content']}" if memory_text else prompt[-1]["content"],
                    }
                )
                output = task_engine.chat_with_image_list(
                    system_prompt=prompt[0]["content"],
                    content_blocks=content_blocks,
                    max_tokens=220,
                )
                pred_element, pred_action = parse_task_agent_output(output)
                rounds.append(
                    {
                        "candidate_ids": candidate_ids,
                        "choices": choices,
                        "memory_text": memory_text,
                        "memory_image_count": len(memory_images),
                        "model_output": output,
                    }
                )
                if pred_element[0] != "A":
                    pred_element = ord(pred_element[0]) - ord("B")
                    try:
                        pred_element = choices[pred_element][0]
                        all_candidates.append(pred_element)
                        final_prediction = (pred_element, pred_action)
                    except IndexError:
                        final_prediction = None

            all_outputs.append(
                [
                    sample_id,
                    {
                        "memory_trace": memory_trace,
                        "memory_text": memory_text,
                        "memory_image_count": len(memory_images),
                        "rounds": rounds,
                    },
                ]
            )

            if len(all_candidates) == 0 or final_prediction is None:
                all_element_acc.append([0, annotation_id])
                all_action_f1.append([0, annotation_id])
                all_step_acc.append([0, annotation_id])
                all_predictions.append([sample_id, "", ""])
            else:
                all_element_acc.append([1 if final_prediction[0] in pos_ids else 0, annotation_id])
                all_action_f1.append([self.calculate_f1(final_prediction[1], target_action), annotation_id])
                all_step_acc.append(
                    [1 if all_element_acc[-1][0] == 1 and all_action_f1[-1][0] == 1 else 0, annotation_id]
                )
                all_predictions.append([sample_id, final_prediction[0], final_prediction[1]])

            if output_path is not None:
                result = self._build_result_summary(
                    all_element_acc, all_action_f1, all_step_acc, sample_to_website
                )
                self._write_outputs(output_path, name, top_k, all_predictions, all_outputs, result)

        result = self._build_result_summary(
            all_element_acc, all_action_f1, all_step_acc, sample_to_website
        )

        if output_path is not None:
            self._write_outputs(output_path, name, top_k, all_predictions, all_outputs, result)
        return result

    def evaluate_dataset_aitw(
        self,
        dataset,
        task_engine,
        memory_engine,
        output_path=None,
        name="default",
        keep_recent_items=3,
    ):
        all_step_scores = []
        sample_to_website = {}
        all_predictions = []
        all_outputs = []

        for sample in dataset.data:
            annotation_id = sample["annotation_id"]
            sample_id = f"{sample['annotation_id']}_{sample['action_uid']}"
            sample_to_website[annotation_id] = sample["website"]

            memory_text, memory_images, memory_trace = build_memory_for_sample(
                sample=sample,
                memory_engine=memory_engine,
                keep_recent_items=keep_recent_items,
            )
            ui_lines = []
            for item in (sample.get("ui_elements") or [])[:15]:
                ui_lines.append(f"- {item}")
            task_text = [
                f"Task: {sample.get('confirmed_task', '')}",
                f"Current activity: {sample.get('current_activity', '')}",
                f"Device: {sample.get('device_type', '')}",
            ]
            if ui_lines:
                task_text.append("Detected UI text/icons:")
                task_text.extend(ui_lines)
            user_text = "\n".join(task_text)
            if memory_text:
                user_text = f"{memory_text}\n\n{user_text}"

            content_blocks = []
            for memory_image in memory_images:
                content_blocks.append(image_to_chat_content(memory_image))
            if sample.get("screenshot") is not None:
                content_blocks.append(image_to_chat_content(sample["screenshot"]))
            content_blocks.append({"type": "text", "text": f"{user_text}\n\nPredict the next action now."})
            raw_output = task_engine.chat_with_image_list(
                system_prompt=AITW_TASK_AGENT_SYSTEM,
                content_blocks=content_blocks,
                max_tokens=220,
            )
            try:
                parsed_output = parse_aitw_action_prediction(raw_output)
                step_score = aitw_action_match(parsed_output, sample["ground_truth_action"])
            except Exception as exc:
                parsed_output = {
                    "action_type": "invalid",
                    "touch_point": None,
                    "lift_point": None,
                    "typed_text": "",
                }
                step_score = 0.0
                logger.warning("Failed to parse AITW task-agent output for sample=%s: %s", sample_id, exc)

            all_step_scores.append([step_score, annotation_id])
            all_predictions.append([sample_id, parsed_output, sample["ground_truth_action"], step_score])
            all_outputs.append(
                [
                    sample_id,
                    {
                        "memory_trace": memory_trace,
                        "memory_text": memory_text,
                        "memory_image_count": len(memory_images),
                        "ground_truth": build_aitw_action_description(sample["ground_truth_action"]),
                        "model_output": raw_output,
                        "parsed_prediction": parsed_output,
                        "step_score": step_score,
                    },
                ]
            )

            if output_path is not None:
                result = self._build_aitw_result_summary(all_step_scores, sample_to_website)
                os.makedirs(output_path, exist_ok=True)
                with open(f"{output_path}/{name}_predictions_top0.json", "w", encoding="utf-8") as f:
                    json.dump(all_predictions, f, indent=2, ensure_ascii=False)
                with open(f"{output_path}/{name}_results_top0.json", "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4)
                with open(f"{output_path}/{name}_outputs_top0.json", "w", encoding="utf-8") as f:
                    json.dump(all_outputs, f, indent=2, ensure_ascii=False)

        result = self._build_aitw_result_summary(all_step_scores, sample_to_website)
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            with open(f"{output_path}/{name}_predictions_top0.json", "w", encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=2, ensure_ascii=False)
            with open(f"{output_path}/{name}_results_top0.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            with open(f"{output_path}/{name}_outputs_top0.json", "w", encoding="utf-8") as f:
                json.dump(all_outputs, f, indent=2, ensure_ascii=False)
        return result


def load_prompt_template(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ids_from_file(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a memory-agent + task-agent GUI system on supported static GUI datasets.")
    parser.add_argument("--dataset-path", default="data/multimodal_mind2web")
    parser.add_argument("--dataset-format", default="auto", choices=["auto", "mind2web", "aitw"])
    parser.add_argument("--score-file", default="data/mind2web_aux/scores_all_data.pkl")
    parser.add_argument("--split", default="test_task")
    parser.add_argument("--output-dir", default="outputs/agentic_memory_task")
    parser.add_argument("--prompt-file", default="src/action_prediction/llm_prompt.json")
    parser.add_argument("--task-model", default="gpt-4o-mini")
    parser.add_argument("--memory-model", default="gpt-4o-mini")
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rate-limit", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--action-uid-file")
    parser.add_argument("--keep-recent-items", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    action_uids = load_ids_from_file(args.action_uid_file)
    samples = load_multimodal_samples(
        dataset_path=args.dataset_path,
        split=args.split,
        action_uids=action_uids,
        dataset_format=args.dataset_format,
    )
    candidate_results = None
    if samples and samples[0].get("action_space") == "mind2web":
        with open(args.score_file, "rb") as f:
            candidate_results = pickle.load(f)
    samples = attach_candidate_ranks(samples, candidate_results)
    dataset = StaticDataset(samples)

    task_engine = OpenAICompatibleEngine(
        model=args.task_model,
        api_base=args.api_base,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
        temperature=args.temperature,
    )
    memory_engine = OpenAICompatibleEngine(
        model=args.memory_model,
        api_base=args.api_base,
        api_key=args.api_key,
        rate_limit=args.rate_limit,
        temperature=args.temperature,
    )
    prompt_template = load_prompt_template(args.prompt_file)
    evaluator = AgenticMemoryTaskEvaluator(tokenizer=None)
    split_name = args.split if args.split != "all" else "mixed_test"
    results = evaluator.evaluate_dataset(
        dataset=dataset,
        task_engine=task_engine,
        memory_engine=memory_engine,
        prompt_template=prompt_template,
        top_k=args.top_k,
        output_path=args.output_dir,
        name=split_name,
        keep_recent_items=args.keep_recent_items,
    )
    logger.info("Results: %s", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

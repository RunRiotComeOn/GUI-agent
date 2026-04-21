# Cross-Task Experience Replay (v2)

This document replaces the experience-extraction and retrieval parts of
[docs/cross_task_memory_pipeline.md](/u/yhuang48/GUI-agent/docs/cross_task_memory_pipeline.md:1).

Compared to v1, v2 makes three hard changes:

- **Success-only.** Only successful trajectories are distilled into experience.
  Failure analysis is out of scope for now.
- **Two-stage distillation.** A per-trajectory *causal summary* is produced
  first; a later *pattern-extraction* pass turns multiple summaries into
  reusable experience entries. No candidate is ever written directly from a
  single trajectory into the library.
- **One experience per step, chosen by an agent.** At inference time, a small
  selector agent sees the whole catalog and the current step context, and
  picks at most one experience. Top-k retrieval (v1 §5–6) is removed.

## 1. Pipeline

```
Successful Trajectory
  └─ Stage A: Trajectory Summarizer (LLM, once per trajectory)
       └─ Causal Summary  →  summary_buffer.jsonl
                              │
                              │ (flush when buffer reaches N summaries)
                              ▼
       └─ Stage B: Pattern Extractor (LLM, batch over the buffer)
            └─ Experience Library  +  Catalog index
                                      │
                                      ▼
   ─────────────────────────────── inference ─────────────────────────────
   For each step t of a running task:
     └─ Stage C: Selector Agent (small model, catalog + last 3 steps)
          └─ exp_id | null
               │
               ▼
          Inject one experience into the step's policy prompt slot
```

All stages are offline except Stage C.

## 2. Stage A — Trajectory → Causal Summary

Run once per successful trajectory. The summarizer's job is to compress an
event stream into an *attributable* summary — i.e. a summary where each
contribution is tied to a concrete decision point.

### 2.1 Inputs

- task instruction
- website, domain
- per-step records:
  `(step_index, observation_desc, candidate_skills, chosen_skill, chosen_tool, action, observation_after)`
- final success signal (must be true; failed trajectories are skipped)

**Default source: Mind2Web gold trajectories.** Each annotation in
`Multimodal-Mind2Web` is already a successful trajectory by construction
(every step is a ground-truth target action: `element_id + action_type +
value`). Stage A runs directly over gold annotations — no model rollout is
required to bootstrap the pipeline. `chosen_skill / chosen_tool / action` in
the per-step record come from ground truth.

Model-rollout-based collection (keep only trajectories where the model
matches gt on every step) is a later-phase augmentation, not the default.

### 2.2 Output schema

Strict JSON. No markdown fences.

```json
{
  "task_id": "string",
  "goal": "abstracted task intent, site/date/names removed",
  "key_trajectory": [
    "step 3: select_from_dropdown located the date picker",
    "step 5: click_and_confirm committed the filter"
  ],
  "skill_effectiveness": {
    "<skill_name>": "decisive | necessary | redundant"
  },
  "critical_turning_points": [
    {
      "step": 5,
      "decision": "switched from browse branch to confirm branch",
      "reason": "modal was open; all subsequent actions had to stay inside it"
    }
  ],
  "tool_usage_patterns": [
    "fill -> confirm -> advance"
  ],
  "outcome": "succeeded; main contribution from turning points at step 3 and 5"
}
```

### 2.3 Hard rules

- `goal` must be abstracted. Keep the intent, drop the specific site, date,
  name, or passenger count. Without this, Stage B cannot cluster.
- `skill_effectiveness` must use the three categorical labels. This is the
  main signal Stage B uses to separate *load-bearing* skills from filler
  steps.
- Every `critical_turning_points` entry must carry a `reason`. Turning points
  without a reason are discarded before Stage B.

Summaries are intermediate artifacts. They are appended to
`summary_buffer.jsonl` and are **not** served at inference time.

## 3. Stage B — Summary Buffer → Experience

### 3.1 Trigger

Count-based. Flush when the buffer reaches **N = 50** summaries
(configurable). Time-based triggers are intentionally not supported in v2.

### 3.2 Pattern-extractor contract

Input: the full batch of summaries since the last flush.

The extractor is asked to identify recurring *success* patterns that:

1. appear across multiple trajectories (support ≥ K, default **K = 3**),
2. show up in `critical_turning_points` or as `decisive` entries in
   `skill_effectiveness`,
3. describe a transferable rule, not a site-specific step.

### 3.3 Output schema

```json
{
  "experience_id": "exp_filter_confirm_001",
  "title": "confirm after filter or picker change (≤ 20 chars)",
  "applicable_context": {
    "when": "a filter, date, picker, or passenger value was just changed",
    "ui_signals": ["Apply/Done/Search button visible", "modal still open"],
    "domain_hint": "generic | travel | shopping | ..."
  },
  "action_guidance": "free-text heuristic in one short paragraph",
  "action_templates": [
    "CLICK [button: Apply|Done|Update|Search|Continue]",
    "CLICK [button: Confirm] inside the currently open modal"
  ],
  "evidence": {
    "support_count": 8,
    "supporting_trajectories": ["task_023", "task_087", "..."],
    "turning_point_hits": 11
  },
  "confidence": 0.0
}
```

- `action_templates` is optional and capped at **2 entries**. These are
  skill/action shapes, not concrete coordinates. If a pattern is a pure
  heuristic, leave the list empty.
- `confidence` is computed from `support_count`, site/domain breadth, and
  extractor self-reported certainty. The exact formula is implementation
  detail and can be iterated separately.

### 3.4 Merging and decay

- Before insert, semantic-dedup against existing `experience_id`s. On match,
  merge by union-ing `supporting_trajectories` and bumping `support_count`.
- Library-level cap (see §4). When over cap, evict by
  `support_count × recency`.
- An experience that has not been selected by Stage C for 30 days should be
  flagged for re-evaluation on the next batch. Eviction is conservative —
  demote to a cold tier rather than delete, so evidence is recoverable.

## 4. Experience Library and Catalog

The library stores full experience records. The **catalog** is a compressed
index built from the library — this is what the selector sees.

Catalog row shape:

```json
{
  "id": "exp_filter_confirm_001",
  "title": "confirm after filter change",
  "trigger": "just changed a filter/picker value and haven't clicked Apply/Done/Search yet"
}
```

Budget:

- Catalog must fit comfortably in the selector prompt.
- Target **≤ 50 entries** and **≤ ~3K tokens** total.
- Evict by `support_count × recency` when the cap is hit.

## 5. Stage C — Step-time Selector Agent

### 5.1 Role

One lightweight call per step (recommended model: Haiku-class). Picks at most
one experience that is clearly applicable, or returns `null`.

### 5.2 Context window

The selector sees:

- task instruction
- **the last 3 steps** of the running task (each as a short
  `(action, observation_summary)` tuple)
- a short summary of the current observation
- the full catalog

"Last 3 steps" is a v2 constant; it is not a tunable hyperparameter.

### 5.3 Prompt skeleton

```text
You are an experience selector for a GUI agent.

Task:
{task}

Recent steps (last 3, oldest first):
{recent_steps}

Current observation summary:
{current_obs_summary}

Experience catalog:
{catalog_rows}

Rules:
- Pick AT MOST ONE experience whose trigger clearly matches the current step.
- If nothing clearly applies, return null. Do not stretch to fit.
- Output strict JSON only. No markdown fences.

Output format:
{"experience_id": "<id>" | null, "reason": "<one sentence>"}
```

### 5.4 Hard rules

- Exactly one selection or `null`. The selector is not allowed to return
  multiple ids.
- `reason` is always logged. These logs are the main signal for improving the
  catalog's `trigger` phrasing later.
- On `null`, the policy prompt's experience slot is left empty — it is not
  filled with a default or a fallback item.

### 5.5 Injection

When an `experience_id` is returned, fetch the full record from the library
and render into a fixed slot in the policy prompt:

```text
[active_experience]
Title: {title}
When it applies: {applicable_context.when}
Guidance: {action_guidance}
Suggested action shapes (if any):
- {action_templates[0]}
- {action_templates[1]}
```

Only this slot is affected. The rest of the prompt is unchanged.

## 6. Hyperparameters (v2 defaults)

| Name | Value | Where |
|------|-------|-------|
| Summary buffer flush size `N` | 50 | Stage B trigger |
| Minimum support for experience `K` | 3 | Stage B |
| Selector context window | last 3 steps | Stage C |
| `action_templates` per experience | ≤ 2 | Stage B output |
| Catalog cap | 50 entries / ~3K tokens | §4 |
| Cold-tier flag after idle | 30 days | §3.4 |

## 7. Relationship to v1

Kept:

- The two-layer memory idea (task-local vs cross-task).
- The principle that the library stores transferable lessons, not raw
  trajectories.
- Canonicalization and dedup ideas from v1 §4.1–4.2 apply to Stage B output.

Replaced / removed:

- v1 §3 (Candidate Extraction) — the split success extractor / failure
  analyzer is removed. Stage A produces causal summaries; there is no
  failure extractor.
- v1 §5 (Retrieval) — hybrid top-k retrieval is removed. Replaced by the
  selector agent in Stage C.
- v1 §6 (Injection) — the multi-item prompt section is removed. Replaced by
  the single-slot injection in §5.5.
- v1 §7 (Failure extraction in interactive benchmarks) — out of scope for v2.

Deprecated files (kept on disk but unused at runtime under v2):

- [src/action_prediction/cross_task_failure_memory_schema.json](/u/yhuang48/GUI-agent/src/action_prediction/cross_task_failure_memory_schema.json:1)
- [docs/cross_task_failure_analyzer_prompt.md](/u/yhuang48/GUI-agent/docs/cross_task_failure_analyzer_prompt.md:1)

Wiring changes required in
[src/action_prediction/evaluate_vlm.py](/u/yhuang48/GUI-agent/src/action_prediction/evaluate_vlm.py:124):

- Replace `load_cross_task_memory_bank` with a loader that reads the
  experience library plus the catalog.
- Replace `retrieve_cross_task_memory(..., top_k=...)` with a selector call
  that returns `experience_id | None`.
- Drop the `cross_task_memory_top_k` CLI flag. The new pipeline has no top-k
  knob.

## 8. Minimal first implementation

1. Stage A summarizer with the schema in §2.2, run offline over
   **Mind2Web gold trajectories** (see §2.1). No model rollout needed for
   bootstrap.
2. Append-only `summary_buffer.jsonl`.
3. Stage B extractor with count-based flush at `N = 50` and `K = 3`.
4. Experience library file + catalog index generator.
5. Selector agent in the evaluator, with the single-slot injection in §5.5.
6. Log selector decisions (`experience_id`, `reason`, step index) for later
   evaluation of the selector itself.

This is the minimum required to compare v2 against the v1 prompt-injection
baseline end-to-end. Model-rollout-based collection is left as a follow-up
once gold-trajectory distillation is validated.

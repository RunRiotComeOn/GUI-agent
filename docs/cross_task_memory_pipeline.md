# Cross-Task Memory Pipeline

This document describes a practical pipeline for building and using
cross-task memory on top of the current task-local hybrid memory stack.

The goal is to build a reusable experience layer that transfers across tasks,
websites, and domains.

The pipeline is designed around three ideas:

- extract reusable experience, not raw trajectories
- keep task-local memory and cross-task memory separate
- retrieve a small set of high-value lessons at inference time

## 1. Memory Layers

The system should maintain two separate memory layers.

### 1.1 Task-Local Memory

Already present in the current codebase:

- `older_summary`
- `recent_buffer`
- `keep_image=true` keyframe screenshots

Use it for:

- current task progress
- current branch state
- local interaction grounding

### 1.2 Cross-Task Memory

New reusable experience layer.

Use it for:

- cross-task heuristics
- reusable procedures
- repeated failure patterns

This layer should **not** store raw task histories.

It should store:

- transferable lessons
- reusable subflow policies
- common error signatures and fixes

## 2. Data Flow

The recommended cross-task memory pipeline has four stages:

1. candidate extraction
2. memory cleaning and aggregation
3. retrieval at inference time
4. prompt or scorer injection

## 3. Stage A: Candidate Extraction

### 3.1 Source Signals

Extract memory candidates from:

- successful interactive trajectories
- failed or drifted trajectories
- success-vs-failure contrast pairs
- disagreement points between two policies
- hand-labeled inspection cases

### 3.2 What To Extract

There are three high-value candidate types.

#### Procedure Memory

Reusable subflow templates.

Examples:

- date picker operation order
- filter panel usage pattern
- checkout form completion order
- search flow: fill values, then trigger search

#### Heuristic Memory

Local cross-task rules.

Examples:

- when a modal is visible, prioritize actions inside the modal
- after choosing filters, look for Apply / Done / Update / Search
- prefer clickable semantic ancestors over decorative child nodes

#### Failure Memory

Reusable error patterns.

Examples:

- decorative child selected instead of button parent
- background page clicked while modal stayed open
- value selected but confirmation action missing
- wrong branch continued after subflow switch

### 3.3 Extraction Strategy

Use two extractors.

#### Success Extractor

Input:

- local success window
- task description
- before/after screenshots
- chosen action

Output:

- procedure memory candidate
- heuristic memory candidate

#### Failure Analyzer

Input:

- suspected failure point
- local rollout window
- chosen action and candidate set
- optional later failure symptom
- optional success/failure contrast

Output:

- failure memory candidate

Use:

- [docs/cross_task_failure_analyzer_prompt.md](/u/yhuang48/GUI-agent/docs/cross_task_failure_analyzer_prompt.md:1)
- [src/action_prediction/cross_task_failure_memory_schema.json](/u/yhuang48/GUI-agent/src/action_prediction/cross_task_failure_memory_schema.json:1)

## 4. Stage B: Cleaning And Aggregation

Raw extracted memories are noisy.

Do not insert them directly into the retrieval index.

### 4.1 Canonicalization

Normalize:

- site names
- domain names
- UI pattern names
- error type names
- wording of generalizable rules

### 4.2 Deduplication

Cluster candidates by:

- `memory_type`
- `error_type`
- `ui_pattern`
- semantic similarity of `generalizable_rule`

Merge near-duplicates into one canonical memory item.

### 4.3 Confidence Aggregation

Aggregate confidence from:

- extractor confidence
- repetition count
- number of websites seen
- number of domains seen
- success impact or failure recurrence

Suggested stored metadata:

```json
{
  "support_count": 17,
  "website_count": 6,
  "domain_count": 3,
  "avg_confidence": 0.81,
  "last_updated": "2026-04-19"
}
```

### 4.4 Keep Only Reusable Items

Filter out:

- one-off task-specific advice
- DOM-specific memorization
- low-support low-confidence items
- contradictory items without enough support

## 5. Stage C: Retrieval At Inference Time

At inference time, do not retrieve the whole memory store.

Retrieve a small set of high-value memories.

### 5.1 Retrieval Query

Build a retrieval query from:

- task text
- website
- domain
- current UI subflow
- current local state summary
- optional recent failure symptom

Useful query fields:

- `domain`
- `site`
- `ui_pattern`
- `interaction_stage`
- `candidate_action_types`

### 5.2 Retrieval Strategy

Recommended retrieval is hybrid:

- symbolic filtering first
- embedding or semantic retrieval second
- rerank by applicability and confidence

Example:

1. filter to matching domain or generic
2. prefer matching UI patterns
3. semantic rerank by task and current local state
4. return top 3 to 5 items

### 5.3 Retrieval Budget

Keep the retrieval budget small.

Recommended:

- `top_k = 3` for prompt injection
- `top_k = 5` only if each item is short

## 6. Stage D: Injection

There are two recommended ways to use cross-task memory.

### 6.1 Prompt Injection

Safest first version.

Inject retrieved memory into the task prompt as a short section:

```text
[cross_task_memory]
- When a modal is visible, prioritize actions inside the modal before using the background page.
- After selecting a filter or date, look for an explicit Apply, Done, Update, or Search action.
- Prefer the semantic clickable ancestor over a decorative child node.
```

Use this for:

- ablation studies
- first deployment
- human-readable debugging

### 6.2 Candidate Scoring Bias

Stronger but riskier second version.

Use memory items to bias ranking or candidate filtering.

Examples:

- modal-related memory increases scores for modal-contained candidates
- apply/search memory increases scores for confirm-style actions after filters
- decorative-child memory downweights non-semantic visual nodes

## 7. Failure Extraction In Interactive Benchmarks

The main difficulty in interactive settings is that final task failure does not
directly tell you which step was the first true mistake.

Therefore, failure extraction should target:

- local failure signatures
- suspected branch-drift points
- repeated downstream failure symptoms

Do not rely on whole-trajectory failed-task summaries alone.

Instead:

1. detect suspicious local points
2. run the failure analyzer only there
3. aggregate repeated patterns across tasks

This yields cleaner cross-task failure memory.

## 8. Suggested Minimal Implementation

A practical first version can be built with:

1. one offline extractor for success heuristics
2. one failure analyzer for local failure points
3. one dedup + aggregation pass
4. one retrieval module
5. one prompt injection path

This is enough to test whether cross-task memory helps before integrating it
into ranking.

## 9. Recommended Stored Record

For every final memory item, store:

```json
{
  "memory_id": "err_modal_context_001",
  "memory_type": "error_pattern",
  "error_type": "modal_context_ignored",
  "generalizable_rule": "when a modal is visible, prioritize actions inside the modal before interacting with the background page",
  "scope": {
    "domain": "generic",
    "site": "",
    "ui_pattern": "modal_overlay"
  },
  "retrieval_tags": [
    "modal",
    "overlay",
    "wrong_context"
  ],
  "support_count": 17,
  "website_count": 6,
  "domain_count": 3,
  "confidence": 0.88
}
```

## 10. Recommended Next Steps

1. Implement offline extraction of memory candidates from existing rollouts.
2. Run the failure analyzer only on suspected drift or failure points.
3. Build a small cleaned memory bank with high-confidence items only.
4. Add prompt injection retrieval first.
5. Evaluate on interactive and static benchmarks before attempting scorer bias.

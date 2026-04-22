# GUI Agent Memory System

本文档总结 Mind2Web 静态评测里当前这套 GUI agent 的记忆系统。整套设计由两个正交的层组成：

- **Task-specific memory** — 只关于*当前这个任务*的上下文，分短时 / 长时。
- **Cross-task memory** — 跨任务复用的*经验*，用一个选择器 agent 在每一步按需注入。

两层记忆都最终拼到 policy prompt 里交给 VLM（默认 `gpt-4o-mini`）。

代码入口：

- `src/action_prediction/evaluate_vlm.py` — 评测主入口，负责拼装上下文
- `src/action_prediction/evaluate_agentic_memory_task.py` — task-specific memory 的在线压缩逻辑
- `src/action_prediction/stage_a_summarizer.py` / `stage_b_pattern_extractor.py` / `stage_c_selector.py` — cross-task experience 三个阶段

---

## 1. 总览

```
                    ┌───────────────────────────────────────────────┐
                    │               Policy Prompt (VLM)              │
                    │                                                 │
  Task-specific ──▶ │  [older_summary]     ← 长时（agentic 压缩）    │
                    │  [recent_buffer]     ← 短时（结构化近 k 步）   │
                    │  [keyframe images]   ← 关键帧截图              │
                    │                                                 │
  Cross-task    ──▶ │  [active_experience] ← 选择器每步挑 ≤1 条      │
                    │                                                 │
                    │  [current screenshot] + [candidate choices]     │
                    └───────────────────────────────────────────────┘
```

- Task-specific 每步都要更新（在线），只活在当前任务生命周期里。
- Cross-task experience 离线攒出来一个 library，在线只做"选 or 不选"。

---

## 2. Task-specific Memory（`agentic_summary_recent` 模式）

处理*同一任务*内的历史，把历史分成近处和远处：

- **近处高保真** — 直接服务当前 action grounding
- **远处压缩** — 保留任务整体进展、分支状态

见 `docs/mind2web_hybrid_history.md`。

### 2.1 短时：`recent_buffer`（近程结构化）

最近 `k` 步的结构化记录，每条是一个 memory agent 产出的 item，保留 action grounding 所需的锚点。

典型 item 形态（见 `docs/online_change_memory_prompt.md`）：

```
[recent_change] element: ... | action_type: CLICK|SELECT|TYPE | value: ... |
                focus_after: ... | next_goal: ...
[recent_change_keyframe] ...（同上，并且对应截图会作为 history image 送进 VLM）
```

每步至少保留：

- 被作用的元素
- action primitive（CLICK / SELECT / TYPE）
- 输入/选中的值（如有）
- 步骤后活跃的 UI 子流程
- 下一个即时局部目标

这些锚点是避免模型在相邻分支（如 `truck options` vs `add-ons`、`guest modal` vs `results page`）之间串台的关键。

### 2.2 长时：`older_summary`（远程压缩）

更早的历史由 agentic memory **在线滚动压缩**成一段 summary：

- 每一步先抽 newest local change → 追加到 recent buffer
- 只有当 buffer 过长时，把*最旧那段前缀*压进 `older_summary`
- 永远不从整条轨迹重新总结

`older_summary` 负责回答「之前整体发生过什么」：任务阶段、子流程、已放弃分支、已确定的关键值。

### 2.3 视觉：`keep_image=true` 关键帧

memory agent 在写 recent item 时可以把一条标记为 `recent_change_keyframe`。只有这些少量关键帧的截图会作为 history image 送进 VLM。

原则是宁缺毋滥 — 上下文稳定、视觉注意力不被稀释。

### 2.4 调用方式

```bash
python src/action_prediction/evaluate_vlm.py \
  --history-mode agentic_summary_recent \
  --recent-k 5 \                    # 或 --recent-k 3 --recent-k-policy adaptive
  --memory-model gpt-4o-mini
```

---

## 3. Cross-task Memory（Experience Replay v2）

跨任务的经验库。与 task-specific 层不同，它不讲"这次发生了什么"，而是讲"这类情境下通常该怎么做"。

见 `docs/cross_task_experience_replay_v2.md`。三个阶段：

```
成功轨迹 ──▶ Stage A ──▶ causal summary ─┐
                                         │ (buffer 到 N=50)
                                         ▼
                                    Stage B ──▶ experience library + catalog
                                                       │
  ─────────────────────── inference ────────────────── │ ────
                                                       ▼
  每一步：task + 最近 3 步 + 当前观测 + catalog ──▶ Stage C ──▶ exp_id | null
                                                       │
                                                       ▼
                                              注入 policy prompt 的 [active_experience] 槽
```

### 3.1 数据源：Mind2Web gold trajectory

Multimodal-Mind2Web 里每个 annotation 本身就是一条完整的成功轨迹（每步都是 ground-truth target action），所以不需要先跑 rollout 就可以 bootstrap 经验库。当前实现 **success-only**，失败分析暂不做。

### 3.2 Stage A — Trajectory Summarizer

每条成功轨迹一次 LLM 调用，压成一个 **causal summary**（严格 JSON，6 个必填字段）：

```json
{
  "goal": "抽象化的任务意图，去掉具体站点/日期/姓名",
  "key_trajectory": ["step 3: select_from_dropdown located date picker", ...],
  "skill_effectiveness": { "<skill_name>": "decisive|necessary|redundant" },
  "critical_turning_points": [
    {"step": 5, "decision": "switched to confirm branch",
     "reason": "modal was open; all subsequent actions had to stay inside it"}
  ],
  "tool_usage_patterns": ["fill -> confirm -> advance"],
  "outcome": "succeeded; main contribution from turning points at step 3 and 5"
}
```

硬约束：

- `goal` 必须抽象（否则 Stage B 无法聚类）
- `skill_effectiveness` 三档标签限定：`decisive | necessary | redundant`
- 每条 `critical_turning_points` 必须带非空 `reason`

输出追加到 `outputs/cross_task_experience/summary_buffer.jsonl`，可断点续跑。

### 3.3 Stage B — Pattern Extractor

Buffer 到 N=50 条 summary 时批量触发。两段式：

1. **分块抽取** — 每块 50 条，让 LLM 提出候选经验
2. **LLM consolidator pass** — 跨块合并、去语义重复、把支持数 < K=3 的丢掉

输出条目 schema（示例）：

```json
{
  "experience_id": "exp_confirm_filter_application",
  "title": "confirm after filter or picker change",
  "applicable_context": {
    "when": "a filter/date/picker value was just changed and Apply/Done is visible",
    "ui_signals": ["Apply/Done button visible", "modal still open"],
    "domain_hint": "generic"
  },
  "action_guidance": "After changing a filter value, locate and click the confirming action...",
  "action_templates": [                      // 最多 2 条
    "CLICK [button: Apply|Done|Update|Search]",
    "CLICK [button: Confirm] inside the currently open modal"
  ],
  "evidence": {
    "support_count": 8,
    "supporting_trajectories": ["task_023", "task_087", ...],
    "turning_point_hits": 11
  },
  "prevents_mistake": "agent hitting back/advance before filter actually applied"
}
```

硬测试（写在 prompt 里）：**"如果一个 plausible agent 不知道这条规则，它会犯什么具体错误？"** 回答不出来的条目直接丢掉。

Prompt 里还列了一批 REJECT 例子来挡住"任务骨架式"的假条目（`search_and_select`、`filter_and_apply` 这类）。

**当前产物**：`experience_library_4o_v3.jsonl` 共 4 条经验，`catalog_4o_v3.json` 是对应的精简 index。

### 3.4 Stage C — Step-time Selector Agent

inference 时每一步一次轻量 LLM 调用。看到的东西：

- 任务指令
- 最近 3 步（action + observation_summary）
- 当前观测的简要摘要
- 完整 catalog

输出严格 JSON：

```json
{"experience_id": "<id>" | null, "reason": "<one sentence>"}
```

硬规则：

- 最多一条，否则返回 null
- "null 好过勉强匹配"
- `reason` 必须引用当前步骤的具体触发信号（用来后续改 trigger 措辞）

命中时，从 library 取出完整记录，渲染进 policy prompt 的固定槽位：

```
[active_experience]
Title: {title}
When it applies: {applicable_context.when}
Guidance: {action_guidance}
Suggested action shapes:
- {action_templates[0]}
- {action_templates[1]}
```

槽位之外的 prompt 完全不动。

### 3.5 超参（v2 默认值）

| 名称 | 默认值 |
|---|---|
| Summary buffer flush size N | 50 |
| 最小支持数 K | 3 |
| Selector 上下文窗口 | 最近 3 步 |
| 每条经验的 action_templates | ≤ 2 |
| Catalog 容量 | ≤ 50 条 / ≈3K tokens |

---

## 4. 两层如何组合

在 `evaluate_vlm.py` 里：

1. 先算 task-specific 记忆 → `older_summary` + `recent_buffer` + keep_image 帧
2. 再跑 Stage C 选择器 → `active_experience_text`（或空）
3. 拼进 policy prompt：

```
Hybrid trajectory context for the same task:
Task: ...

[older_summary]
<agentic memory 对更早轨迹的压缩>

[recent_buffer]
1. [recent_change] ...
2. [recent_change_keyframe] ...
3. [recent_change] ...

[active_experience]          ← 来自 cross-task experience library（可能为空）
Title: ...
When it applies: ...
Guidance: ...
```

两层互不替代：

- task-specific 负责"这次到哪儿了、刚做了什么"
- cross-task 负责"这类情境下一条可迁移的规则"

---

## 5. 当前评测现状

### 10 条 >15 步的困难 case（`tmp_action_uids_step_gt15_10.txt`）

| metric | baseline (recent-k=5) | +experience | Δ |
|---|---|---|---|
| element_acc | 0.400 | 0.400 | 0 |
| action_f1 | 0.940 | 0.950 | +0.010 |
| marco_action_f1 | 0.925 | 0.938 | +0.013 |
| marco_step_acc | 0.375 | 0.375 | 0 |

命中 5/10，4× `exp_confirm_filter_application` + 1× `exp_semantic_clickable_ancestor`。

### 100 条 random_seed123（adaptive `recent-k=3`）

| metric | baseline | +experience | Δ |
|---|---|---|---|
| element_acc | 0.290 | 0.300 | +0.010 |
| action_f1 | 0.585 | 0.621 | **+0.036** |
| marco_element_acc | 0.292 | 0.311 | +0.018 |
| marco_action_f1 | 0.583 | 0.622 | **+0.039** |
| marco_step_acc | 0.224 | 0.215 | −0.009 |

命中 28/100（19× `exp_confirm_filter_application`，9× `exp_semantic_clickable_ancestor`，72× null）。

**读数**：对连续型指标（element/action_f1）+1~4 pts；对 all-or-nothing 的 `step_acc` 略微负。经验把一部分 1-错案例救成 0-错，同时也把一部分 0-错案例拖成 1-错 — 靠谱命中的收益和假阳性命中的伤害互相抵消。

---

## 6. 相关产物

**代码**：

- `src/action_prediction/evaluate_vlm.py`
- `src/action_prediction/evaluate_agentic_memory_task.py`
- `src/action_prediction/stage_a_summarizer.py`
- `src/action_prediction/stage_b_pattern_extractor.py`
- `src/action_prediction/stage_c_selector.py`
- `src/action_prediction/multimodal_utils.py`

**数据 / artifacts**（均在 `outputs/cross_task_experience/`）：

- `summary_buffer.jsonl` — 150 条 causal summary（test_task/test_website/test_domain 各 50）
- `experience_library_4o_v3.jsonl` — 当前使用的 4 条经验
- `catalog_4o_v3.json` — 对应的 selector catalog

**相关文档**：

- `docs/mind2web_hybrid_history.md` — task-specific hybrid 模式原始设计
- `docs/online_change_memory_prompt.md` / `online_change_memory_template.md` — memory agent 的 prompt 规格
- `docs/cross_task_experience_replay_v2.md` — experience v2 设计
- `docs/cross_task_memory_pipeline.md` — v1（已被 v2 §3–6 取代的部分）

---

## 7. 设计里的三条主线

1. **近处保真、远处压缩**（task-specific 层） — 不要把近几步也丢给 memory agent 抽象掉，也不要保留整条原始轨迹。
2. **经验是*跨任务可迁移的规则*，不是任务片段**（cross-task 层） — Stage B 的 HARD TEST 是拒绝任务骨架、保留能防止具体错误的条目。
3. **选择器 agent 而不是 top-k retrieval**（Stage C） — v2 只注入一条或一条都不注入，由一个小模型显式判断是否触发；`null` 永远是合法答案。

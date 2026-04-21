# Mind2Web Hybrid History 方法说明

这份文档总结当前仓库里用于静态 `Multimodal-Mind2Web` 评测的一条混合历史建模路线：

- 近程历史保留结构化 `recent` memory
- 远程历史使用 `agentic memory` 做在线压缩
- 最终仍然使用官方 `ActionEvaluatorMultiChoice` 指标

对应代码入口在：

- `src/action_prediction/evaluate_vlm.py`
- `src/action_prediction/evaluate_agentic_memory_task.py`

## 1. 背景

这条方法线的出发点是把历史分成两类：

- 近程历史：需要高保真，直接服务于当前 action grounding
- 远程历史：更适合压缩，用来保留任务整体进展与分支状态

因此更合理的方向不是“全历史都交给 memory agent”，而是：

- 近处保真
- 远处压缩

## 2. 方法核心

当前实现的混合模式名为：

- `agentic_summary_recent`

它的思路是：

1. 对同一任务的更早轨迹，使用 `agentic memory` 在线更新
2. 将较远历史压缩成一段 `older_summary`
3. 最近 `k` 步保留为 memory agent 产出的结构化 `recent_buffer`
4. 如果 recent item 标记了 `keep_image=true`，则把对应关键截图作为 history image 一并送入 VLM
5. 最终将两部分一起作为历史上下文提供给 `evaluate_vlm.py`

最终历史上下文结构如下：

```text
Hybrid trajectory context for the same task:
Task: ...

[older_summary]
<agentic memory 对更早轨迹的压缩摘要>

[recent_buffer]
1. [recent_change] ... | element: ... | action_type: ... | focus_after: ... | next_goal: ...
2. [recent_change_keyframe] ... | element: ... | action_type: ... | focus_after: ... | next_goal: ... | action: ...
3. [recent_change] ... | element: ... | action_type: ... | focus_after: ... | next_goal: ...
```

其中：

- `older_summary` 负责保留远程任务进展、分支切换、关键状态变化
- `recent_buffer` 负责保留当前决策真正依赖的细粒度近程操作
- `keep_image=true` 的 recent item 对应的 screenshot 会作为少量关键帧图补充给模型

## 3. 为什么这么做

### 3.1 近程历史需要高保真

在静态 GUI action 选择任务里，模型通常非常依赖最近几步的细粒度轨迹信息。

典型原因包括：

- 当前控件选择依赖最近刚输入的值
- 当前页面局部状态依赖最近几个 click/type/select
- 当前分支判断依赖最近 2 到 5 步的小范围 UI 演化

如果这些信息被过早压缩成抽象叙述，模型会更容易：

- 选错元素
- 选错 action type
- 或者直接退化成 `A. None of the above`

### 3.2 远程历史适合压缩

更早的历史往往用于回答这类问题：

- 任务大致推进到了哪个阶段
- 当前仍然处于哪个子流程
- 某个旧分支是否已经被放弃
- 某些关键值是否已经在更早阶段确定

这些信息不一定需要逐步保留原始轨迹，压缩成 `older_summary` 通常已经足够。

所以更合理的分工是：

- 近程历史回答“刚刚发生了什么”
- 远程历史回答“之前整体发生过什么”

## 4. 当前实现

### 4.1 新增的 history mode

在 `src/action_prediction/evaluate_vlm.py` 里新增了：

- `--history-mode agentic_summary_recent`

相关参数：

- `--recent-k`
- `--memory-model`
- `--memory-rate-limit`

默认使用方式示例：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_task \
  --history-mode agentic_summary_recent \
  --recent-k 3 \
  --memory-model gpt-4o-mini
```

### 4.2 复用了 agentic memory 的在线压缩逻辑

`evaluate_vlm.py` 复用了 `evaluate_agentic_memory_task.py` 中的 memory building 逻辑。

关键点是：

- 对每个新 step 比较前后截图
- 生成 `recent_change` 或 `recent_change_keyframe`
- 当 recent buffer 过长时，只压缩 older prefix
- 保留滚动的 `older_summary`

然后在 hybrid 模式里：

- 取 `older_summary`
- 再拼上最近 `k` 步结构化 `recent_buffer`
- 如果 recent item 带 `keep_image=true`，则把对应 screenshot 作为 history image 传给 VLM

### 4.3 当前只保留少量 memory 关键帧图

当前实现的原则不是堆很多历史图片，而是只接 memory agent 显式保留的少量关键帧。

当前设计重点是：

- 远程信息用 summary 进文本
- 最近 `3` 步用结构化 recent buffer 保真
- 只引入 `keep_image=true` 的少量关键截图

这样做的好处是：

- 上下文长度稳定
- 不容易因为历史图片过多导致视觉注意力被稀释
- 同时不会浪费 memory agent 已经挑出来的关键视觉状态

## 5. 上下文是否够用

这条方法设计里，真正需要控制的是历史组织方式，而不是盲目扩大量。

当前判断是：

- 文本上下文通常不是瓶颈
- 更大的风险是压缩方式是否会损伤近程细节
- 图片历史需要严格控量，避免因为历史图过多让视觉注意力分散

因此目前这条方法线的重点应当放在：

- 如何组织历史
- 如何保留近程精细信息

而不是优先担心 token 上限。

## 6. 推荐默认策略

当前方法建议如下：

1. 近程保留结构化 `recent_buffer`
2. 远程使用 `agentic memory` 生成 `older_summary`
3. 当前默认先用 `recent-k = 3`
4. 只保留 memory agent 显式选择的关键帧图，而不是堆很多远程图片

也就是：

- 近程解决局部操作 grounding
- 远程解决长程任务进展与分支状态

## 7. 注意事项

实现这条方法时，需要特别注意两个点：

1. 不要只保留 `older_summary`，否则 recent memory 的结构化信息会被浪费
2. 不要丢掉 `keep_image=true` 选出的关键截图，否则 memory agent 的关键帧选择就没有真正进入 VLM 上下文

也就是说，完整的 hybrid 必须同时保留：

- 文本上的 `older_summary`
- 文本上的结构化 `recent_buffer`
- 图像上的少量 `keep_image=true` keyframe

## 9. 相关文件

实现和实验相关文件如下：

- `src/action_prediction/evaluate_vlm.py`
- `src/action_prediction/evaluate_agentic_memory_task.py`
- `scripts/run_vlm_recent5_step_8_9_10_15.sbatch`
- `scripts/run_vlm_agentic_summary_recent3_step_8_9_10_15.sbatch`
- `scripts/run_agentic_memory_task_step_8_9_10_15.sbatch`
- `tmp_action_uids_step_8_9_10_15.txt`

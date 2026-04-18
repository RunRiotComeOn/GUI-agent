# Mind2Web 静态数据集 + 官方指标 VLM 评测

这个目录现在已经接好了一个“尽量贴近官方流程”的静态评测方案：

- 数据集使用 `osunlp/Multimodal-Mind2Web`，也就是带截图的静态 Mind2Web。
- 候选元素排序使用官方仓库里提供的 `scores_all_data.pkl`。
- 最终打分继续复用官方 `ActionEvaluatorMultiChoice` 的多选 action metric。
- 推理接口先走 OpenAI 兼容的 `/chat/completions`，方便后面替换成 OpenAI 或兼容服务。

## 1. 准备数据

先安装依赖，然后把测试集和官方候选分数下载到本地：

```bash
python scripts/prepare_mind2web_static.py
```

默认会下载：

- `data/multimodal_mind2web/`
- `data/mind2web_aux/scores_all_data.pkl`

如果你还想把训练集也拉下来：

```bash
python scripts/prepare_mind2web_static.py --include-train
```

## 2. 跑一个最小 dry-run

先只检查数据筛选和命令参数，不实际调用模型：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_task \
  --limit 5 \
  --dry-run
```

## 3. 正式评测

最简单的完整命令：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --api-base https://api.openai.com/v1 \
  --split test_task \
  --limit 20 \
  --output-dir outputs/mind2web_vlm_test_task_20
```

如果用环境变量放 key：

```bash
set OPENAI_API_KEY=你的key
python src/action_prediction/evaluate_vlm.py --model gpt-4o-mini --split test_task --limit 20
```

## 4. 只跑部分任务

这个入口专门加了几组适合抽样评测的参数。

按前 N 条 action 跑：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_website \
  --limit 30
```

按区间跑：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_domain \
  --start-index 100 \
  --end-index 140
```

按任务 id 跑：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_task \
  --annotation-id 91695df8-f256-47c9-8c37-06e8d0fc758f
```

按通配符批量挑任务：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_task \
  --annotation-pattern "91695d*"
```

从文件里读取任务 id：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_task \
  --annotation-id-file .\\task_ids.txt
```

按网站或 domain 过滤：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_domain \
  --website united \
  --domain travel
```

## 5. HTML-only 对照

如果你想先和官方 text-only LLM 方式做个近似对照，可以不发图片：

```bash
python src/action_prediction/evaluate_vlm.py \
  --model gpt-4o-mini \
  --split test_task \
  --limit 20 \
  --html-only
```

## 6. 输出文件

输出目录下会保留官方 evaluator 一样的三类结果文件：

- `*_outputs_topK.json`
- `*_predictions_topK.json`
- `*_results_topK.json`

其中最重要的是 `*_results_topK.json`，建议主要看：

- `marco_step_acc`
- `marco_element_acc`
- `marco_action_f1`

官方仓库在 2023-10-30 的更新里特别说明过，论文对比建议优先看 macro average。

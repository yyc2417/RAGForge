# 评测 v2 · 设计

> 状态：统一 schema、校验器、CRUD-RAG Read 120 题、多格式 45 题、bge-m3/CUDA 检索基线、CI 回归门禁已实现；多轮（10×3）与 clarify（16 例）数据集已建；本地/云端 4 模式全量已完成。2026-10-21 起端到端评测把 clarify / 生成降级 / 空回答单列，并支持可选 LLM 语义判分；RGB 适配待执行。

## 1. 目标

- 用同一套 schema 承接自定义集与公开集；
- 区分 dev / test，test 冻结，不在 test 上调参；
- 检索指标与生成指标分开报表；
- 本地模型先跑 dev，最终本地 + API 各跑一次 test。

## 2. 统一 schema

```json
{
  "id": "q001",
  "query": "远程办公每周最多几天？",
  "query_type": "factual",
  "difficulty": "L1",
  "answerable": true,
  "expected_behavior": "answer",
  "expected_sources": ["company_handbook.md"],
  "evidence": [{"source": "company_handbook.md", "h1": "远程办公"}],
  "gold_keywords": ["两天"],
  "reference_answer": "每周最多两天。",
  "split": "test",
  "tags": ["single-hop"]
}
```

clarify 样本额外使用 `clarification_keywords`（如 `["星云笔记", "哪个产品"]`），
用于检查澄清问题是否点到了缺失实体；`expected_sources` 可保留为澄清后的参考来源，
但不参与当前召回/忠实度统计。

- `expected_behavior`：answer / refuse / clarify / chitchat；
- `expected_sources`：支持多跳（列表）；
- `gold_keywords` 与 `reference_answer` 二选一即可：自定义集用关键词判召回，
  公开集（如 CRUD-RAG）通常只有参考答案，校验器接受 `reference_answer` 兜底；
- `evidence`：可选，保留 chunk/page/span 级信息；
- `split`：dev / test；
- 兼容旧字段 `expected_source`，加载时自动转成列表。

实现：`src/evaluation/schema.py`；校验 CLI：`scripts/validate_eval_dataset.py`。

## 3. 数据分层

| 数据集 | 用途 | 规模 |
|---|---|---|
| `tests/eval_dataset.json` | v1 冻结回归（旧 45 题） | 45 |
| `tests/eval_dataset_hard.json` | 用户点踩晋升的难例 | 动态 |
| CRUD-RAG Read | 中文公开泛化 | 100–200 |
| RGB 子集 | 噪声/拒答/反事实 | 可选 100 |
| 自定义多格式 | PDF/HTML/DOCX/CSV/JSONL、多跳、来源 | 40–60 |
| 多轮 / clarify / 优质库 | 交互与沉淀 | 10 会话 × 3 轮 / 16 / 6–10 |

公开集不提交仓库，放 `data/public_eval/` 并 gitignore。

## 4. 评测模式与指标

检索层（离线，可进 CI）：

- vector / bm25 / hybrid / hybrid+reranker；
- doc 级 `hit_any` / `hit_all` / `MRR_any` / `NDCG@5`（二值相关性，按来源去重）；
- 旧 v1 仍保留 chunk 级 Recall@5 / MRR@5；
- CI 用多格式集 + hybrid 粗排跑回归门禁（`--skip-reranker --min-hit-any --min-mrr`）。

端到端（全量 LLM）：

- baseline / linear_hybrid / agent / agent_reranker；
- **分层口径**：clarify 只计入 `clarify_rate`，不进入召回分母；`generation_failed` /
  空回答只单列，不进入忠实度、F1、语义判分分母；
- 检索层：`recall_top5` / `mrr_at5` 只在“实际发生检索的可答题”上计算；
- 生成层：忠实度、词面 F1、语义判分只在“实际作答样本”上计算；
- 行为层：`clarify_rate`、`expected_clarify_count`、`clarify_accuracy`、
  `over_clarification_rate`、`under_clarification_rate`、
  `clarification_keyword_hit_rate`、`degraded_count`、`empty_answer_count`、
  拒答准确率、编造率；
- 语义判分默认关闭，`--semantic-judge` 开启，`--judge-limit N` 控制每模式判分条数；
- P50 + max 延迟、token、错误/未验证。

交互层：

- 多轮：轮级命中、会话成功率、会话隔离；
- clarify：澄清准确率、过度澄清率、澄清问题关键词命中率；`--with-followup`
  额外统计补充后来源命中率；
- 优质库：归档幂等、优质命中、无关不误召回、移除回滚。

## 5. 本地 / API 协议

1. 冻结 test 集与 dataset hash；
2. 同一 embedding、chunk、top_k、reranker、judge；
3. 本地模型先跑 dev / smoke；
4. 最终在同一 git tag 上跑本地 test 和 API test；
5. 报告记录 provider / model / commit / hash / judge 版本。

检索层报告额外记录 `dataset_name / dataset_version / dataset_license /
dataset_hash / k`，用于复现与许可追溯。

判官固定（DeepSeek 或另一固定 judge），抽 20 样本人工复核；本地模型自判只用于
链路 smoke，不作为模型结论。当前 `--semantic-judge` 复用本次运行的 LLM 配置，
正式对比时应显式固定外部 judge；词面 F1 仅作辅助指标。

## 6. 命令

```bash
# schema / 标签 / 来源校验
uv run python scripts/validate_eval_dataset.py --corpus-dir data/sample

# 公开数据字段映射（CRUD-RAG / RGB / MultiHop-RAG 等）
uv run python scripts/adapt_public_eval.py \
  --input data/public_eval/crud_rag_read.jsonl \
  --output data/public_eval/crud_read_eval.json \
  --query-field question --answer-field answer \
  --source-field doc_id --keywords-field keywords --split test

# CRUD-RAG 官方 split → Read 子集（120 题，含本地语料）
uv run python scripts/import_crud_rag.py --limit 120
uv run python scripts/eval_v2.py \
  --dataset data/public_eval/crud_read_eval.json \
  --corpus-dir data/public_eval/crud_corpus \
  --report reports/eval_v2_crud_read_report.json \
  --dataset-name CRUD-RAG-Read --dataset-version crud_split_2024 \
  --dataset-license Apache-2.0

# 检索层 v2 基线
uv run python scripts/eval_v2.py --dataset tests/eval_dataset.json --corpus-dir data/sample

# 端到端 v2（4 模式，需 LLM；有 reference_answer 时附词面 token F1）
uv run python scripts/eval.py \
  --dataset tests/eval_dataset_multiformat.json --corpus-dir data/demo \
  --rebuild --report reports/eval_e2e_multiformat.json

# 可选：LLM 语义判分（每模式最多 20 条，控制本地/云端成本）
uv run python scripts/eval.py \
  --dataset tests/eval_dataset_multiformat.json --corpus-dir data/demo \
  --semantic-judge --judge-limit 20 --report reports/eval_e2e_multiformat_judged.json

# 对照：关闭 clarify 跑同一数据集（回答质量才可比）
uv run python scripts/eval.py \
  --dataset tests/eval_dataset_multiformat.json --corpus-dir data/demo \
  --clarify off --report reports/eval_e2e_multiformat_noclarify.json

# clarify 判定 + 多轮会话（需真实 LLM）
uv run python scripts/eval_interaction.py --mode both --limit 5
uv run python scripts/eval_interaction.py --mode clarify --with-followup
# 报告：reports/eval_interaction_report.json（含 commit / model / base_url）

# 多格式集基线（45 题）
uv run python scripts/eval_v2.py \
  --dataset tests/eval_dataset_multiformat.json --corpus-dir data/demo \
  --rebuild --report reports/eval_v2_multiformat_report.json \
  --dataset-name multiformat-demo --dataset-version 2026-10-18 \
  --dataset-license internal

# CI 回归门禁（多格式集 + hybrid，跳过 reranker）
uv run python scripts/eval_v2.py \
  --dataset tests/eval_dataset_multiformat.json --corpus-dir data/demo \
  --rebuild --skip-reranker --report reports/ci_retrieval_report.json \
  --gate-mode hybrid --min-hit-any 0.85 --min-mrr 0.60

# 旧 4 模式端到端（全量 LLM，后置）
uv run python scripts/eval.py --rebuild
```

## 7. 离线检索基线（2026-10-18）

> **口径提醒**：以下 2026-10-20 / 10-21 的端到端报告生成于旧口径——clarify 的空
> `answer` 被计入召回分母，空回答还可能被忠实度判官判为 `True`。新代码已修正
> （见本节末），旧报告保留用于历史对照；主表复跑后再替换。

### CRUD-RAG Read

120 题（1doc / 2docs / 3docs 各 40），文档级 top-5，CUDA embedding
（BAAI/bge-m3，1024 维），独立评测索引，不调 LLM：

| 模式 | hit_any | hit_all | MRR_any | NDCG@5 |
|------|---:|---:|---:|---:|
| vector | 100.0% | 95.0% | 98.2% | 97.3% |
| bm25 | 99.2% | 90.0% | 96.9% | 94.7% |
| hybrid | 100.0% | 92.5% | 96.5% | 95.5% |
| reranker | 100.0% | 94.2% | 97.9% | 96.9% |

解读：换 bge-m3 后 vector hit_any 60.0%→100.0%，hybrid hit_all 65.8%→92.5%；
BM25 兜底保持稳定。报告：`reports/eval_v2_crud_read_report.json`
（dataset_hash `2b087ab38af08ad2`，Apache-2.0）。

### 多格式自定义集

45 题覆盖 8 个 demo 文件、7 种格式，含 4 道多跳、10 道跨语言样本，
CUDA embedding（BAAI/bge-m3）：

| 模式 | hit_any | hit_all | MRR_any | NDCG@5 |
|------|---:|---:|---:|---:|
| vector | 100.0% | 100.0% | 95.6% | 96.7% |
| bm25 | 93.3% | 91.1% | 87.4% | 88.1% |
| hybrid | 100.0% | 100.0% | 91.3% | 94.1% |
| reranker | 100.0% | 97.8% | 97.8% | 97.5% |

换 bge-m3 后 vector/hybrid hit_any 到 100%，跨语言失败消除；reranker MRR 97.8%。
报告：`reports/eval_v2_multiformat_report.json`。

### Embedding 选型对照

三个候选均为 1024 维、CUDA、normalize、相同 chunk 与语料：

| 模型 | 体积 | CRUD vector hit_all | CRUD reranker hit_all | 多格式 hybrid hit_any | 多格式 reranker NDCG@5 |
|---|---:|---:|---:|---:|---:|
| **bge-m3（当前）** | 2.2GB | **95.0%** | 94.2% | **100.0%** | 97.5% |
| bge-large-zh-v1.5 | 1.3GB | 88.3% | 95.0% | 100.0% | 97.5% |
| Qwen3-Embedding-0.6B | 1.2GB | 93.3% | **95.8%** | 97.8% | 97.5% |

选型：bge-m3 在多格式 hybrid 与跨语言均衡上最好；Qwen3 的 CRUD reranker 略高，
但多格式 hybrid hit_any 下降，因此保留 bge-m3。报告：
`reports/eval_v2_{crud,multiformat}_{bge_large_zh,qwen3_06b}.json`。

### 本地模型 smoke（2026-10-18）

本地 `Qwen3.6-35B-A3B-NVFP4`（OpenAI 兼容 endpoint）+ agent 模式，CRUD-RAG
前 2 题（bge-m3）：Recall@5 / MRR@5 100%，忠实度 100%，answer F1 81.9%，
P50 143.8s，max 188.4s。clarify 16 例带 follow-up：准确率 87.5%、过度澄清
20.0%、问题关键词 90.9%、补充后解决率 83.3%（前一次无 follow-up 为 93.8%）。
多轮全量 10 会话 × 3 轮：轮级命中 100%、会话成功率 100%；修正前 v1 为
63.3% / 50%，根因是首轮缺主体与 condense 丢实体。样本仅用于本地链路验证，
不作为模型结论。报告：`reports/local_e2e_crud_smoke.json`、
`reports/local_clarify_followup.json`、`reports/local_multiturn_full.json`。

### 本地全量 4 模式（2026-10-20）

CRUD-RAG 120 题：baseline / linear / agent / agent_reranker 的 Recall@5 为
100.0 / 100.0 / 99.2 / 98.3%，MRR 98.2 / 96.5 / 94.2 / 97.1%，忠实度
100 / 100 / 100 / 99.2%，P50 80.4 / 62.3 / 91.5 / 132.4s。

多格式 45 题：baseline / linear / agent / agent_reranker 的 Recall@5 为
100.0 / 100.0 / 80.0 / 80.0%，MRR 94.4 / 91.1 / 76.7 / 78.9%。Agent 的
80% 来自 9/45 题被 clarify（题目缺少主体），检索基线仍是 100%。
报告：`reports/local_e2e_crud_full.json`、`reports/local_e2e_multiformat_full.json`。

### 云端全量 4 模式（2026-10-21）

USTC `deepseek-flash`，云端生成 + 本地 bge-m3 检索。补跑后两份报告均 0 errors：

- CRUD 120 题：baseline / linear / agent / agent_reranker 的 Recall@5 为
  100.0 / 100.0 / 97.5 / 97.5%，MRR 98.2 / 96.5 / 95.4 / 96.3%，
  忠实度 100%，answer F1 55.2 / 55.6 / 55.0 / 53.6%。
- 多格式 45 题：Recall@5 为 100.0 / 100.0 / 68.9 / 68.9%；
  Agent 的 68.9% 来自 14/45 clarify，检索基线仍为 100%。
- 云端总 token 约 2.19M；服务端拥堵导致 51 条超时/降级日志，延迟不作为性能结论。
  报告：`reports/cloud_e2e_crud_full.json`、`reports/cloud_e2e_multiformat_full.json`。

### 评测口径修正（2026-10-21）

- `clarify` 节点返回空 `answer`；旧版 `eval.py` 仍把它计入召回分母，导致多格式
  Agent 的 80.0%（本地 9/45）与 68.9%（云端 14/45）被误读为检索退化；
- 旧版空回答会被 LLM 忠实度判官判为 `True`，出现“14 条空回答但忠实度 100%”；
- 新口径：`clarified` 单列并输出 `clarify_rate`；召回只在“实际检索”的可答题上
  统计；忠实度 / F1 / 语义判分只在“实际作答”样本上统计；空回答与
  `generation_failed` 单独计数；
- `generate` 节点 LLM 失败现在写入 `generation_failed` / `generation_error`，
  跳过 verify，并在 UI / API / 会话 metrics / 评测报告中透传；
- `--clarify off` 提供关闭 clarify 的对照跑；`--semantic-judge` 提供可选的
  LLM 语义判分（默认关闭，避免本地 35B 与云端 20rpm 下成本失控）；
- 多格式 45 题人工复核：37 题 `answer`，8 题缺少产品/设备/说明书主体、标为
  `clarify`（mf-025/027/029/030/031/032/036/037）。数据集文件 hash
  `e2010df3c478d180`；旧端到端报告 hash 为 `5846bc120a9cce7c`，两者不可混用。
  检索基线不受行为标签影响，但 clarify 准确率 / 过度澄清率需在新报告中重算。

## 8. 待办

- 用 2026-10-21 新口径复跑本地 / 云端 4 模式矩阵，旧报告保留对照；
- 语义判分抽 20 条人工复核，并固定外部 judge（避免本地模型自判）；
- 多格式 clarify 标注已落地（8 clarify / 37 answer），待复跑验证过度/未澄清率；
- RGB 噪声/拒答/反事实子集适配；
- 云端 runner 级限速/重试与重复跑方差。

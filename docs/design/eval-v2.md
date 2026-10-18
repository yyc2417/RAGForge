# 评测 v2 · 设计

> 状态：统一 schema、校验器、检索层基线已实现；CRUD-RAG/RGB 适配与全量 LLM 双跑待执行。

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

- `expected_behavior`：answer / refuse / clarify / chitchat；
- `expected_sources`：支持多跳（列表）；
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
| 多轮 / clarify / 优质库 | 交互与沉淀 | 10 会话 / 10–20 / 6–10 |

公开集不提交仓库，放 `data/public_eval/` 并 gitignore。

## 4. 评测模式与指标

检索层（离线，可进 CI）：

- vector / bm25 / hybrid / hybrid+reranker；
- doc 级 `hit_any` / `hit_all` / `mrr_any`；
- 旧 v1 仍保留 chunk 级 Recall@5 / MRR@5。

端到端（全量 LLM，后置）：

- baseline / linear_hybrid / agent / agent_reranker；
- 忠实度、拒答、编造率、答案正确性；
- P50 + max 延迟、token、错误/未验证。

交互层：

- 多轮：轮级命中、会话成功率、会话隔离；
- clarify：澄清准确率、过度澄清率、补充后解决率；
- 优质库：归档幂等、优质命中、无关不误召回、移除回滚。

## 5. 本地 / API 协议

1. 冻结 test 集与 dataset hash；
2. 同一 embedding、chunk、top_k、reranker、judge；
3. 本地模型先跑 dev / smoke；
4. 最终在同一 git tag 上跑本地 test 和 API test；
5. 报告记录 provider / model / commit / hash / judge 版本。

判官固定（DeepSeek 或另一固定 judge），抽 20 样本人工复核；本地模型不自判。

## 6. 命令

```bash
# schema / 标签 / 来源校验
uv run python scripts/validate_eval_dataset.py --corpus-dir data/sample

# 公开数据字段映射（CRUD-RAG / RGB / MultiHop-RAG 等）
uv run python scripts/adapt_public_eval.py \
  --input data/public_eval/crud_rag_read.jsonl \
  --output tests/eval_dataset_public.json \
  --query-field question --answer-field answer \
  --source-field doc_id --keywords-field keywords --split test

# 检索层 v2 基线
uv run python scripts/eval_v2.py --dataset tests/eval_dataset.json --corpus-dir data/sample

# clarify 判定 + 多轮会话（需真实 LLM）
uv run python scripts/eval_interaction.py --mode both --limit 5

# 旧 4 模式端到端（全量 LLM，后置）
uv run python scripts/eval.py --rebuild
```

## 7. 待办

- CRUD-RAG Read adapter 与 100–200 题子集；
- RGB noise/rejection/integration/counterfactual adapter；
- 多轮 / clarify 真实 LLM 跑分（数据与 `eval_interaction.py` 已就绪）；
- 评测 v2 端到端矩阵与 README 报表；
- 本地 + API 双跑。

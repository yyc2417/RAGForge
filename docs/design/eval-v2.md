# 评测 v2 · 设计

> 状态：统一 schema、校验器、检索层基线、CRUD-RAG Read 120 题与离线基线已实现；RGB 适配与全量 LLM 双跑待执行。

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
  --output data/public_eval/crud_read_eval.json \
  --query-field question --answer-field answer \
  --source-field doc_id --keywords-field keywords --split test

# CRUD-RAG 官方 split → Read 子集（120 题，含本地语料）
uv run python scripts/import_crud_rag.py --limit 120
uv run python scripts/eval_v2.py \
  --dataset data/public_eval/crud_read_eval.json \
  --corpus-dir data/public_eval/crud_corpus \
  --report reports/eval_v2_crud_read_report.json

# 检索层 v2 基线
uv run python scripts/eval_v2.py --dataset tests/eval_dataset.json --corpus-dir data/sample

# clarify 判定 + 多轮会话（需真实 LLM）
uv run python scripts/eval_interaction.py --mode both --limit 5
# 报告：reports/eval_interaction_report.json（含 commit / model / base_url）

# 旧 4 模式端到端（全量 LLM，后置）
uv run python scripts/eval.py --rebuild
```

## 7. 离线检索基线（2026-10-18）

### CRUD-RAG Read

120 题（1doc / 2docs / 3docs 各 40），文档级 top-5，CPU embedding
（all-MiniLM-L6-v2），独立评测索引，不调 LLM：

| 模式 | hit_any | hit_all | MRR_any |
|------|---:|---:|---:|
| vector | 60.0% | 30.0% | 50.5% |
| bm25 | 99.2% | 90.0% | 96.9% |
| hybrid | 99.2% | 65.8% | 80.5% |
| reranker | 99.2% | 89.2% | 97.9% |

解读：英文 embedding 在中文公开集上向量单路明显偏弱；BM25 兜底后 hit_any
接近饱和。hybrid 的 top-10 候选池压缩了多跳 hit_all，reranker 精排可恢复到
89.2%。报告：`reports/eval_v2_crud_read_report.json`（dataset_hash
`2b087ab38af08ad2`）。

### 多格式自定义集

45 题覆盖 8 个 demo 文件、7 种格式，含 4 道多跳、10 道跨语言样本：

| 模式 | hit_any | hit_all | MRR_any |
|------|---:|---:|---:|
| vector | 80.0% | 80.0% | 61.0% |
| bm25 | 93.3% | 91.1% | 87.4% |
| hybrid | 91.1% | 88.9% | 76.9% |
| reranker | 93.3% | 91.1% | 92.2% |

剩余未命中集中在中文 query 检索英文 PDF/DOCX；格式解析本身正常。
报告：`reports/eval_v2_multiformat_report.json`。

## 8. 待办

- RGB 噪声/拒答/反事实子集适配；
- CRUD-RAG / 多格式集端到端生成评测与结果报表；
- 多轮 / clarify 真实 LLM 跑分（数据与 `eval_interaction.py` 已就绪）；
- 评测 v2 端到端矩阵与 README 报表；
- 本地 + API 双跑。

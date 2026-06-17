# ADR-002：为什么用 RRF 融合向量和 BM25 检索，而非分数归一化方案

> **状态**：已采纳
> **日期**：2026-06-14
> **决策者**：宇诚

## 背景

RAGForge 的混合检索（`src/retrieval/hybrid.py`）需要同时利用向量检索（ChromaDB 语义相似度）和 BM25（rank-bm25 + jieba 关键词匹配）两路结果。向量检索返回 relevance score ∈ [0, 1]（余弦相似度），而 BM25 返回无界正分数（通常 0~20+，取决于文档长度和词频）。两者分数量纲完全不同，无法直接比较或相加，需要一种融合策略将两路排名合并为统一排序。

检索组件遵循统一的 `search` 与 `search_with_scores` 接口规范（编码规范），`HybridRetriever` 也需实现这两个方法以便与 `VectorStore`、`BM25Retriever` 组合与替换。

## 备选方案

### 方案 A：RRF（Reciprocal Rank Fusion）— 选定
- **概述**：`score(d) = Σ 1/(rrf_k + rank(d))`，rrf_k=60（`settings.rrf_k` 标准值）。仅使用文档在各自列表中的排名，不使用原始分数
- **优点**：
  - 零调参：不需要归一化、不需要权重，rrf_k=60 是经验证的最优默认值
  - 鲁棒性强：排名不受离群分数影响，新文档入库不影响已有排名计算
  - 天然可比：不同检索器的排名天然具有可比性，无需假设分数分布
  - 实现简单：字典累加排名倒数 + 排序，无需额外计算开销
- **缺点**：
  - 完全忽略原始分数信息，两条结果分数差 0.01 和差 0.5 在排名相同时贡献相同
  - 小型知识库上可能引入噪声
- **决定**：选，最稳健且零调参

### 方案 B：Min-Max 归一化后加权求和
- **概述**：`norm(s) = (s - min) / (max - min)`，然后 `final = α·norm_vec + (1-α)·norm_bm25`
- **优点**：保留分数信息，直觉上更"精确"
- **缺点**：
  - 需要全局 min/max，对离群值极度敏感
  - 新文档入库后 min/max 可能变化，需重算或维护统计量
  - 权重 α 需要调参
- **决定**：不选，对离群值敏感，维护成本高

### 方案 C：Z-score 标准化后加权求和
- **概述**：`z(s) = (s - μ) / σ`，然后加权求和
- **优点**：比 Min-Max 更鲁棒，考虑了分数分布
- **缺点**：
  - 假设分数近似正态分布——BM25 的分数分布明显偏态，假设不成立
  - 需要维护 μ 和 σ，权重仍需调参
- **决定**：不选，分布假设不成立

### 方案 D：分位数归一化
- **概述**：按分位数映射到 [0, 1]
- **优点**：不受分布假设限制
- **缺点**：每次检索都需要对结果排序计算分位数，计算成本高；对小结果集不稳定
- **决定**：不选，计算开销不合理

## 决定的理由

RRF 是唯一不依赖分数可比性的方案。向量分数和 BM25 分数本质上不可比（一个是余弦相似度，一个是概率模型得分），而排名天然可比。Cormack et al. (2009) 的论文系统验证了 RRF 在多种检索融合场景下的有效性。

实际实现要点（`src/retrieval/hybrid.py`）：

```python
# HybridRetriever.search_with_scores() 核心逻辑
candidate_k = max(k * 2, k + 1)  # 每路扩展至 2k 候选，扩大召回池
v_docs, _ = self._vector.search_with_scores(query, k=candidate_k)
b_docs, _ = self._bm25.search_with_scores(query, k=candidate_k)

# RRF 融合：用文档内容前 80 字符作为去重键
for rank, doc in enumerate(v_docs, start=1):
    key = doc.page_content[:80]  # 去重键，避免同一文档被两路重复计入
    rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
```

关键设计：候选扩展至 `2k` 提升融合质量，内容前 80 字符作为去重键平衡去重精度与计算开销。

## 风险与缓解

| 风险 | 可能性 | 缓解措施 |
|------|--------|----------|
| 小型知识库（如 10 chunks）上 RRF 引入噪声 | 中 | Reranker（CrossEncoder bge-reranker-v2-m3）作为第二层修正排名 |
| 忽略原始分数导致排序不够精确 | 低 | 评估数据集验证，必要时调整 `settings.rrf_k` 参数 |

## 参考资料

- [Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Elasticsearch RRF 文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html)
- [src/retrieval/hybrid.py](file://src/retrieval/hybrid.py)

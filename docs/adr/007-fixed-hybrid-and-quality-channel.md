# ADR-007：固定 hybrid 检索与加权 RRF 优质通道

> **状态**：已采纳（固定 hybrid 已落地；优质通道 Phase 3 落地）
> **日期**：2026-10-18
> **决策者**：宇诚

## 背景

RAGForge 原来的检索策略是逐级升级链：

- `decide` 按查询类型选择 `vector` / `bm25` / `hybrid`；
- `evaluate` 诊断低召回后进入 `switch_strategy`，按 `vector → bm25 → hybrid` 单向升级；
- 已是 `hybrid` 时短路生成。

这条链路存在两个问题：

1. **策略选择是纯规则**，不构成 Agent 智能，却增加了节点、条件边和面试口径的复杂度；
2. **逐级升级会产生无效检索轮次**，低召回时先试弱策略，再升级到 hybrid，成本和延迟都不划算。

同时，项目要新增“优质问答归档”：用户点赞的问答会整理成优质库文档，并希望在后续检索中优先召回。需要明确优质文档的加权方式和边界。

## 决策

### 1. 固定 hybrid 检索

所有需要检索的查询直接使用 hybrid：

- 向量检索（ChromaDB）+ BM25 + RRF 融合；
- reranker 可用时对融合结果重排；
- 不再从 `vector` / `bm25` 起步逐级升级。

`decide` 只输出 `none / hybrid`：寒暄或不检索走 `none`，其余走 `hybrid`。

### 2. 删除 switch_strategy，保留 7 节点主链路

删除 `switch_strategy` 节点、条件边和策略升级链，主链路为：

```text
analyze → decide → retrieve → evaluate → reformulate → generate → verify
```

`evaluate` 只输出 `proceed / reformulate`：

- 检索充分 → `proceed`；
- 空结果、低召回、结果无关 → `reformulate`；
- 达到 `max_iterations` → 强制生成。

`reformulate` 与 `verify` 两个自愈闭环保留。

> 拓扑口径：旧 8 节点含 `switch_strategy`；固定 hybrid 后为 7 节点主链路。Phase 1 新增 clarify 终止分支后为 8 节点（含 clarify，不含 switch），见 ADR-008。

### 3. 优质库加权：加权 RRF 优质通道

优质库归档文档的 chunk metadata 标记：

```text
source_type = "qa_archive"
```

检索时在普通 hybrid 之外增加一个“优质通道”：

- 给 `VectorStore.search_with_scores` 增加 `filter` 参数；
- 用 Chroma metadata filter 查 `source_type=qa_archive`；
- 按 `quality_weight / (rrf_k + rank)` 计入融合分；
- BM25 不参与优质通道，避免再维护一套过滤索引；
- 未过最低相关度阈值的优质文档不参与加权；
- 优质库为空时跳过该通道。

默认参数：

```text
quality_weight = 1.3
quality_top_k = 3
quality_min_relevance = 0.3
quality_max_per_source = 2
```

优质通道属于检索层加权，不是新的 Agent 策略，也不改变节点拓扑。

## 备选方案

### 方案 A：保留策略升级链

- 优点：保留“按查询类型选策略”的演示点；
- 缺点：规则路由不构成 Agent 智能；逐级升级产生无效轮次；拓扑和文档复杂度高；
- 结论：不选。

### 方案 B：固定 hybrid + 融合后简单 boost

- 优点：实现最小，只在 RRF 分数上给优质文档乘系数；
- 缺点：只能重排已经进入候选池的文档，不能保证优质库被召回；
- 结论：不选，作为降级方案记录。

### 方案 C：固定 hybrid + 独立优质索引/通道

- 优点：召回保证更强；
- 缺点：需要额外维护一套索引或 BM25 过滤；
- 结论：当前规模不必要，先用 Chroma filter 的单通道加权。

## 后果

- 主链路从 8 节点（含 switch）精简为 7 节点，文档、测试和演示同步更新；
- 旧 8 节点指标（如 switch 贡献的 +23.4pp）变为历史值，重跑前不能当现状；
- 单次检索同时跑向量与 BM25，成本高于单路，但省掉弱策略试错；
- 优质通道需要 `VectorStore` 支持 metadata filter，并校准权重/阈值；
- 优质库不是事实正确，仍需人工移除错误归档；
- 旧 8 节点快照保留在 git tag `pre-7node`，可回滚。

## 关联文档

- [ADR-002：RRF 融合向量和 BM25](002-rrf-hybrid-retrieval.md)
- [ADR-006：检索分数语义与评估口径](006-score-semantics-and-eval-calibration.md)
- [ADR-008：clarify 终止分支](008-clarify-terminal-node.md)
- [Agent 模块文档](../modules/agent.md)
- [检索层模块文档](../modules/retrieval.md)
- [升级补全方案](../../RAGForge-升级补全方案.md)（私有）

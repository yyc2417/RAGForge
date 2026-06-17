# Task 6：混合检索与重排序

## 阶段概述

本阶段实现混合检索（Hybrid Retrieval）和重排序（Reranker），是检索系统的进阶优化。Task 6 将 Task 3 的向量检索和 BM25 用 RRF 算法融合，并用 CrossEncoder 重排序进一步提升精度。这是"检索质量"的关键提升阶段。

**前置依赖**：Task 1-5（配置、文档处理、检索、生成、Agent）  
**后续依赖**：Task 7（API 服务）、Task 8（评估系统的 hybrid/reranker 模式）

---

## 核心知识点

### 知识点 1：RRF 算法推导

RRF（Reciprocal Rank Fusion）是基于排名的融合算法，公式为：

$$\text{RRF\_score}(d) = \sum_{i=1}^{n} \frac{1}{k + \text{rank}_i(d)}$$

其中 `rank_i(d)` 是文档 `d` 在第 `i` 路检索结果中的排名（1-based），`k` 是平滑常数（标准值 60）。

```python
class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever, rrf_k: int | None = None):
        self._vector = vector_store
        self._bm25 = bm25
        self._rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
    
    def search_with_scores(self, query: str, k: int | None = None):
        k = k or settings.retrieval_k
        candidate_k = max(k * 2, k + 1)  # 每路取 top-2k 候选
        
        v_docs, _ = self._vector.search_with_scores(query, k=candidate_k)
        b_docs, _ = self._bm25.search_with_scores(query, k=candidate_k)
        
        # RRF 融合：用文档内容前 80 字符作为去重键
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        
        for rank, doc in enumerate(v_docs, start=1):
            key = self._doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
            doc_map[key] = doc
        
        for rank, doc in enumerate(b_docs, start=1):
            key = self._doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
            doc_map[key] = doc
        
        # 按 RRF 分数降序取 top-k
        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
        docs = [doc_map[key] for key in sorted_keys]
        scores = [round(rrf_scores[key], 6) for key in sorted_keys]
        return docs, scores
    
    @staticmethod
    def _doc_key(doc: Document) -> str:
        """文档去重键：内容前 80 字符。"""
        return doc.page_content[:80]
```

**RRF 的直觉理解**：
- 排名第 1 的文档贡献 `1/(60+1) = 0.0164`
- 排名第 2 的文档贡献 `1/(60+2) = 0.0161`
- 排名第 100 的文档贡献 `1/(60+100) = 0.0063`

排名越靠前，贡献越大，但差异被 `k=60` 平滑（避免 top-1 过度主导）。

**k=60 的含义**：`k` 是平滑常数，控制排名差异的权重衰减速度：
- `k` 很小（如 1）：`1/(1+1) = 0.5` vs `1/(1+2) = 0.333`，top-1 和 top-2 差异大
- `k=60`：`1/(60+1) = 0.0164` vs `1/(60+2) = 0.0161`，top-1 和 top-2 差异小，更平滑

**RRF 不依赖分数可比性的原因**：向量检索返回 [0, 1] 的 relevance score，BM25 返回无界正分数，两者尺度不同无法直接加权。RRF 只用排名（rank），不用原始分数，因此无需归一化。这是 RRF 的核心优势——鲁棒且无需调参。

**为什么每路取 top-2k**：`candidate_k = max(k * 2, k + 1)` 扩大候选池。如果最终要 top-3，每路取 top-6，这样 RRF 融合时有更多候选参与排名，避免"某路检索的 top-4 被另一路的 top-1 挤出"的情况。

### 知识点 2：CrossEncoder vs BiEncoder

CrossEncoder 和 BiEncoder 是两种文本相似度计算方式：

**BiEncoder（双编码器）**：
- 分别编码 query 和 document 为向量，然后计算余弦相似度
- 优点：document 向量可预计算，检索时只需计算 query 向量，速度快
- 缺点：query 和 document 独立编码，无法捕获交叉注意力

**CrossEncoder（交叉编码器）**：
- 将 query 和 document 拼接为 `[CLS] query [SEP] document [SEP]`，一起输入 Transformer
- 优点：query 和 document 做交叉注意力，精度更高
- 缺点：无法预计算，每对 (query, document) 都要完整推理，速度慢

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def rerank(self, query: str, documents: list[Document], top_n: int | None = None):
        """对候选文档做交叉重排序。"""
        top_n = top_n or settings.reranker_top_n
        
        if not documents:
            return []
        
        self._ensure_loaded()
        if self._model is None:
            # 优雅降级：保持原顺序截断
            return documents[:top_n]
        
        # CrossEncoder 对 (query, doc) pair 打分
        pairs = [(query, doc.page_content) for doc in documents]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.warning(f"[reranker] 推理失败，降级为原顺序：{e}")
            return documents[:top_n]
        
        # 按分数降序排序取 top_n
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_n]
        result = [doc for doc, _ in ranked]
        return result
```

**CrossEncoder 适合重排序场景**：
1. **候选集小**：重排序的输入是前序检索的 top-k（如 5-10 个文档），不是全库
2. **精度优先**：重排序是最后一道筛选，精度比速度更重要
3. **延迟可接受**：5-10 对 (query, document) 的推理时间在百毫秒级，对端到端延迟影响可控

### 知识点 3：优雅降级模式（Graceful Degradation）

Reranker 模型（BGE-Reranker-v2-m3，约 560MB）下载可能失败，采用懒加载 + 失败标记 + 降级路径：

```python
class Reranker:
    """CrossEncoder 重排序器（懒加载单例 + 优雅降级）。"""
    
    _instance: "Reranker | None" = None
    _lock = threading.Lock()
    
    def __new__(cls, model_name: str | None = None) -> "Reranker":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._model = None
                instance._model_name = model_name or settings.reranker_model
                instance._load_attempted = False
                cls._instance = instance
            return cls._instance
    
    def _ensure_loaded(self) -> None:
        """懒加载模型（仅首次调用时尝试，失败后不再重试）。"""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"[reranker] 尝试加载模型: {self._model_name}")
            self._model = CrossEncoder(self._model_name)
            logger.info(f"[reranker] 模型加载成功，重排序可用")
        except Exception as e:
            logger.warning(f"[reranker] 模型加载失败，已降级为跳过重排序：{e}")
            self._model = None
    
    def is_available(self) -> bool:
        """Reranker 是否可用（模型加载成功）。"""
        self._ensure_loaded()
        return self._model is not None
```

**懒加载（Lazy Loading）**：模型不在 `__new__` 中加载，而在首次调用 `_ensure_loaded()` 时加载。这避免了 `Reranker()` 构造时就触发 560MB 下载（可能阻塞启动）。

**失败标记（`_load_attempted`）**：首次加载失败后，`_load_attempted = True`，后续调用直接返回，不再重试。这避免了每次 rerank 都尝试下载（网络不可用时每次都要等超时）。

**降级路径**：`self._model is None` 时，`rerank()` 直接返回 `documents[:top_n]`（保持原顺序截断），不影响主流程。

**`is_available()` 查询模式**：调用方先检查 `reranker.is_available()`，可用才调用 `rerank()`。`retrieve_node` 中的用法：

```python
if reranker is not None and getattr(reranker, "is_available", lambda: False)():
    docs = reranker.rerank(query, docs)
    scores = [0.0] * len(docs)  # rerank 后分数失去含义
```

**为什么 Reranker 选择单例**：
1. **模型加载成本高**：560MB 模型下载 + 加载到内存约需 10-30 秒
2. **内存占用大**：模型常驻内存约 2GB，多实例会导致内存溢出
3. **全局共享**：所有请求共用同一模型实例，线程安全（GIL 保护）

### 知识点 4：对比实验方法论

Task 8 的评估脚本跑 4 套配置对比实验：

```python
# scripts/eval.py
MODES = ["baseline", "agent", "hybrid", "reranker"]

def _get_pipeline(self, mode: str):
    if mode == "baseline":
        pipeline = (self._vector_store, self._llm)  # 线性 RAG
    else:
        hybrid = None
        reranker = None
        if mode in ("hybrid", "reranker"):
            hybrid = HybridRetriever(self._vector_store, self._bm25)
        if mode == "reranker":
            reranker = self._reranker
        pipeline = build_agent_graph_from_pipeline(
            vector_store=self._vector_store,
            bm25_retriever=self._bm25,
            llm_client=self._llm,
            hybrid_retriever=hybrid,
            reranker=reranker,
        )
    return pipeline
```

**控制变量**：4 套配置使用相同数据集（`eval_dataset.json`）、相同查询、相同 LLM（DeepSeek），只变化检索策略：
- **baseline**：Task 4 线性 RAG（vector-only，无 Agent）
- **agent**：Task 5 Agent 状态机（vector/bm25 策略，无混合）
- **hybrid**：Agent + HybridRetriever（RRF 融合，无 Reranker）
- **reranker**：Agent + HybridRetriever + Reranker（CrossEncoder 重排序）

**多维度指标**：
- **Top-K 召回率**：检索质量（是否找到正确文档）
- **幻觉率**：忠实度（1 - faithfulness_rate）
- **P50/P95/P99 延迟**：响应速度
- **Token 消耗**：成本

**如何分析实验结果**（来自 `reports/eval_report.json`）：

| 配置 | 召回率 | 幻觉率 | P99 延迟 | 平均 Token |
|------|:------:|:-----:|:-------:|:---------:|
| baseline | 100% | 0% | 3.8s | 695 |
| agent | 100% | 16.7% | 35.3s | 617 |
| hybrid | 70% | 16.7% | 58.8s | 641 |
| reranker | 100% | 16.7% | 89.6s | 626 |

**关键洞察**：
1. **hybrid 召回率下降的原因**：小型知识库（仅 10 chunks）上 RRF 融合引入 BM25 噪声（BM25 在小库上关键词匹配不够精准），稀释了向量检索的高质量结果
2. **reranker 的修正效果**：CrossEncoder 重排序把 70% 拉回 100%，证明 reranker 能有效修正 RRF 的缺陷
3. **延迟随复杂度递增**：baseline 单题 2-4s，agent 因 reformulate 循环升至 35s，reranker 额外叠加 CrossEncoder 推理

---

## 设计模式与架构决策

**融合模式（RRF）**：基于排名融合两路检索结果，无需归一化分数，鲁棒且无需调参。

**优雅降级模式（Reranker）**：懒加载 + 失败标记 + 降级路径，确保"锦上添花"的功能失败不影响主流程。

**去重模式（文档键）**：用内容前 80 字符作为去重键，避免同一文档被两路重复计入 RRF 分数。

**候选扩展模式（top-2k）**：每路取 2 倍候选，扩大 RRF 融合池，提升最终 top-k 质量。

---

## 关键代码解读

### hybrid.py：混合检索

```python
def search_with_scores(self, query: str, k: int | None = None):
    k = k or settings.retrieval_k
    candidate_k = max(k * 2, k + 1)  # 每路取 top-2k 候选，扩大召回池
    
    v_docs, _ = self._vector.search_with_scores(query, k=candidate_k)
    b_docs, _ = self._bm25.search_with_scores(query, k=candidate_k)
    
    # RRF 融合
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    
    for rank, doc in enumerate(v_docs, start=1):
        key = self._doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
        doc_map[key] = doc
    
    for rank, doc in enumerate(b_docs, start=1):
        key = self._doc_key(doc)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
        doc_map[key] = doc
    
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
    docs = [doc_map[key] for key in sorted_keys]
    scores = [round(rrf_scores[key], 6) for key in sorted_keys]
    
    logger.info(
        f"[hybrid] RRF 融合：vector={len(v_docs)}, bm25={len(b_docs)} "
        f"→ top-{k}（rrf_k={self._rrf_k}）"
    )
    return docs, scores
```

**设计意图**：两路各取 top-2k（`candidate_k`），然后用 RRF 融合取 top-k。`rrf_scores` 字典累积每个文档在两路中的 RRF 分数，`doc_map` 保存文档用于最终取回。`round(..., 6)` 避免浮点精度问题。

### reranker.py：重排序

```python
def rerank(self, query: str, documents: list[Document], top_n: int | None = None):
    top_n = top_n or settings.reranker_top_n
    
    if not documents:
        return []
    
    self._ensure_loaded()
    if self._model is None:
        # 优雅降级：保持原顺序截断
        logger.debug("[reranker] 模型不可用，跳过重排序（降级）")
        return documents[:top_n]
    
    # CrossEncoder 对 (query, doc) pair 打分
    pairs = [(query, doc.page_content) for doc in documents]
    try:
        scores = self._model.predict(pairs)
    except Exception as e:
        logger.warning(f"[reranker] 推理失败，降级为原顺序：{e}")
        return documents[:top_n]
    
    # 按分数降序排序取 top_n
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[:top_n]
    result = [doc for doc, _ in ranked]
    logger.info(
        f"[reranker] 重排序完成：{len(documents)} → {len(result)} 篇"
        f"（top_score={float(ranked[0][1]):.4f}）"
    )
    return result
```

**设计意图**：`pairs` 列表构造 (query, document) 对，`self._model.predict(pairs)` 批量推理（比逐个推理快）。`zip(documents, scores)` 将文档和分数配对，`sorted(..., key=lambda x: x[1], reverse=True)` 按分数降序排序。

---

## 踩坑记录

### 问题 1：RRF 融合后分数为 0 的文档被误判

**问题现象**：evaluate 节点将 hybrid 检索结果判为 `low_recall`，但实际检索到了相关文档。

**排查过程**：
1. 打印 `retrieval_scores` 发现某些分数为 0.0
2. 检查 RRF 实现发现 `rerank` 后 `scores = [0.0] * len(docs)`（分数失去含义）

**根因**：reranker 重排序后，原始 RRF 分数被丢弃，新分数是 CrossEncoder 输出的绝对值（不在 [0, 1] 范围），evaluate 节点的阈值判断失效。

**解决方案**：在 `retrieve_node` 中，rerank 后 `scores = [0.0] * len(docs)`，evaluate 节点检测到 `valid_scores = [s for s in scores if s > 0]` 为空时，跳过规则层判断，只用 LLM 层语义诊断。

### 问题 2：Reranker 模型下载超时

**问题现象**：首次运行 reranker 模式时卡在"尝试加载模型"，10 分钟后超时。

**排查过程**：
1. 检查网络发现无法访问 HuggingFace Hub（需要代理）
2. 查看日志发现 `ConnectionError: HTTPSConnectionPool(host='huggingface.co'...)`

**根因**：BGE-Reranker-v2-m3 模型（560MB）需要从 HuggingFace Hub 下载，网络不可达时超时。

**解决方案**：优雅降级模式生效——`_ensure_loaded()` 捕获异常，`self._model = None`，`rerank()` 返回原顺序前 top_n。日志记录 `WARNING`，但不影响主流程。生产环境建议预下载模型或使用 HF 镜像。

---

## 与其他模块的交互

**输入接口**：
- `VectorStore` / `BM25Retriever`：从 Task 3 注入，作为 RRF 融合的两路检索器
- `settings.rrf_k` / `settings.reranker_model` / `settings.reranker_top_n`：从 Task 1 配置读取

**输出接口**：
- `HybridRetriever.search(query, k)` → `list[Document]`：供 Task 5 retrieve_node 使用
- `HybridRetriever.search_with_scores(query, k)` → `(list[Document], list[float])`：供 evaluate 节点
- `Reranker.rerank(query, documents, top_n)` → `list[Document]`：供 retrieve_node 重排序
- `Reranker.is_available()` → `bool`：供 retrieve_node 判断是否启用重排序

**衔接方式**：
- Task 5 的 `build_agent_graph` 接收 `hybrid_retriever` 和 `reranker` 参数（Task 5 阶段为 None）
- Task 7 的 `build_agent_pipeline` 构造 `HybridRetriever` 和 `Reranker` 实例注入
- Task 8 的 evaluator 根据 mode 决定是否注入 hybrid / reranker

---

## 本阶段收获总结

1. **RRF 是基于排名融合的鲁棒算法**：无需归一化两路不同尺度的分数，`k=60` 平滑排名差异
2. **CrossEncoder 精度高但速度慢，适合重排序**：候选集小（top-k）时延迟可接受，精度优于 BiEncoder
3. **优雅降级确保"锦上添花"功能失败不影响主流程**：Reranker 不可用时保持原顺序，HybridRetriever 仍可用
4. **小知识库上 RRF 可能引入噪声**：BM25 在小库上关键词匹配不够精准，稀释向量检索的高质量结果，Reranker 能修正
5. **对比实验需控制变量**：相同数据集、相同查询、相同 LLM，只变化检索策略，才能得出可靠结论
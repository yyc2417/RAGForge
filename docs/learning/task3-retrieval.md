> ⚠️ **历史快照（2026-06）**：本文为学习笔记，记录写作当时的机制与数字，部分已被 2026-09 修复取代（双预算终止、LANGSMITH_* 环境变量、45 题数据集、token 化分块等）。现行设计以 `docs/modules/` 与 `docs/adr/`（尤其 006）为准；逐项修复记录见本地 `docs/planning/fix-log-2026-09.md`（私有文档，不入库）。
>
> **本文具体过时点**：0.3 阈值现仅对 vector 策略生效（bm25/hybrid 仅空结果硬规则，见 ADR-006）；BM25 小写归一与零分过滤为 9 月新增。

# Task 3：检索层（向量 + BM25）

## 阶段概述

本阶段实现 RAG 系统的核心检索能力——向量检索和 BM25 关键词检索。目标是提供两种互补的检索方式，为后续 Task 6 的混合检索奠定基础。Task 3 是检索系统的基础，Task 4 的基础 RAG 和 Task 5 的 Agent 都依赖本阶段的检索器。

**前置依赖**：Task 1（配置）、Task 2（文档向量）  
**后续依赖**：Task 4（基础 RAG）、Task 5（Agent 检索节点）、Task 6（混合检索）

---

## 核心知识点

### 知识点 1：向量检索原理

向量检索的核心流程是：**文本 → Embedding 向量 → ChromaDB 存储 → 余弦相似度检索**。

```python
from langchain_chroma import Chroma

class VectorStore:
    def build_index(self, documents: list[Document]) -> None:
        """从文档列表构建向量索引（写入持久化目录）。"""
        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self._embedding.embedding_function,
            collection_name=settings.chroma_collection_name,
            persist_directory=self.persist_directory,
        )
```

**Embedding 过程**：每个 Document 的 `page_content`（文本）通过 `embedding_function` 转换为 384 维向量（all-MiniLM-L6-v2 模型）。ChromaDB 将向量和原文一起存储。

**余弦相似度检索**：查询时，查询文本也转换为向量，然后计算查询向量与所有文档向量的余弦相似度：

$$\text{similarity} = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \times |\vec{d}|}$$

余弦相似度范围 [-1, 1]，值越大表示语义越相似。ChromaDB 的 `similarity_search_with_relevance_scores` 返回归一化到 [0, 1] 的 relevance score：

```python
def search_with_scores(
    self, query: str, k: int | None = None
) -> tuple[list[Document], list[float]]:
    """带相关性分数的检索，供 evaluate 节点诊断使用。"""
    if self._store is None:
        raise RuntimeError("索引未构建，请先调用 build_index() 或 load_or_build()")
    k = k or settings.retrieval_k
    # similarity_search_with_relevance_scores 返回 [(Document, score), ...]
    results = self._store.similarity_search_with_relevance_scores(query, k=k)
    docs = [doc for doc, _ in results]
    scores = [float(score) for _, score in results]
    return docs, scores
```

**为什么用 relevance score 而非原始余弦相似度**：ChromaDB 的 relevance score 经过归一化处理，范围固定为 [0, 1]，便于跨查询比较和阈值设定（如 evaluate 节点用 0.3 作为 low_recall 阈值）。

### 知识点 2：BM25 算法原理

BM25（Best Matching 25）是 TF-IDF 的改进版，是信息检索领域最经典的关键词检索算法：

```python
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def build_index(self, documents: list[Document]) -> None:
        """构建 BM25 倒排索引。"""
        tokenized = [_tokenize(doc.page_content) for doc in documents]
        self._index = BM25Okapi(tokenized)
        self._docs = documents
```

**BM25 公式**：

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

其中：
- `f(t,d)`：词 `t` 在文档 `d` 中的词频（TF）
- `|d|`：文档长度，`avgdl`：平均文档长度
- `k_1`（通常 1.2-2.0）：词频饱和参数，控制 TF 的饱和速度
- `b`（通常 0.75）：文档长度归一化参数
- `IDF(t)`：逆文档频率，罕见词权重更高

**与 TF-IDF 的区别**：
1. **词频饱和**：BM25 的分母包含 `k_1` 项，当词频 `f(t,d)` 增大时，分数增长趋于饱和（避免某个词重复 100 次就无限提权）
2. **文档长度归一化**：`b` 参数控制文档长度对分数的影响，长文档不会因为词频绝对值高就自动获得高分

**`get_scores()` 返回无界正分数**：BM25 分数范围 [0, +∞)，取决于查询词在文档中的匹配程度。与向量检索的 [0, 1] 分数不可直接比较，这也是 Task 6 用 RRF（基于排名而非分数）融合的原因。

### 知识点 3：jieba 分词 vs 正则分词

中文分词是 BM25 的关键前置步骤，`_tokenize()` 函数提供两种策略：

```python
def _tokenize(text: str) -> list[str]:
    """对文本分词，支持中英文混合。
    
    优先尝试 jieba（更准确的中文分词），不可用时 fallback 到正则：
    中文按单字、英文按词。
    """
    try:
        import jieba  # type: ignore
        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        # fallback：中文按字，英文按词；忽略标点与空白
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)
```

**jieba 的精确模式**：`jieba.lcut(text)` 使用精确模式，基于词典和 HMM 模型切分。例如"机器学习是人工智能的分支"会被切分为 `["机器", "学习", "是", "人工智能", "的", "分支"]`。

**中文分词的挑战**：
1. **无空格分隔**：英文用空格天然分词，中文需要算法识别词边界
2. **歧义消解**："研究生命的起源"可以切分为"研究/生命/的/起源"或"研究生/命/的/起源"
3. **新词识别**：专业术语（如"过拟合"）可能不在词典中

**fallback 到正则的设计考量**：`[\u4e00-\u9fff]|[a-zA-Z0-9]+` 的含义：
- `[\u4e00-\u9fff]`：匹配单个中文汉字（CJK 统一汉字范围）
- `[a-zA-Z0-9]+`：匹配连续英文字母或数字

**为什么 fallback 用单字而非词典**：正则 fallback 是"保底方案"，用于未安装 jieba 的环境。单字切分虽然不够精准（"机器学习"被切为"机/器/学/习"），但能保证 BM25 仍能工作——关键词"机器学习"的查询会匹配包含"机"、"器"、"学"、"习"的文档，召回率虽低但不会完全失效。

### 知识点 4：统一接口设计

所有检索器（VectorStore / BM25Retriever / HybridRetriever）都实现相同的三个方法：

```python
# VectorStore
def build_index(self, documents: list[Document]) -> None: ...
def search(self, query: str, k: int | None = None) -> list[Document]: ...
def search_with_scores(self, query: str, k: int | None = None) -> tuple[list[Document], list[float]]: ...

# BM25Retriever
def build_index(self, documents: list[Document]) -> None: ...
def search(self, query: str, k: int | None = None) -> list[Document]: ...
def search_with_scores(self, query: str, k: int | None = None) -> tuple[list[Document], list[float]]: ...

# HybridRetriever（Task 6，无 build_index，由底层检索器各自构建）
def search(self, query: str, k: int | None = None) -> list[Document]: ...
def search_with_scores(self, query: str, k: int | None = None) -> tuple[list[Document], list[float]]: ...
```

**鸭子类型 vs Protocol**：Python 不强制要求实现接口（鸭子类型："如果它走起来像鸭子、叫起来像鸭子，那它就是鸭子"）。这里没有用 `typing.Protocol` 或 `abc.ABC` 定义正式接口，而是靠约定保持方法签名一致。

**为什么统一接口**：
1. **Agent 节点无需关心检索器类型**：`retrieve_node` 根据 `strategy` 选择检索器，但调用方式完全相同（`search_with_scores(query)`）
2. **Task 6 的 HybridRetriever 无缝集成**：因为它也实现相同接口，Agent 节点无需特殊处理
3. **便于测试和替换**：可以用 Mock 对象替换真实检索器进行单元测试

---

## 设计模式与架构决策

**统一接口模式**：所有检索器实现 `build_index` / `search` / `search_with_scores` 三个方法，Agent 节点可以透明地切换检索策略。

**持久化加载模式（VectorStore.load_or_build）**：ChromaDB 支持持久化存储，`load_or_build` 方法检查持久化目录是否存在且非空，存在则加载已有索引，不存在则重新构建。这避免了每次启动都重新嵌入所有文档（Embedding 是 CPU 密集型操作）。

**fallback 模式（分词器）**：jieba 优先，正则兜底。这种"优雅降级"设计确保系统在不同环境下都能工作（虽然降级后性能下降）。

---

## 关键代码解读

### vector_store.py：向量检索

```python
def build_index(self, documents: list[Document]) -> None:
    """从文档列表构建向量索引（写入持久化目录）。"""
    self._store = Chroma.from_documents(
        documents=documents,
        embedding=self._embedding.embedding_function,
        collection_name=settings.chroma_collection_name,
        persist_directory=self.persist_directory,
    )
    logger.info(f"[vector] 索引构建完成：{len(documents)} 个 chunk → {self.persist_directory}")
```

**设计意图**：`Chroma.from_documents` 是 LangChain 的高级 API，它自动完成"文档 → 向量 → 存储"的全流程。`embedding` 参数接受 `HuggingFaceEmbeddings` 实例，`collection_name` 指定 ChromaDB 中的集合名（类似数据库表名），`persist_directory` 指定持久化目录。

```python
@classmethod
def load_or_build(
    cls, documents: list[Document], embedding_service: EmbeddingService
) -> "VectorStore":
    """持久化加载：目录存在且非空则复用，否则重新构建。"""
    from pathlib import Path
    
    instance = cls(embedding_service)
    persist_dir = Path(instance.persist_directory)
    # ChromaDB 持久化目录非空才视为可复用
    if persist_dir.exists() and any(persist_dir.iterdir()):
        instance._store = Chroma(
            embedding_function=embedding_service.embedding_function,
            collection_name=settings.chroma_collection_name,
            persist_directory=str(persist_dir),
        )
        logger.info(f"[vector] 从持久化目录加载索引：{persist_dir}")
    else:
        instance.build_index(documents)
    return instance
```

**设计意图**：`@classmethod` 提供工厂方法，封装"加载或构建"的判断逻辑。`any(persist_dir.iterdir())` 检查目录是否非空（空目录视为不可用）。这种设计让调用方无需关心索引是否已存在，只需调用 `load_or_build` 即可。

### bm25.py：BM25 检索

```python
def build_index(self, documents: list[Document]) -> None:
    """构建 BM25 倒排索引。"""
    tokenized = [_tokenize(doc.page_content) for doc in documents]
    self._index = BM25Okapi(tokenized)
    self._docs = documents
    logger.info(f"[bm25] 索引构建完成：{len(documents)} 个 chunk")
```

**设计意图**：BM25 需要先对文档分词，`BM25Okapi` 接受分词后的列表（每个元素是 token 列表）。`self._docs` 保存原始 Document 列表，检索时按索引取回。BM25 不持久化（内存索引），每次启动重新构建（BM25 构建速度远快于 Embedding）。

```python
def search_with_scores(
    self, query: str, k: int | None = None
) -> tuple[list[Document], list[float]]:
    """带 BM25 分数的检索，供 evaluate 节点诊断使用。"""
    if self._index is None:
        raise RuntimeError("索引未构建，请先调用 build_index()")
    k = k or settings.retrieval_k
    scores = self._index.get_scores(_tokenize(query))
    top_indices = self._top_k_indices(scores, k)
    docs = [self._docs[i] for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]
    return docs, top_scores
```

**设计意图**：`get_scores()` 返回所有文档的 BM25 分数（数组），需要手动排序取 top-k。`_top_k_indices` 按分数降序取前 k 个索引，然后用索引从 `self._docs` 取回文档。这种"先排序索引再取文档"的方式避免了复制整个 Document 对象。

```python
@staticmethod
def _top_k_indices(scores: list[float], k: int) -> list[int]:
    """按分数降序取前 k 个索引（k 超过长度时自动截断）。"""
    n = len(scores)
    k = min(k, n)
    # 按分数降序排序，取前 k 个原始索引
    return sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
```

**设计意图**：`sorted(range(n), key=lambda i: scores[i], reverse=True)` 对索引排序（而非对分数排序），这样能保留原始索引用于取回文档。`min(k, n)` 防止 k 超过文档总数导致越界。

---

## 踩坑记录

### 问题 1：ChromaDB 持久化目录为空时报错

**问题现象**：`Chroma(persist_directory=...)` 加载时抛出 `ValueError: No collection found`。

**排查过程**：
1. 检查 `persist_directory` 发现目录存在但为空
2. 查看 `load_or_build` 逻辑，发现只检查 `persist_dir.exists()`，未检查是否非空

**根因**：ChromaDB 要求持久化目录包含 `chroma.sqlite3` 等文件，空目录会导致加载失败。

**解决方案**：在 `load_or_build` 中增加 `any(persist_dir.iterdir())` 检查，目录非空才视为可复用：

```python
if persist_dir.exists() and any(persist_dir.iterdir()):
    # 加载已有索引
else:
    # 重新构建
```

### 问题 2：BM25 分数为 0 的文档被误判为低质量

**问题现象**：evaluate 节点将 BM25 检索结果判为 `low_recall`，但实际检索到了相关文档。

**排查过程**：
1. 打印 `retrieval_scores` 发现所有分数都是 0.0
2. 检查 BM25 索引发现查询分词后与文档无匹配词
3. 检查文档发现是英文文档，查询是中文

**根因**：BM25 是关键词匹配，中英文查询无法匹配英文文档（除非文档包含中文术语）。

**解决方案**：这是 BM25 的固有限制，跨语言场景需要向量检索（语义匹配）。Task 6 的混合检索（RRF 融合向量+BM25）能缓解这个问题——向量检索捕获语义，BM25 补充精确关键词匹配。

---

## 与其他模块的交互

**输入接口**：
- `EmbeddingService.embedding_function`：从 Task 2 注入 Embedding 模型
- `list[Document]`：从 Task 2 的 chunker 输出的文档列表
- `settings.chroma_persist_dir` / `settings.retrieval_k`：从 Task 1 配置读取

**输出接口**：
- `VectorStore.search(query, k)` → `list[Document]`：供 Task 4 基础 RAG 使用
- `VectorStore.search_with_scores(query, k)` → `(list[Document], list[float])`：供 Task 5 evaluate 节点诊断
- `BM25Retriever.search_with_scores(query, k)` → `(list[Document], list[float])`：供 Task 5 retrieve 节点
- `VectorStore` / `BM25Retriever` 实例：供 Task 6 的 HybridRetriever 组合

**衔接方式**：
- Task 4 的 `build_rag_pipeline()` 构建 `VectorStore`，`answer()` 调用 `vector_store.search()`
- Task 5 的 `retrieve_node` 根据 `strategy` 选择 `vector_store` 或 `bm25`
- Task 5 的 `evaluate_node` 读取 `retrieval_scores` 判断检索质量
- Task 6 的 `HybridRetriever` 接收 `VectorStore` 和 `BM25Retriever` 实例，用 RRF 融合结果

---

## 本阶段收获总结

1. **向量检索和 BM25 是互补的检索方式**：向量检索捕获语义相似性（"汽车"匹配"轿车"），BM25 捕获精确关键词匹配（"GIL"只匹配包含"GIL"的文档）
2. **BM25 的分数无界，不适合与向量分数直接比较**：这是 Task 6 用 RRF（基于排名融合）而非分数加权的原因
3. **中文分词是 BM25 的关键挑战**：jieba 提供精准分词，正则 fallback 保证系统可用性
4. **统一接口设计让检索器可透明切换**：Agent 节点无需关心底层是向量还是 BM25，只需调用 `search_with_scores()`
5. **持久化加载避免重复 Embedding**：ChromaDB 持久化让重启后无需重新嵌入所有文档（Embedding 是 CPU 密集型操作）
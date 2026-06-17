# ADR-004：Reranker 采用懒加载单例 + 优雅降级

> **状态**：已采纳
> **日期**：2026-06-14
> **决策者**：宇诚

## 背景

RAGForge 使用 BGE-Reranker-v2-m3 模型对混合检索结果进行重排序，提升最终送入 LLM 的文档相关性。该模型约 560MB，首次使用时需要从 Hugging Face 下载。在无网络、无 GPU、下载超时或磁盘空间不足等情况下，模型可能不可用。系统不能因为 Reranker 不可用就整体不可用。

## 备选方案

### 方案 A：懒加载单例 + 失败降级 — 选定
- **概述**：`__new__` 实现单例模式；`_ensure_loaded()` 在首次调用时尝试加载模型，失败后 `_model=None`；`rerank()` 检测到模型不可用时直接返回原列表的前 top_n 条结果
- **优点**：
  - 模型不可用时系统仍可用，只是跳过重排序
  - 懒加载避免启动时阻塞（首次请求才下载）
  - 单例确保只尝试加载一次，避免重复下载
  - `is_available()` 接口让调用方可以主动判断
- **缺点**：
  - 首次请求有模型下载延迟
  - 降级后结果质量下降
- **决定**：选，可用性和健壮性优先

### 方案 B：启动时强制加载，失败则程序退出
- **概述**：`__init__` 中直接加载模型，加载失败抛异常终止程序
- **优点**：
  - 逻辑最简单，"要么完全可用，要么不启动"
- **缺点**：
  - 对生产环境不友好，网络波动就导致服务不可用
  - 开发/测试环境中模型不一定可用
- **决定**：不选，可用性太差

### 方案 C：始终跳过 Reranker
- **概述**：不使用重排序，直接返回 RRF 融合结果
- **优点**：
  - 零依赖，零延迟
- **缺点**：
  - 丧失重排序能力，评估数据显示 hybrid 召回率从 100% 降至 70%
- **决定**：不选，质量损失不可接受

### 方案 D：用轻量 API Reranker（如 Cohere Rerank API）
- **概述**：调用外部 API 代替本地模型
- **优点**：
  - 无本地模型管理问题
- **缺点**：
  - 增加外部依赖和网络延迟
  - 增加 API 调用成本
  - 与项目"低依赖"定位不符
- **决定**：不选，外部依赖过多

## 决定的理由

懒加载单例 + 优雅降级在可用性和质量之间取得了最佳平衡。核心实现：

```python
class Reranker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._model = None
                cls._instance._load_attempted = False
            return cls._instance

    def _ensure_loaded(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            self._model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        except Exception:
            self._model = None  # 降级

    def is_available(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    def rerank(self, query, docs, top_n=5):
        self._ensure_loaded()
        if self._model is None:
            return docs[:top_n]  # 降级：返回原序前 N 条
        # 正常重排序逻辑...
```

关键设计：`_load_attempted` 标记确保只尝试加载一次，避免每次请求都触发失败的下载。

## 风险与缓解

| 风险 | 可能性 | 缓解措施 |
|------|--------|----------|
| 降级后 HybridRetriever 返回纯 RRF 结果，质量下降 | 中 | 评估数据证明 Reranker 可将 hybrid 召回率从 70% 拉回 100% |
| 首次请求延迟（模型下载） | 中 | 文档提示用户首次使用需等待；可提前 `python -c "from src.retrieval.reranker import Reranker; Reranker()"` 预热 |
| 多线程首次同时调用 | 低 | `threading.Lock` 保护加载过程 |

## 参考资料

- [BGE-Reranker-v2-m3 模型页](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [sentence-transformers CrossEncoder 文档](https://www.sbert.net/docs/usage/cross_encoder.html)
- [FlagEmbedding Reranker 使用指南](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/reranker)

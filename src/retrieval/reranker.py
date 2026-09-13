"""重排序器：CrossEncoder bge-reranker-v2-m3（优雅降级单例）

设计要点（优雅降级）：
- BGE-Reranker-v2-m3 模型约 560MB，下载可能失败/超时（无网络/无代理）
- 采用懒加载单例：首次实例化时尝试加载，任何异常 → _model=None + warning
- is_available() 供调用方判断；不可用时 rerank() 直接返回原列表前 top_n
- 这样 HybridRetriever(RRF) 始终可用，Reranker 是锦上添花

CrossEncoder 评分：对 (query, doc) pair 输出相关性分数，按分数重排。
"""

import threading
import time

from langchain_core.documents import Document

from src.config import settings
from src.utils.logger import logger

# 加载失败后的冷却时间（秒）：冷却期内直接降级，避免每次请求都重复尝试下载
_LOAD_RETRY_COOLDOWN_SEC = 600


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
                instance._last_failure = 0.0
                instance._load_lock = threading.Lock()
                cls._instance = instance
            return cls._instance

    def _ensure_loaded(self) -> None:
        """懒加载模型（加锁的 check-then-act；失败后冷却期内不重试）。"""
        with self._load_lock:
            if self._model is not None:
                return
            if time.monotonic() - self._last_failure < _LOAD_RETRY_COOLDOWN_SEC:
                return
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"[reranker] 尝试加载模型: {self._model_name}")
                self._model = CrossEncoder(self._model_name)
                logger.info("[reranker] 模型加载成功，重排序可用")
            except Exception as e:  # noqa: BLE001 - 下载/导入失败均降级
                self._last_failure = time.monotonic()
                logger.warning(
                    f"[reranker] 模型加载失败，已降级为跳过重排序"
                    f"（{_LOAD_RETRY_COOLDOWN_SEC}s 后可重试）：{type(e).__name__}: {e}"
                )
                self._model = None

    def is_available(self) -> bool:
        """Reranker 是否可用（模型加载成功）。"""
        self._ensure_loaded()
        return self._model is not None

    def rerank(
        self, query: str, documents: list[Document], top_n: int | None = None
    ) -> list[Document]:
        """对候选文档做交叉重排序。

        Args:
            query: 用户查询文本
            documents: 候选文档列表
            top_n: 重排序后保留数量，默认读 settings.reranker_top_n

        Returns:
            按相关性重排后的文档列表；模型不可用时直接返回原顺序前 top_n
        """
        top_n = top_n if top_n is not None else settings.reranker_top_n

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
        except Exception as e:  # noqa: BLE001 - 推理失败也降级
            logger.warning(f"[reranker] 推理失败，降级为原顺序：{e}")
            return documents[:top_n]

        # 按分数降序排序取 top_n
        ranked = sorted(
            zip(documents, scores, strict=False), key=lambda x: x[1], reverse=True
        )[:top_n]
        result = [doc for doc, _ in ranked]
        logger.info(
            f"[reranker] 重排序完成：{len(documents)} → {len(result)} 篇"
            f"（top_score={float(ranked[0][1]):.4f}）"
        )
        return result

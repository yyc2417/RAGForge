"""BM25 检索器：rank-bm25 + 中文分词（Task 3 实现）

分词器优先使用 jieba（pyproject 已声明依赖），环境缺失时 fallback 到正则
按字符/英文单词切分。两条路径均做小写归一，保证英文查询大小写无关。

零分文档过滤：查询分词与语料无词汇交集时，BM25 全 0 分文档不进入结果，
避免完全无关文档按默认顺序挤占 top-k 并在 RRF 融合中重复计票。
"""

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import settings
from src.utils.logger import logger


def _tokenize(text: str) -> list[str]:
    """对文本分词（小写归一），支持中英文混合。

    优先尝试 jieba（更准确的中文分词），不可用时 fallback 到正则：
    中文按单字、英文按连续字母分组。

    Args:
        text: 待分词文本

    Returns:
        token 列表（过滤掉空白 token）
    """
    text = text.lower()  # 英文侧大小写归一；中文不受影响
    try:
        import jieba  # type: ignore

        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        # fallback：中文按字，英文按词；忽略标点与空白
        return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)


class BM25Retriever:
    """BM25 关键词检索器。

    接口与 VectorStore 完全一致：
    ``build_index`` / ``search`` / ``search_with_scores``。
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._docs: list[Document] = []

    def build_index(self, documents: list[Document]) -> None:
        """构建 BM25 倒排索引。

        Args:
            documents: 已切分的 Document 列表
        """
        tokenized = [_tokenize(doc.page_content) for doc in documents]
        self._index = BM25Okapi(tokenized)
        self._docs = documents
        logger.info(f"[bm25] 索引构建完成：{len(documents)} 个 chunk")

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """BM25 关键词检索，返回 Top-k 文档（零分文档已过滤）。

        Args:
            query: 用户查询文本
            k: 返回数量，默认读 settings.retrieval_k

        Returns:
            相关性从高到低的 Document 列表；无词汇命中时返回空列表
        """
        docs, _ = self.search_with_scores(query, k)
        return docs

    def search_with_scores(
        self, query: str, k: int | None = None
    ) -> tuple[list[Document], list[float]]:
        """带 BM25 分数的检索，供 evaluate 节点诊断使用。

        Args:
            query: 用户查询文本
            k: 返回数量，默认读 settings.retrieval_k

        Returns:
            (documents, scores) 二元组，分数越高越相关；零分文档不返回
        """
        if self._index is None:
            raise RuntimeError("索引未构建，请先调用 build_index()")
        k = k if k is not None else settings.retrieval_k
        scores = self._index.get_scores(_tokenize(query))
        # 过滤零分文档：与查询无词汇交集，不应挤占 top-k
        candidates = [i for i, s in enumerate(scores) if s > 0]
        top_indices = self._top_k_indices(scores, k, candidates)
        docs = [self._docs[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]
        return docs, top_scores

    @staticmethod
    def _top_k_indices(
        scores: list[float], k: int, candidates: list[int] | None = None
    ) -> list[int]:
        """按分数降序取前 k 个索引（默认全量参与，可传候选子集；不足 k 自动截断）。"""
        pool = range(len(scores)) if candidates is None else candidates
        return sorted(pool, key=lambda i: scores[i], reverse=True)[:k]

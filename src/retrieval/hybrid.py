"""混合检索器：RRF（Reciprocal Rank Fusion）融合向量 + BM25

RRF 公式：
    score(d) = Σ_i  1 / (rrf_k + rank_i(d))

其中 rank_i(d) 是文档 d 在第 i 路检索结果中的排名（1-based）。
rrf_k 为平滑常数（标准值 60），避免 top-1 过度主导。
两路各取 top-2k 候选，融合后按 RRF 分数取 top-k。

优点：无需归一化两路不同尺度的分数（向量 relevance ∈ [0,1]，BM25 无界），
直接用排名融合，鲁棒且无需调参。
"""

from langchain_core.documents import Document

from src.config import settings
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import VectorStore
from src.utils.logger import logger


class HybridRetriever:
    """RRF 混合检索器，融合向量检索与 BM25 关键词检索。

    接口与 VectorStore / BM25Retriever 一致：
    ``search`` / ``search_with_scores``（build_index 由底层检索器各自完成）。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25: BM25Retriever,
        rrf_k: int | None = None,
    ) -> None:
        """初始化混合检索器。

        Args:
            vector_store: 已构建索引的向量检索器
            bm25: 已构建索引的 BM25 检索器
            rrf_k: RRF 平滑常数，默认读 settings.rrf_k
        """
        self._vector = vector_store
        self._bm25 = bm25
        self._rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """RRF 融合检索，返回 Top-k 文档。"""
        docs, _ = self.search_with_scores(query, k)
        return docs

    def search_with_scores(
        self, query: str, k: int | None = None
    ) -> tuple[list[Document], list[float]]:
        """RRF 融合检索，返回 Top-k 文档与融合分数。

        Args:
            query: 用户查询文本
            k: 最终返回数量，默认读 settings.retrieval_k

        Returns:
            (documents, scores) 二元组，分数为 RRF 融合分（越高越相关）
        """
        k = k if k is not None else settings.retrieval_k
        # 每路取 top-2k 候选，扩大召回池
        candidate_k = max(k * 2, k + 1)

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

        logger.info(
            f"[hybrid] RRF 融合：vector={len(v_docs)}, bm25={len(b_docs)} "
            f"→ top-{k}（rrf_k={self._rrf_k}）"
        )
        return docs, scores

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """文档去重键：内容前 80 字符（避免同一文档被两路重复计入）。"""
        return doc.page_content[:80]

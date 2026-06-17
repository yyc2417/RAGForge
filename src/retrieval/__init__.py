"""检索层 - 向量检索、BM25、混合检索、重排序

提供与统一接口约定一致的检索器：
- VectorStore：ChromaDB 向量检索（Task 3）
- BM25Retriever：rank-bm25 关键词检索（Task 3）
- HybridRetriever：RRF 融合检索（Task 6）
- Reranker：CrossEncoder 重排序（Task 6）

所有检索器实现统一签名：
    build_index(documents) -> None
    search(query, k=None) -> list[Document]
    search_with_scores(query, k=None) -> tuple[list[Document], list[float]]
"""

from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

__all__ = ["VectorStore", "BM25Retriever", "HybridRetriever", "Reranker"]

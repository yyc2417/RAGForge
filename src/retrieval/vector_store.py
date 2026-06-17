"""向量检索器：封装 ChromaDB 相似度检索（Task 3 实现）"""

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import PROJECT_ROOT, settings
from src.ingestion.embedder import EmbeddingService
from src.utils.logger import logger


class VectorStore:
    """ChromaDB 向量检索封装。

    提供与其他检索器（BM25Retriever / HybridRetriever）完全一致的接口：
    ``build_index`` / ``search`` / ``search_with_scores``。
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """初始化检索器，注入 Embedding 服务。

        Args:
            embedding_service: 已加载好模型的 Embedding 单例
        """
        self._embedding = embedding_service
        self._store: Chroma | None = None

    @property
    def persist_directory(self) -> str:
        """返回 ChromaDB 持久化目录的绝对路径。"""
        return str(PROJECT_ROOT / settings.chroma_persist_dir)

    def build_index(self, documents: list[Document]) -> None:
        """从文档列表构建向量索引（写入持久化目录）。

        Args:
            documents: 已切分的 Document 列表
        """
        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self._embedding.embedding_function,
            collection_name=settings.chroma_collection_name,
            persist_directory=self.persist_directory,
        )
        logger.info(f"[vector] 索引构建完成：{len(documents)} 个 chunk → {self.persist_directory}")

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """相似度检索，返回 Top-k 文档。

        Args:
            query: 用户查询文本
            k: 返回数量，默认读 settings.retrieval_k

        Returns:
            相关性从高到低的 Document 列表
        """
        if self._store is None:
            raise RuntimeError("索引未构建，请先调用 build_index() 或 load_or_build()")
        k = k or settings.retrieval_k
        return self._store.similarity_search(query, k=k)

    def search_with_scores(
        self, query: str, k: int | None = None
    ) -> tuple[list[Document], list[float]]:
        """带相关性分数的检索，供 evaluate 节点诊断使用。

        Args:
            query: 用户查询文本
            k: 返回数量，默认读 settings.retrieval_k

        Returns:
            (documents, scores) 二元组，分数越高越相关（relevance score）
        """
        if self._store is None:
            raise RuntimeError("索引未构建，请先调用 build_index() 或 load_or_build()")
        k = k or settings.retrieval_k
        # similarity_search_with_relevance_scores 返回 [(Document, score), ...]
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        docs = [doc for doc, _ in results]
        scores = [float(score) for _, score in results]
        return docs, scores

    @classmethod
    def load_or_build(
        cls, documents: list[Document], embedding_service: EmbeddingService
    ) -> "VectorStore":
        """持久化加载：目录存在且非空则复用，否则重新构建。

        Args:
            documents: 用于构建索引的 Document 列表（仅在重建时使用）
            embedding_service: Embedding 服务实例

        Returns:
            就绪的 VectorStore 实例
        """
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

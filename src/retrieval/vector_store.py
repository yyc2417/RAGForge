"""向量检索器：封装 ChromaDB 相似度检索（Task 3 实现）

索引一致性保障（详见 ADR-006）：
- build_index 写入前删除同名 collection（Chroma.from_documents 对已存在
  collection 只追加不去重，会造成重复写入与 RRF 重复计票）
- 确定性 ID（source:chunk_index 的 uuid5），同语料重复构建天然幂等
- collection metadata 记录 embedding_model 指纹，load_or_build 复用前校验
  文档数与指纹，不匹配即重建，杜绝陈旧索引静默复用
"""

import uuid
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

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
        """从文档列表构建向量索引（写入持久化目录，先删旧再建新）。

        Args:
            documents: 已切分的 Document 列表（metadata 需含 source / chunk_index）
        """
        # 删除旧 collection：from_documents 对已存在 collection 只追加，必须先清
        client = chromadb.PersistentClient(path=self.persist_directory)
        existing = {c.name for c in client.list_collections()}
        if settings.chroma_collection_name in existing:
            client.delete_collection(settings.chroma_collection_name)
            logger.warning(
                f"[vector] 已删除旧 collection：{settings.chroma_collection_name}"
            )

        # 确定性 ID：同一语料重复构建时 upsert 幂等，不会产生重复文档
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{doc.metadata.get('source', 'unknown')}:"
                    f"{doc.metadata.get('chunk_index', i)}",
                )
            )
            for i, doc in enumerate(documents)
        ]
        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self._embedding.embedding_function,
            collection_name=settings.chroma_collection_name,
            persist_directory=self.persist_directory,
            ids=ids,
            collection_metadata={
                # 显式声明余弦距离：relevance score ∈ [0,1]，阈值语义稳定
                # （Chroma 默认 l2，relevance score 可能为负，无法与 0.3 阈值比较）
                "hnsw:space": "cosine",
                # 指纹：load_or_build 复用前校验，embedding 模型变更即重建
                "embedding_model": settings.embedding_model,
            },
        )
        logger.info(f"[vector] 索引构建完成：{len(documents)} 个 chunk → {self.persist_directory}")

    def count(self) -> int:
        """返回当前 collection 的文档数量（索引未构建/不存在时返回 0）。"""
        if self._store is None:
            return 0
        client = chromadb.PersistentClient(path=self.persist_directory)
        existing = {c.name for c in client.list_collections()}
        if settings.chroma_collection_name not in existing:
            return 0
        return client.get_collection(settings.chroma_collection_name).count()

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
        k = k if k is not None else settings.retrieval_k
        return self._store.similarity_search(query, k=k)

    def search_with_scores(
        self, query: str, k: int | None = None
    ) -> tuple[list[Document], list[float]]:
        """带相关性分数的检索，供 evaluate 节点诊断使用。

        Args:
            query: 用户查询文本
            k: 返回数量，默认读 settings.retrieval_k

        Returns:
            (documents, scores) 二元组，分数越高越相关（cosine relevance score ∈ [0,1]）
        """
        if self._store is None:
            raise RuntimeError("索引未构建，请先调用 build_index() 或 load_or_build()")
        k = k if k is not None else settings.retrieval_k
        # similarity_search_with_relevance_scores 返回 [(Document, score), ...]
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        docs = [doc for doc, _ in results]
        scores = [float(score) for _, score in results]
        return docs, scores

    @classmethod
    def load_or_build(
        cls,
        documents: list[Document],
        embedding_service: EmbeddingService,
        force_rebuild: bool = False,
    ) -> "VectorStore":
        """持久化加载：复用条件全部满足才加载，否则删除重建。

        复用条件（防止陈旧索引静默复用）：
        - 目录存在且非空
        - collection 文档数 == len(documents)
        - collection 记录的 embedding_model 与当前配置一致

        Args:
            documents: 用于构建索引的 Document 列表
            embedding_service: Embedding 服务实例
            force_rebuild: 为 True 时跳过复用检查，强制重建

        Returns:
            就绪的 VectorStore 实例
        """
        instance = cls(embedding_service)
        persist_dir = Path(instance.persist_directory)

        reusable = False
        if not force_rebuild and persist_dir.exists() and any(persist_dir.iterdir()):
            try:
                client = chromadb.PersistentClient(path=str(persist_dir))
                existing = {c.name for c in client.list_collections()}
                if settings.chroma_collection_name in existing:
                    col = client.get_collection(settings.chroma_collection_name)
                    meta = col.metadata or {}
                    reusable = (
                        col.count() == len(documents)
                        and meta.get("embedding_model") == settings.embedding_model
                    )
                    if not reusable:
                        logger.warning(
                            f"[vector] 索引指纹不匹配"
                            f"（count={col.count()}/{len(documents)}, "
                            f"model={meta.get('embedding_model')}/"
                            f"{settings.embedding_model}），重建索引"
                        )
                else:
                    logger.warning("[vector] collection 不存在，重建索引")
            except Exception as e:  # noqa: BLE001 - 校验异常一律走重建，保证可用
                logger.warning(f"[vector] 索引校验失败（{e}），重建索引")

        if reusable:
            instance._store = Chroma(
                embedding_function=embedding_service.embedding_function,
                collection_name=settings.chroma_collection_name,
                persist_directory=str(persist_dir),
            )
            logger.info(f"[vector] 从持久化目录加载索引：{persist_dir}")
        else:
            instance.build_index(documents)
        return instance

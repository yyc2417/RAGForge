"""Embedding 生成与索引构建（线程安全单例）"""

import threading

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.utils.logger import logger


class EmbeddingService:
    """Embedding 服务，封装 HuggingFaceEmbeddings。

    线程安全单例模式：模型加载成本高（首次约 1-2 秒），
    全局共享同一实例，避免重复加载。
    model_name 与已加载实例不一致时直接报错（而非静默返回旧模型），
    避免配置错误被吞掉后产出与配置不符的向量。
    """

    _instance: "EmbeddingService | None" = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str | None = None):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._model_name = model_name or settings.embedding_model
                logger.info(f"[embedder] 加载模型: {instance._model_name}")
                instance._model = HuggingFaceEmbeddings(
                    model_name=instance._model_name,
                    model_kwargs={"device": settings.embedding_device},
                    encode_kwargs={"batch_size": settings.embedding_batch_size},
                )
                logger.info("[embedder] 模型加载完成")
                cls._instance = instance
            elif model_name and model_name != cls._instance._model_name:
                raise ValueError(
                    f"EmbeddingService 单例已加载模型 {cls._instance._model_name}，"
                    f"拒绝再用 {model_name} 初始化（如需更换模型请重建索引并重启进程）"
                )
        return cls._instance

    def __init__(self, model_name: str | None = None):
        # 初始化已在 __new__ 的锁内完成，此处无需操作
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量。"""
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """生成单条查询向量。"""
        return self._model.embed_query(text)

    @property
    def embedding_function(self) -> HuggingFaceEmbeddings:
        """暴露底层 HuggingFaceEmbeddings 实例，供 ChromaDB 直接使用。"""
        return self._model

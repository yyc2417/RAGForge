"""文本切分器：递归语义切分，保留元数据传递"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.utils.logger import logger


class TextChunker:
    """递归语义文本切分器。

    封装 RecursiveCharacterTextSplitter，切分后为每个 chunk
    的 metadata 追加 chunk_index，保留原始 source 等元数据。
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """初始化切分器，参数为空时从 settings 读取默认值。"""
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
            length_function=len,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """切分文档列表，保留原始 metadata 并追加 chunk_index。

        Args:
            documents: 待切分的 Document 列表

        Returns:
            切分后的 Document 列表，每个 chunk 的 metadata 包含：
            - source: 原始文件路径（来自 parser）
            - file_type: 文件类型（来自 parser）
            - chunk_index: 该文档内的 chunk 序号（0-based）
        """
        chunks = self._splitter.split_documents(documents)

        # 为每个 chunk 追加 chunk_index（按 source 分组编号）
        source_counter: dict[str, int] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            idx = source_counter.get(source, 0)
            chunk.metadata["chunk_index"] = idx
            source_counter[source] = idx + 1

        logger.info(
            f"[chunker] {len(documents)} 个文档 → {len(chunks)} 个 chunks"
            f"（chunk_size={self._chunk_size}, overlap={self._chunk_overlap}）"
        )
        return chunks

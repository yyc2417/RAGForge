"""文本切分器：按 token 预算递归切分 + Markdown 标题上下文注入

设计要点（详见 ADR-006 与修复日志）：
- chunk_size 以 token 计（cl100k_base 估算）：embedding 模型窗口有限
  （all-MiniLM-L6-v2 为 256 token），按字符切 500 字符的中文长块会被
  编码器静默截断，约一半内容从未进入向量
- 先按 Markdown 标题切节，chunk 携带各级标题（metadata.h1/h2/h3 并拼入
  正文前缀），FAQ/教程类文档的短小 chunk 不再丢失标题语义
"""

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import settings
from src.utils.logger import logger

# 提取进 metadata 的标题层级（与 MarkdownHeaderTextSplitter 的键对应）
_HEADER_KEYS = ("h1", "h2", "h3")


class TextChunker:
    """按 token 预算切分 + 标题上下文注入的文档切分器。"""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """初始化切分器，参数为空时从 settings 读取默认值（单位：token）。"""
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        # from_tiktoken_encoder：length_function 按 token 数计算
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True,  # 标题行移出正文，由统一的 header_path 前缀携带
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """切分文档列表：按标题切节 → 按 token 切块 → 注入标题前缀与 metadata。

        Args:
            documents: 待切分的 Document 列表

        Returns:
            切分后的 Document 列表，每个 chunk 的 metadata 包含：
            - source: 原始文件路径（来自 parser）
            - file_type / page_number: 来自 parser
            - h1/h2/h3: 所属各级标题（如有）
            - chunk_index: 该文档内的 chunk 序号（0-based）
        """
        chunks: list[Document] = []
        for doc in documents:
            sections = self._header_splitter.split_text(doc.page_content)
            for section in sections:
                headers = {k: section.metadata[k] for k in _HEADER_KEYS if k in section.metadata}
                header_path = " > ".join(headers[k] for k in _HEADER_KEYS if k in headers)
                for piece in self._splitter.split_text(section.page_content):
                    # 标题路径拼入正文前缀：同时进入向量与 BM25，检索可命中标题词
                    content = f"{header_path}\n\n{piece}" if header_path else piece
                    chunks.append(
                        Document(page_content=content, metadata={**doc.metadata, **headers})
                    )

        # 为每个 chunk 追加 chunk_index（按 source 分组编号，供确定性 ID 使用）
        source_counter: dict[str, int] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            idx = source_counter.get(source, 0)
            chunk.metadata["chunk_index"] = idx
            source_counter[source] = idx + 1

        logger.info(
            f"[chunker] {len(documents)} 个文档 → {len(chunks)} 个 chunks"
            f"（chunk_size={self._chunk_size} tokens, overlap={self._chunk_overlap}）"
        )
        return chunks

"""文档处理层：解析、切分、向量化

提供三个核心类：
- DocumentParser：多格式文档解析（Markdown + PDF）
- TextChunker：递归语义文本切分
- EmbeddingService：Embedding 生成（线程安全单例）
"""

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import EmbeddingService

__all__ = ["DocumentParser", "TextChunker", "EmbeddingService"]

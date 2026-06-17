"""Task 2 验证：文档处理层（Ingestion）

验证项目：
1. DocumentParser 解析 data/sample/python_basics.md 返回非空列表
2. TextChunker 切分后 chunk 数量 ≥ 5（量化门槛）
3. 每个 chunk 的 page_content 长度 ≤ chunk_size + 合理余量
4. 切分后 metadata 包含 source 和 chunk_index
5. EmbeddingService.embed_query("test") 返回 384 维向量
6. 三步串联无报错：parse → split → embed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")

from src.config import PROJECT_ROOT, settings
from src.ingestion import DocumentParser, TextChunker, EmbeddingService

DATA_DIR = PROJECT_ROOT / settings.data_dir


def test_parser():
    print("=" * 50)
    print("测试 1：DocumentParser 解析 Markdown")
    print("=" * 50)
    parser = DocumentParser()

    # 解析单文件
    docs = parser.parse(DATA_DIR / "python_basics.md")
    assert len(docs) > 0, "解析结果不应为空"
    assert docs[0].metadata.get("source"), "metadata 应包含 source"
    assert docs[0].metadata.get("file_type") == "markdown", "file_type 应为 markdown"
    print(f"[OK] python_basics.md -> {len(docs)} docs")

    # 解析目录
    all_docs = parser.parse_directory(DATA_DIR)
    assert len(all_docs) >= 2, f"目录应至少含 2 个文件，实际 {len(all_docs)} 段"
    print(f"[OK] data/sample/ dir -> {len(all_docs)} docs")
    return all_docs


def test_chunker(docs):
    print("\n" + "=" * 50)
    print("测试 2：TextChunker 切分")
    print("=" * 50)
    chunker = TextChunker()
    chunks = chunker.split(docs)

    # 量化门槛
    assert len(chunks) >= 5, f"chunk 数量应 ≥ 5，实际 {len(chunks)}"
    print(f"[OK] {len(docs)} docs -> {len(chunks)} chunks (>= 5)")

    # 长度检查（允许 10% 余量，因 separators 可能在边界处略超）
    max_len = max(len(c.page_content) for c in chunks)
    expected_max = settings.chunk_size * 1.1
    assert max_len <= expected_max, f"最大 chunk 长度 {max_len} 超过阈值 {expected_max:.0f}"
    print(f"[OK] max chunk length: {max_len} (threshold: {expected_max:.0f})")

    # metadata 检查
    for chunk in chunks:
        assert "source" in chunk.metadata, "metadata 缺少 source"
        assert "chunk_index" in chunk.metadata, "metadata 缺少 chunk_index"
    print(f"[OK] all chunks have source + chunk_index in metadata")

    # 展示 chunk_index 分布
    from collections import Counter
    source_counts = Counter(c.metadata["source"] for c in chunks)
    for source, count in source_counts.items():
        print(f"   {Path(source).name}: {count} chunks")

    return chunks


def test_embedder(chunks):
    print("\n" + "=" * 50)
    print("测试 3：EmbeddingService")
    print("=" * 50)
    embedder = EmbeddingService()

    # 单条查询向量
    vec = embedder.embed_query("test")
    assert len(vec) == 384, f"all-MiniLM-L6-v2 输出应为 384 维，实际 {len(vec)}"
    print(f"[OK] embed_query('test') -> {len(vec)} dims")

    # 批量文档向量（取前 3 个 chunk 测试）
    texts = [c.page_content for c in chunks[:3]]
    vecs = embedder.embed_documents(texts)
    assert len(vecs) == 3, f"应返回 3 个向量，实际 {len(vecs)}"
    assert all(len(v) == 384 for v in vecs), "所有向量应为 384 维"
    print(f"[OK] embed_documents(3 chunks) -> {len(vecs)} x 384 dims")


def test_pipeline():
    print("\n" + "=" * 50)
    print("测试 4：三步串联 parse → split → embed")
    print("=" * 50)
    parser = DocumentParser()
    chunker = TextChunker()
    embedder = EmbeddingService()

    docs = parser.parse_directory(DATA_DIR)
    chunks = chunker.split(docs)
    sample_vec = embedder.embed_query(chunks[0].page_content)

    assert len(sample_vec) == 384
    print(f"[OK] pipeline: {len(docs)} docs -> {len(chunks)} chunks -> 384 dims")


if __name__ == "__main__":
    print("RAGForge Task 2 验证：文档处理层（Ingestion）\n")
    docs = test_parser()
    chunks = test_chunker(docs)
    test_embedder(chunks)
    test_pipeline()
    print("\n" + "=" * 50)
    print("[PASS] Task 2 All Tests Passed!")
    print("=" * 50)

"""文档处理层单元测试（Ingestion）——离线，无网络依赖

覆盖：
1. DocumentParser：单文件 / 目录解析，metadata 完整
2. TextChunker：token 预算（cl100k 计量，非字符）、标题上下文注入、
   metadata 完整性、切分确定性（确定性 ID 的前提）
3. EmbeddingService：向量维度、批量一致性、model_name 不一致报错
"""

import tiktoken

from src.config import settings
from src.ingestion import DocumentParser, EmbeddingService, TextChunker

_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


# ── DocumentParser ────────────────────────────────────────────────


def test_parser_single_file(sample_docs):
    parser = DocumentParser()
    docs = parser.parse(sample_docs[0].metadata["source"])
    assert len(docs) > 0, "解析结果不应为空"
    assert docs[0].metadata.get("source"), "metadata 应包含 source"
    assert docs[0].metadata.get("file_type") == "markdown", "file_type 应为 markdown"


def test_parser_directory(sample_docs):
    assert len(sample_docs) >= 2, f"目录应至少含 2 个文件，实际 {len(sample_docs)} 段"


def test_parser_rejects_unsupported(tmp_path):
    bad = tmp_path / "a.txt"
    bad.write_text("hello", encoding="utf-8")
    try:
        DocumentParser().parse(bad)
        raise AssertionError(".txt 应被拒绝")
    except ValueError as e:
        assert "不支持的文件格式" in str(e)


# ── TextChunker ───────────────────────────────────────────────────


def test_chunker_token_budget(chunks):
    """每个 chunk 的 token 数不超过 chunk_size + 标题前缀余量。

    标题路径前缀在切分后拼接，额外占用少量 token，余量取 30。
    （修复前按字符切 500 字符，超出 embedding 模型 256 token 窗口被静默截断。）
    """
    for chunk in chunks:
        tokens = _token_len(chunk.page_content)
        assert tokens <= settings.chunk_size + 30, (
            f"chunk token 数 {tokens} 超过预算 {settings.chunk_size}+30: "
            f"{chunk.page_content[:50]!r}"
        )


def test_chunker_metadata(chunks):

    counters: dict[str, set[int]] = {}
    for chunk in chunks:
        assert "source" in chunk.metadata, "metadata 缺少 source"
        assert "chunk_index" in chunk.metadata, "metadata 缺少 chunk_index"
        counters.setdefault(chunk.metadata["source"], set()).add(chunk.metadata["chunk_index"])
    # chunk_index 在同一 source 内连续编号（确定性 ID 依赖此约定）
    for source, idx_set in counters.items():
        assert idx_set == set(range(len(idx_set))), f"{source} 的 chunk_index 不连续"
    assert sum(len(v) for v in counters.values()) == len(chunks)


def test_chunker_header_context(chunks):
    """Markdown 语料的 chunk 应携带标题 metadata 并把标题路径拼入正文前缀。"""
    with_headers = [c for c in chunks if "h1" in c.metadata or "h2" in c.metadata]
    assert with_headers, "没有 chunk 携带标题 metadata（标题上下文注入失效）"
    for chunk in with_headers[:5]:
        prefix = chunk.page_content.split("\n\n", 1)[0]
        assert " > " in prefix, f"正文前缀应含标题路径: {prefix!r}"


def test_chunker_deterministic(sample_docs):
    """两次切分结果完全一致（vector_store 确定性 ID 依赖此性质）。"""
    c1 = TextChunker().split(sample_docs)
    c2 = TextChunker().split(sample_docs)
    assert len(c1) == len(c2)
    assert [c.page_content for c in c1] == [c.page_content for c in c2]
    assert [c.metadata for c in c1] == [c.metadata for c in c2]


# ── EmbeddingService ──────────────────────────────────────────────


def test_embedder_query_dims(embedder):
    vec = embedder.embed_query("test")
    assert len(vec) == 384, f"all-MiniLM-L6-v2 输出应为 384 维，实际 {len(vec)}"


def test_embedder_batch_consistency(embedder, chunks):
    texts = [c.page_content for c in chunks[:3]]
    vecs = embedder.embed_documents(texts)
    assert len(vecs) == 3
    assert all(len(v) == 384 for v in vecs), "所有向量应为 384 维"


def test_embedder_model_mismatch_rejected(embedder):
    """单例已加载模型后，用不同 model_name 初始化应报错而非静默返回旧模型。"""
    try:
        EmbeddingService("some-other-model")
        raise AssertionError("model_name 不一致应被拒绝")
    except ValueError as e:
        assert "拒绝再用" in str(e)

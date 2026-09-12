"""检索层单元测试（向量 + BM25 + 混合 + 重排序）——离线，本地模型

覆盖：
1. 向量/BM25 召回与分数格式（cosine relevance ∈ [0,1]）
2. BM25 大小写归一与零分文档过滤
3. 检索延迟门槛
4. 混合 RRF 召回 ≥ 纯向量；Reranker 优雅降级
5. 索引重建幂等 + load_or_build 指纹校验（回归：重复写入缺陷）
"""

import time

from src.retrieval import BM25Retriever, HybridRetriever, Reranker, VectorStore
from src.config import settings

SOURCE_NAME = lambda d: d.metadata.get("source", "").split("/")[-1].split("\\")[-1]  # noqa: E731

# 对比用例：(查询, 期望命中的源文件)
CASES = [
    ("Python 装饰器", "python_basics.md"),
    ("过拟合", "machine_learning_faq.md"),
    ("列表推导式", "python_basics.md"),
    ("梯度下降", "machine_learning_faq.md"),
    ("Transformer 自注意力", "machine_learning_faq.md"),
]


# ── 召回与分数 ────────────────────────────────────────────────────


def test_vector_recall(vector_store):
    docs = vector_store.search("Python 装饰器", k=3)
    assert len(docs) == 3, f"应返回 3 条，实际 {len(docs)}"
    sources = {SOURCE_NAME(d) for d in docs}
    assert "python_basics.md" in sources, f"Top-3 应命中 python_basics.md，实际 {sources}"


def test_vector_scores_in_unit_range(vector_store):
    """cosine 距离下的 relevance score 应在 [0,1]（评估阈值 0.3 的前提）。"""
    _, scores = vector_store.search_with_scores("过拟合", k=3)
    assert all(0.0 <= s <= 1.0 for s in scores), f"分数超出 [0,1]: {scores}"


def test_bm25_recall(bm25):
    docs = bm25.search("过拟合", k=3)
    assert docs, "BM25 应返回非空结果"
    sources = {SOURCE_NAME(d) for d in docs}
    assert "machine_learning_faq.md" in sources, f"Top-3 应命中 machine_learning_faq.md，实际 {sources}"


def test_bm25_case_insensitive(bm25):
    """大小写归一：'PYTHON' 与 'python' 召回一致。"""
    assert bm25.search("PYTHON 装饰器", k=3), "大写查询应命中"
    assert bm25.search("python 装饰器", k=3), "小写查询应命中"


def test_bm25_zero_score_filtered(bm25):
    """无词汇交集的查询应返回空（零分文档不再挤占 top-k）。"""
    docs, scores = bm25.search_with_scores("zzqqxx yywwuu vvkkpp", k=3)
    assert docs == [] and scores == [], f"零分文档未过滤: {docs}"


def test_score_format(vector_store, bm25):
    v_docs, v_scores = vector_store.search_with_scores("Python 装饰器", k=3)
    b_docs, b_scores = bm25.search_with_scores("过拟合", k=3)
    for docs, scores in ((v_docs, v_scores), (b_docs, b_scores)):
        assert isinstance(docs, list) and isinstance(scores, list)
        assert len(docs) == len(scores), "docs/scores 长度应一致"
        assert all(isinstance(s, float) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in v_scores), "向量分数应在 [0,1]"


def test_retrieval_latency(vector_store, bm25):
    threshold_ms = 500
    t0 = time.perf_counter()
    vector_store.search("列表推导式", k=3)
    v_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    bm25.search("梯度下降", k=3)
    b_ms = (time.perf_counter() - t0) * 1000
    assert v_ms < threshold_ms, f"向量检索延迟 {v_ms:.2f}ms 超过 {threshold_ms}ms"
    assert b_ms < threshold_ms, f"BM25 检索延迟 {b_ms:.2f}ms 超过 {threshold_ms}ms"


# ── 混合检索与重排序 ──────────────────────────────────────────────


def test_hybrid_recall_ge_vector(vector_store, bm25):
    """RRF 融合召回应不低于纯向量（核心价值验证）。"""
    hybrid = HybridRetriever(vector_store, bm25)
    v_hits = sum(
        1 for q, expected in CASES if expected in {SOURCE_NAME(d) for d in vector_store.search(q, k=5)}
    )
    h_hits = sum(
        1 for q, expected in CASES if expected in {SOURCE_NAME(d) for d in hybrid.search(q, k=5)}
    )
    assert h_hits >= v_hits, f"混合召回 {h_hits} 应 ≥ 纯向量 {v_hits}"


def test_hybrid_rrf_score_ceiling(vector_store, bm25):
    """RRF 融合分上限约 2/(rrf_k+1)≈0.033——evaluate 规则层不得对其套用 0.3 阈值。"""
    hybrid = HybridRetriever(vector_store, bm25)
    _, scores = hybrid.search_with_scores("过拟合", k=3)
    assert scores, "混合检索应返回结果"
    assert max(scores) <= 2.0 / (settings.rrf_k + 1) + 1e-6, f"RRF 分数异常: {scores}"


def test_reranker_graceful_degradation(vector_store, bm25):
    reranker = Reranker()
    hybrid = HybridRetriever(vector_store, bm25)
    docs = hybrid.search("过拟合", k=10)
    result = reranker.rerank("过拟合", docs, top_n=5)
    assert len(result) <= 5, "重排序结果不应超过 top_n"
    if reranker.is_available():
        # 模型可用时结果应为原候选的子集
        assert {d.page_content for d in result} <= {d.page_content for d in docs}
    else:
        # 降级：保持原顺序截断
        assert [d.page_content for d in result] == [d.page_content for d in docs[:5]]


# ── 索引一致性（回归：重复写入 / 陈旧索引缺陷）────────────────────


def test_rebuild_idempotent(chunks, embedder, chroma_dir):
    """同一语料重复构建：count 不变、内容不重复（确定性 ID）。

    注意：build_index 会删除并重建 collection，重建后旧实例的句柄失效
    （生产环境单实例不受影响），因此对比均通过新实例进行。
    """
    vs1 = VectorStore(embedder)
    vs1.build_index(chunks)
    count1 = vs1.count()
    docs_first = vs1.search("过拟合", k=3)
    assert count1 == len(chunks), f"首次构建 count={count1} != chunks={len(chunks)}"

    vs2 = VectorStore(embedder)
    vs2.build_index(chunks)
    assert vs2.count() == count1, "重复构建后 count 应不变"

    docs_second = vs2.search("过拟合", k=3)
    assert [d.page_content for d in docs_first] == [d.page_content for d in docs_second]


def test_load_or_build_reuses_matching_index(chunks, embedder, chroma_dir):
    vs = VectorStore(embedder)
    vs.build_index(chunks)
    reloaded = VectorStore.load_or_build(chunks, embedder)
    assert reloaded.count() == len(chunks)
    docs, scores = reloaded.search_with_scores("装饰器", k=2)
    assert len(docs) == 2 and len(scores) == 2, "复用索引后应能正常检索"


def test_load_or_build_rebuilds_on_fingerprint_mismatch(chunks, embedder, chroma_dir, monkeypatch):
    """embedding 模型指纹不匹配时应自动重建，而非静默复用陈旧索引。"""
    vs = VectorStore(embedder)
    vs.build_index(chunks)
    monkeypatch.setattr(settings, "embedding_model", "other-model-for-test")
    reloaded = VectorStore.load_or_build(chunks, embedder)
    assert reloaded.count() == len(chunks), "指纹不匹配应触发重建且 count 正确"
    # 重建后的 collection 应记录新指纹
    import chromadb

    col = chromadb.PersistentClient(path=str(chroma_dir)).get_collection(
        settings.chroma_collection_name
    )
    assert col.metadata.get("embedding_model") == "other-model-for-test"

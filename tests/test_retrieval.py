"""Task 3 验证：检索层（向量 + BM25）

验证项目：
1. "Python 装饰器" 向量检索 Top-3 命中 python_basics.md
2. "过拟合" BM25 检索 Top-3 命中 machine_learning_faq.md
3. 两种检索器 search_with_scores 返回格式一致 tuple[list[Document], list[float]]
4. 单次检索延迟 < 500ms
5. VectorStore.load_or_build 持久化加载正确
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from src.config import PROJECT_ROOT, settings
from src.ingestion import DocumentParser, TextChunker, EmbeddingService
from src.retrieval import VectorStore, BM25Retriever, HybridRetriever, Reranker

DATA_DIR = PROJECT_ROOT / settings.data_dir

# 复用单例，避免重复加载模型
_embedder: EmbeddingService | None = None


def _build_chunks():
    """解析 + 切分，返回 Document 列表（每个测试复用同一批）。"""
    parser = DocumentParser()
    docs = parser.parse_directory(DATA_DIR)
    chunker = TextChunker()
    return chunker.split(docs)


def _get_embedder() -> EmbeddingService:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService()
    return _embedder


def _source_name(doc) -> str:
    """从 Document.metadata.source 提取文件名。"""
    return os.path.basename(doc.metadata.get("source", ""))


def test_vector_recall():
    print("=" * 60)
    print("测试 1：向量检索召回（'Python 装饰器'）")
    print("=" * 60)
    chunks = _build_chunks()
    embedder = _get_embedder()
    vs = VectorStore(embedder)
    vs.build_index(chunks)

    docs = vs.search("Python 装饰器", k=3)
    assert len(docs) == 3, f"应返回 3 条，实际 {len(docs)}"
    sources = {_source_name(d) for d in docs}
    assert "python_basics.md" in sources, f"Top-3 应命中 python_basics.md，实际 {sources}"
    print(f"✅ 向量检索 Top-3 命中：{sources}")
    for i, d in enumerate(docs):
        print(f"   [{i+1}] {_source_name(d)} | score_meta={d.metadata.get('chunk_index')}")
    return vs


def test_bm25_recall():
    print("\n" + "=" * 60)
    print("测试 2：BM25 检索召回（'过拟合'）")
    print("=" * 60)
    chunks = _build_chunks()
    bm25 = BM25Retriever()
    bm25.build_index(chunks)

    docs = bm25.search("过拟合", k=3)
    assert len(docs) > 0, "BM25 应返回非空结果"
    sources = {_source_name(d) for d in docs}
    assert "machine_learning_faq.md" in sources, f"Top-3 应命中 machine_learning_faq.md，实际 {sources}"
    print(f"✅ BM25 检索 Top-3 命中：{sources}")
    return bm25


def test_score_format(vs: VectorStore, bm25: BM25Retriever):
    print("\n" + "=" * 60)
    print("测试 3：search_with_scores 返回格式一致性")
    print("=" * 60)
    v_docs, v_scores = vs.search_with_scores("Python 装饰器", k=3)
    b_docs, b_scores = bm25.search_with_scores("过拟合", k=3)

    assert isinstance(v_docs, list) and isinstance(v_scores, list), "向量结果应为 list"
    assert isinstance(b_docs, list) and isinstance(b_scores, list), "BM25 结果应为 list"
    assert len(v_docs) == len(v_scores), "向量 docs/scores 长度应一致"
    assert len(b_docs) == len(b_scores), "BM25 docs/scores 长度应一致"
    assert all(isinstance(s, float) for s in v_scores), "向量分数应为 float"
    assert all(isinstance(s, float) for s in b_scores), "BM25 分数应为 float"
    print(f"✅ 向量 scores: {[round(s, 4) for s in v_scores]}")
    print(f"✅ BM25  scores: {[round(s, 4) for s in b_scores]}")
    print(f"✅ 两个检索器 search_with_scores 返回 tuple[list[Document], list[float]] 格式一致")


def test_latency(vs: VectorStore, bm25: BM25Retriever):
    print("\n" + "=" * 60)
    print("测试 4：单次检索延迟 < 500ms")
    print("=" * 60)
    THRESHOLD_MS = 500

    t0 = time.perf_counter()
    vs.search("列表推导式", k=3)
    v_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    bm25.search("梯度下降", k=3)
    b_ms = (time.perf_counter() - t0) * 1000

    print(f"   向量检索延迟: {v_ms:.2f}ms")
    print(f"   BM25 检索延迟: {b_ms:.2f}ms")
    assert v_ms < THRESHOLD_MS, f"向量检索延迟 {v_ms:.2f}ms 超过 {THRESHOLD_MS}ms"
    assert b_ms < THRESHOLD_MS, f"BM25 检索延迟 {b_ms:.2f}ms 超过 {THRESHOLD_MS}ms"
    print(f"✅ 两者均 < {THRESHOLD_MS}ms")


def test_load_or_build():
    print("\n" + "=" * 60)
    print("测试 5：VectorStore.load_or_build 持久化加载")
    print("=" * 60)
    from pathlib import Path

    persist_dir = Path(PROJECT_ROOT / settings.chroma_persist_dir)
    # 首次构建（如目录已存在则加载，否则构建）
    chunks = _build_chunks()
    embedder = _get_embedder()

    vs_loaded = VectorStore.load_or_build(chunks, embedder)
    assert vs_loaded._store is not None, "load_or_build 后 _store 应非空"

    # 复用持久化索引也能正确检索
    docs, scores = vs_loaded.search_with_scores("装饰器", k=2)
    assert len(docs) == 2, f"持久化加载后应能检索到 2 条，实际 {len(docs)}"
    sources = {_source_name(d) for d in docs}
    assert "python_basics.md" in sources, f"持久化检索应命中 python_basics.md，实际 {sources}"
    exists_nonempty = persist_dir.exists() and any(persist_dir.iterdir())
    print(f"✅ 持久化目录 {persist_dir} 存在且非空: {exists_nonempty}")
    print(f"✅ 持久化加载后检索命中: {sources}")


def test_hybrid_comparison(vs: VectorStore, bm25: BM25Retriever):
    """Task 6 对比实验：纯向量 vs 纯 BM25 vs 混合 RRF vs 混合+Reranker。"""
    print("\n" + "=" * 60)
    print("测试 6：Task 6 对比实验（RRF 融合 + Reranker 降级）")
    print("=" * 60)
    hybrid = HybridRetriever(vs, bm25)
    reranker = Reranker()

    # 对比用例：(查询, 期望命中的源文件)
    cases = [
        ("Python 装饰器", "python_basics.md"),
        ("过拟合", "machine_learning_faq.md"),
        ("列表推导式", "python_basics.md"),
        ("梯度下降", "machine_learning_faq.md"),
        ("Transformer 自注意力", "machine_learning_faq.md"),
    ]

    def top5_hit(retriever, query: str, expected: str, strategy: str) -> tuple[bool, float]:
        """对单查询跑一次检索，返回 (Top-5 是否命中, 延迟ms)。"""
        t0 = time.perf_counter()
        if strategy == "rerank":
            docs, _ = hybrid.search_with_scores(query, k=10)
            docs = reranker.rerank(query, docs, top_n=5)
        else:
            docs = retriever.search(query, k=5)
        latency = (time.perf_counter() - t0) * 1000
        hit = expected in {_source_name(d) for d in docs}
        return hit, latency

    strategies = [
        ("纯向量", vs, "vector"),
        ("纯 BM25", bm25, "bm25"),
        ("混合 RRF", hybrid, "hybrid"),
        ("混合+Reranker", None, "rerank"),
    ]

    # 汇总统计
    results: dict[str, dict] = {name: {"hits": 0, "latencies": []} for name, _, _ in strategies}
    print(f"{'查询':<22} | " + " | ".join(f"{name:<14}" for name, _, _ in strategies))
    print("-" * 90)

    for query, expected in cases:
        row = f"{query:<20} | "
        for name, retriever, strat in strategies:
            hit, lat = top5_hit(retriever, query, expected, strat)
            results[name]["hits"] += int(hit)
            results[name]["latencies"].append(lat)
            row += f"{'✓' if hit else '✗'}{lat:>5.1f}ms{'':<5} | "
        print(row.rstrip(" | "))

    # 汇总表
    print("\n汇总（Top-5 召回率 / P99 延迟）：")
    print(f"{'策略':<16} | {'召回率':<10} | {'P99延迟':<10}")
    print("-" * 42)
    reranker_status = ""
    for name, _, _ in strategies:
        hits = results[name]["hits"]
        recall = hits / len(cases)
        lats = sorted(results[name]["latencies"])
        p99 = lats[-1] if lats else 0
        marker = " (Reranker 降级)" if name == "混合+Reranker" and not reranker.is_available() else ""
        print(f"{name:<16} | {recall*100:>6.1f}% ({hits}/{len(cases)})  | {p99:>6.1f}ms{marker}")
        reranker_status = marker

    # 断言：混合 RRF 召回率应不低于纯向量（核心价值验证）
    v_recall = results["纯向量"]["hits"] / len(cases)
    h_recall = results["混合 RRF"]["hits"] / len(cases)
    assert h_recall >= v_recall, (
        f"混合 RRF 召回率 {h_recall*100:.0f}% 应 ≥ 纯向量 {v_recall*100:.0f}%"
    )
    print(f"\n✅ 混合 RRF 召回率 {h_recall*100:.0f}% ≥ 纯向量 {v_recall*100:.0f}%（RRF 融合有效）")

    if reranker.is_available():
        print("✅ Reranker 模型可用，重排序生效")
    else:
        print("✅ Reranker 模型不可用，已优雅降级（跳过重排序，RRF 结果直接用）")


if __name__ == "__main__":
    print("RAGForge Task 3 + Task 6 验证：检索层（向量 + BM25 + 混合 + Reranker）\n")
    vs = test_vector_recall()
    bm25 = test_bm25_recall()
    test_score_format(vs, bm25)
    test_latency(vs, bm25)
    test_load_or_build()
    test_hybrid_comparison(vs, bm25)
    print("\n" + "=" * 60)
    print("🎉 Task 3 + Task 6 验证全部通过！")
    print("=" * 60)

"""评测 v2 检索层基线（不调 LLM）

读取统一 schema 数据集，在指定语料上比较：
vector / bm25 / hybrid / hybrid+reranker

指标：
- hit_any：top-k 中至少命中一个 expected_source
- hit_all：top-k 中包含全部 expected_source（多跳样本）
- MRR_any：第一个命中的 expected_source 的倒数排名

用法：
    uv run python scripts/eval_v2.py
    uv run python scripts/eval_v2.py --dataset tests/eval_dataset_hard.json --corpus-dir data/demo
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROJECT_ROOT as CFG_ROOT  # noqa: E402
from src.config import settings  # noqa: E402
from src.evaluation import dataset_hash, load_dataset, validate_dataset  # noqa: E402
from src.ingestion import DocumentParser, EmbeddingService, TextChunker  # noqa: E402
from src.retrieval import BM25Retriever, HybridRetriever, Reranker, VectorStore  # noqa: E402
from src.utils.logger import logger  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "reports" / "eval_v2_retrieval_report.json"
EVAL_K = 5


def _basename(source: str) -> str:
    return Path(str(source).replace("\\", "/")).name


def _rank_sources(docs, expected: list[str]) -> list[int]:
    ranks = []
    for source in expected:
        rank = 0
        for i, doc in enumerate(docs, start=1):
            if _basename(getattr(doc, "metadata", {}).get("source", "")) == Path(source).name:
                rank = i
                break
        ranks.append(rank)
    return ranks


def _ndcg_at_k(docs, expected: list[str], k: int) -> float:
    """文档级 NDCG@k（二值相关性，按来源去重后排序）。"""
    expected_names = {Path(source).name for source in expected}
    ranked: list[str] = []
    for doc in docs:
        name = _basename(getattr(doc, "metadata", {}).get("source", ""))
        if name and name not in ranked:
            ranked.append(name)
        if len(ranked) >= k:
            break
    gains = [1.0 if name in expected_names else 0.0 for name in ranked[:k]]
    dcg = sum(gain / math.log2(i + 1) for i, gain in enumerate(gains, start=1))
    ideal = min(len(expected_names), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


def _gate_failures(
    modes: list[dict],
    *,
    mode: str,
    min_hit_any: float | None,
    min_mrr: float | None,
) -> list[str]:
    """按指定模式检查回归门槛；返回空列表表示通过。"""
    if min_hit_any is None and min_mrr is None:
        return []
    selected = next((row for row in modes if row["mode"] == mode), None)
    if selected is None:
        return [f"没有找到模式 {mode}"]
    failures = []
    if min_hit_any is not None and selected["hit_any"] < min_hit_any:
        failures.append(
            f"{mode} hit_any={selected['hit_any']:.4f} < {min_hit_any:.4f}"
        )
    if min_mrr is not None and selected["mrr_any"] < min_mrr:
        failures.append(f"{mode} mrr_any={selected['mrr_any']:.4f} < {min_mrr:.4f}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 v2 检索层基线")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "tests" / "eval_dataset.json"))
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="跳过 CrossEncoder 重排（CI 或无模型缓存时使用）",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--dataset-license", default="unspecified")
    parser.add_argument("--gate-mode", default="hybrid")
    parser.add_argument("--min-hit-any", type=float, default=None)
    parser.add_argument("--min-mrr", type=float, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    items = load_dataset(dataset_path)
    errors = validate_dataset(items, corpus_dir=args.corpus_dir)
    if errors:
        print("数据集校验失败：")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else CFG_ROOT / settings.eval_data_dir
    samples = [
        item for item in items
        if item.expected_behavior == "answer" and item.expected_sources
    ]
    if not samples:
        print("没有带 expected_sources 的可答题")
        sys.exit(1)

    settings.chroma_persist_dir = settings.eval_chroma_persist_dir
    settings.chroma_collection_name = settings.eval_chroma_collection_name

    t0 = time.perf_counter()
    docs = DocumentParser().parse_directory(corpus_dir)
    chunks = TextChunker().split(docs)
    embedder = EmbeddingService()
    vector = VectorStore(embedder)
    if args.rebuild:
        vector.build_index(chunks)
    else:
        vector = VectorStore.load_or_build(chunks, embedder)
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    hybrid = HybridRetriever(vector, bm25)
    reranker = Reranker()
    logger.info(f"[eval-v2] 工件就绪：docs={len(docs)}, chunks={len(chunks)}")

    modes = {
        "vector": lambda q: vector.search(q, k=EVAL_K),
        "bm25": lambda q: bm25.search(q, k=EVAL_K),
        "hybrid": lambda q: hybrid.search(q, k=EVAL_K),
        "reranker": lambda q: reranker.rerank(
            q,
            hybrid.search(q, k=max(EVAL_K * 2, settings.reranker_top_n)),
            top_n=EVAL_K,
        ),
    }
    if args.skip_reranker:
        modes.pop("reranker")

    report = {
        "dataset": str(dataset_path),
        "dataset_name": args.dataset_name or dataset_path.stem,
        "dataset_version": args.dataset_version,
        "dataset_license": args.dataset_license,
        "dataset_hash": dataset_hash(items),
        "corpus_dir": str(corpus_dir),
        "samples": len(samples),
        "k": EVAL_K,
        "metric": "doc-level hit_any/hit_all/MRR_any/NDCG@5",
        "modes": [],
    }
    for name, fn in modes.items():
        any_hits = all_hits = 0
        rr_sum = 0.0
        ndcg_sum = 0.0
        for item in samples:
            docs_hit = fn(item.query)
            ranks = _rank_sources(docs_hit, item.expected_sources)
            if any(r > 0 for r in ranks):
                any_hits += 1
                rr_sum += 1.0 / min(r for r in ranks if r > 0)
            if all(r > 0 for r in ranks):
                all_hits += 1
            ndcg_sum += _ndcg_at_k(docs_hit, item.expected_sources, EVAL_K)
        row = {
            "mode": name,
            "hit_any": any_hits / len(samples),
            "hit_all": all_hits / len(samples),
            "mrr_any": rr_sum / len(samples),
            "ndcg_at5": ndcg_sum / len(samples),
            "hit_any_count": any_hits,
            "hit_all_count": all_hits,
        }
        report["modes"].append(row)
        print(
            f"{name:10s} hit_any={row['hit_any']*100:5.1f}% "
            f"hit_all={row['hit_all']*100:5.1f}% "
            f"mrr_any={row['mrr_any']*100:5.1f}% "
            f"ndcg@5={row['ndcg_at5']*100:5.1f}%"
        )

    report["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存：{args.report}")

    failures = _gate_failures(
        report["modes"],
        mode=args.gate_mode,
        min_hit_any=args.min_hit_any,
        min_mrr=args.min_mrr,
    )
    if failures:
        for failure in failures:
            print(f"回归门禁失败：{failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()

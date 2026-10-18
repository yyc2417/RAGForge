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


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 v2 检索层基线")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "tests" / "eval_dataset.json"))
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
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

    report = {
        "dataset": str(dataset_path),
        "dataset_hash": dataset_hash(items),
        "corpus_dir": str(corpus_dir),
        "samples": len(samples),
        "modes": [],
    }
    for name, fn in modes.items():
        any_hits = all_hits = 0
        rr_sum = 0.0
        for item in samples:
            docs_hit = fn(item.query)
            ranks = _rank_sources(docs_hit, item.expected_sources)
            if any(r > 0 for r in ranks):
                any_hits += 1
                rr_sum += 1.0 / min(r for r in ranks if r > 0)
            if all(r > 0 for r in ranks):
                all_hits += 1
        row = {
            "mode": name,
            "hit_any": any_hits / len(samples),
            "hit_all": all_hits / len(samples),
            "mrr_any": rr_sum / len(samples),
            "hit_any_count": any_hits,
            "hit_all_count": all_hits,
        }
        report["modes"].append(row)
        print(
            f"{name:10s} hit_any={row['hit_any']*100:5.1f}% "
            f"hit_all={row['hit_all']*100:5.1f}% "
            f"mrr_any={row['mrr_any']*100:5.1f}%"
        )

    report["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已保存：{args.report}")


if __name__ == "__main__":
    main()

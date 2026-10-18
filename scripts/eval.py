"""RAGForge 自动化评估脚本

跑 4 套 RAG 配置，计算召回率、幻觉率、拒答准确率、延迟、token 消耗，输出对比表。

四套配置（固定 hybrid 后的评测矩阵）：
- baseline：线性 vector RAG（无 Agent）
- linear_hybrid：线性 hybrid RAG（无 Agent）
- agent：7 节点 Agent 主链路 + 固定 HybridRetriever（无 Reranker）
- agent_reranker：7 节点 Agent 主链路 + HybridRetriever + Reranker

数据集分层（对齐业界惯例：RAGAS 题型分布 / SQuAD 式 is_impossible）：
- 可答题（answerable=true 且有 expected_source(s)）：chunk 级召回判定
  （来源匹配；配置了 gold_keywords 时再叠加关键词条件）→ Recall@5 + MRR@5；
  生成层判忠实度；有 reference_answer 时附加词面 token F1（v2）
- 不可答题（answerable=false，语料外问题）：LLM 裁决「诚实拒答 / 编造 /
  其他」→ 拒答准确率 + 编造率——幻觉率在这里才有真实含义
- chitchat：不计召回与忠实度统计

数据集支持 v1 列表与 v2 schema（`expected_sources` / `reference_answer` /
`expected_behavior`），可用 `--dataset` / `--corpus-dir` 指向 CRUD-RAG 或多格式集。

运行：
    uv run python scripts/eval.py                       # 完整评估（需 DEEPSEEK_API_KEY）
    uv run python scripts/eval.py --mode agent          # 只跑指定套
    uv run python scripts/eval.py --resume              # 增量补跑（跳过报告中已完成模式）
    uv run python scripts/eval.py --rebuild             # 评估前清库重建索引
    uv run python scripts/eval.py --retrieval-only      # 检索层离线评估（不调 LLM，免 API）
    uv run python scripts/eval.py --dataset tests/eval_dataset_multiformat.json \
        --corpus-dir data/demo --retrieval-only --skip-reranker
输出：
    - 终端 rich 表格对比矩阵
    - reports/eval_report.json 完整报告（retrieval-only 输出 reports/eval_retrieval_report.json）
    - 路径可分别用 --report / --retrieval-report 覆盖
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import jieba
from rich.console import Console
from rich.table import Table

# 项目根目录加入 sys.path（脚本独立运行场景）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import build_agent_graph_from_pipeline, initial_state  # noqa: E402
from src.config import PROJECT_ROOT as CFG_ROOT  # noqa: E402
from src.config import settings  # noqa: E402
from src.generation.llm_client import LLMClient  # noqa: E402
from src.generation.prompts import PromptManager  # noqa: E402
from src.generation.schemas import RefusalJudgeResult, VerifyResult  # noqa: E402
from src.ingestion import DocumentParser, EmbeddingService, TextChunker  # noqa: E402
from src.retrieval import BM25Retriever, HybridRetriever, Reranker, VectorStore  # noqa: E402
from src.utils.logger import logger  # noqa: E402
from src.utils.metrics import get_current_collector  # noqa: E402

console = Console()
DATA_DIR = CFG_ROOT / settings.eval_data_dir
DATASET_PATH = PROJECT_ROOT / "tests" / "eval_dataset.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "eval_report.json"
RETRIEVAL_REPORT_PATH = PROJECT_ROOT / "reports" / "eval_retrieval_report.json"
MODES = ["baseline", "linear_hybrid", "agent", "agent_reranker"]
EVAL_K = 5  # 评估统一检索数量，与 recall_top5 指标口径一致


def _build_base_artifacts(
    rebuild: bool = False, corpus_dir: Path | str | None = None
) -> tuple[list, VectorStore, BM25Retriever]:
    """解析文档、切分、构建向量/BM25 索引（所有模式共享）。

    Args:
        rebuild: 为 True 时清库重建向量索引，否则按指纹校验复用
        corpus_dir: 评测语料目录；默认 settings.eval_data_dir
    """
    # 评测索引与 app 运行时索引隔离，避免互相按 count 重建
    settings.chroma_persist_dir = settings.eval_chroma_persist_dir
    settings.chroma_collection_name = settings.eval_chroma_collection_name

    parser = DocumentParser()
    docs = parser.parse_directory(Path(corpus_dir) if corpus_dir else DATA_DIR)
    chunker = TextChunker()
    chunks = chunker.split(docs)
    embedder = EmbeddingService()
    if rebuild:
        vector_store = VectorStore(embedder)
        vector_store.build_index(chunks)
    else:
        vector_store = VectorStore.load_or_build(chunks, embedder)
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    logger.info(f"[eval] 基础工件就绪：chunks={len(chunks)}")
    return chunks, vector_store, bm25


def _source_filename(source: str) -> str:
    """从 metadata.source 提取纯文件名（去路径），便于匹配 expected_source。"""
    if not source:
        return ""
    return Path(str(source)).name


def _doc_sources(state: dict) -> list[str]:
    """从 final_state 提取检索文档的来源文件名列表。"""
    docs = state.get("documents", []) or []
    return [
        _source_filename(getattr(doc, "metadata", {}).get("source", ""))
        for doc in docs
        if hasattr(doc, "metadata")
    ]


def _expected_sources(item: dict) -> list[str]:
    """兼容 v1 expected_source 与 v2 expected_sources。"""
    sources = item.get("expected_sources")
    if isinstance(sources, list) and sources:
        return [str(source) for source in sources if str(source).strip()]
    source = item.get("expected_source")
    return [str(source)] if source else []


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


def _answer_tokens(text: str) -> list[str]:
    """中英混合分词，过滤纯标点 token。"""
    return [
        token
        for token in jieba.lcut(str(text).lower())
        if token.strip() and _TOKEN_RE.search(token)
    ]


def _answer_f1(prediction: str, reference: str | None) -> float | None:
    """参考答案的词面 token F1；无参考答案时返回 None。"""
    if not reference or not str(reference).strip():
        return None
    pred_tokens = _answer_tokens(prediction)
    ref_tokens = _answer_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _chunk_hit_rank(docs: list, expected_sources: list[str] | str, gold_keywords: list[str]) -> int:
    """chunk 级命中判定：返回第一个命中 chunk 的 1-based 排名，未命中返回 0。

    命中条件：来源文件属于 expected_sources；若配置了 gold_keywords，
    chunk 还需包含任一关键词。v2 公开集可只给来源、不给关键词。
    """
    sources = [expected_sources] if isinstance(expected_sources, str) else expected_sources
    names = {_source_filename(str(source)) for source in sources}
    for rank, doc in enumerate(docs, start=1):
        if not hasattr(doc, "page_content"):
            continue
        src = _source_filename(getattr(doc, "metadata", {}).get("source", ""))
        keyword_ok = not gold_keywords or any(kw in doc.page_content for kw in gold_keywords)
        if src in names and keyword_ok:
            return rank
    return 0


def _index_fingerprint(chunks: list, vector_store: VectorStore) -> dict:
    """索引指纹：写入报告，保证评估结果可归因、可复现。"""
    return {
        "chunks": len(chunks),
        "collection_count": vector_store.count(),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieval_k": settings.retrieval_k,
        "rrf_k": settings.rrf_k,
        "reranker_top_n": settings.reranker_top_n,
    }


class Evaluator:
    """自动化评估器：跑 4 套配置，计算召回率/幻觉率/延迟/token。

    LLM 客户端惰性创建：--retrieval-only 离线模式无需 API key。

    Args:
        dataset_path: 评估数据集 JSON 路径
        rebuild: 为 True 时评估前清库重建索引
    """

    def __init__(
        self,
        dataset_path: Path | str = DATASET_PATH,
        rebuild: bool = False,
        corpus_dir: Path | str | None = None,
    ) -> None:
        self.dataset = self._load_dataset(dataset_path)
        self._chunks, self._vector_store, self._bm25 = _build_base_artifacts(
            rebuild, corpus_dir
        )
        self._llm: LLMClient | None = None
        self._reranker = Reranker()  # 单例，不可用时自动降级
        # 缓存各模式的执行管道
        self._pipelines: dict[str, object] = {}

    @staticmethod
    def _load_dataset(path: Path | str) -> list[dict]:
        """加载评估数据集（支持列表或 {"items": [...]}）。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("items", [])
        logger.info(f"[eval] 数据集加载：{len(data)} 个 QA 对")
        return data

    def _ensure_llm(self) -> LLMClient:
        """惰性创建 LLM 客户端（首次使用时校验 API key）。"""
        if self._llm is None:
            key = settings.deepseek_key_value()
            if not key or key.startswith("sk-your"):
                console.print("[red]x DEEPSEEK_API_KEY 未配置（请在 .env 中填入有效 key）[/red]")
                console.print("  完整评估需真实调用 DeepSeek；离线检索评估请用 --retrieval-only。")
                sys.exit(1)
            self._llm = LLMClient()
        return self._llm

    def _get_pipeline(self, mode: str):
        """按模式惰性构建并缓存执行管道。

        - baseline：返回 (vector_store, llm) 二元组，线性 vector 执行
        - linear_hybrid：返回 (hybrid, llm) 二元组，线性 hybrid 执行
        - agent / agent_reranker：返回编译后的 7 节点 Agent graph
        """
        if mode in self._pipelines:
            return self._pipelines[mode]

        if mode == "baseline":
            pipeline = (self._vector_store, self._ensure_llm())
        elif mode == "linear_hybrid":
            pipeline = (HybridRetriever(self._vector_store, self._bm25), self._ensure_llm())
        else:
            reranker = self._reranker if mode == "agent_reranker" else None
            pipeline = build_agent_graph_from_pipeline(
                vector_store=self._vector_store,
                bm25_retriever=self._bm25,
                llm_client=self._ensure_llm(),
                hybrid_retriever=HybridRetriever(self._vector_store, self._bm25),
                reranker=reranker,
            )
        self._pipelines[mode] = pipeline
        return pipeline

    def _judge_faithfulness(
        self, query: str, answer: str, docs: list, keywords: list[str]
    ) -> tuple[bool | None, str]:
        """统一忠实度裁决（所有模式共用同一口径）。

        优先 LLM 判定（与 verify 节点同款 prompt）；失败回退关键词粗判并标记
        来源，不再默认 True（避免幻觉被系统性美化）。

        Returns:
            (is_faithful, source)：source ∈ {"llm_judge", "keyword"}
        """
        try:
            llm = self._ensure_llm()
            context = (
                "\n\n".join(getattr(d, "page_content", "") for d in docs) or "（无检索文档）"
            )
            result: VerifyResult = llm.invoke_structured(
                PromptManager.VERIFY_PROMPT, VerifyResult,
                query=query, answer=answer, context=context,
            )
            return result.is_faithful, "llm_judge"
        except Exception as e:  # noqa: BLE001 - LLM 失败回退关键词，绝不默认忠实
            logger.warning(f"[eval] LLM 忠实度裁决失败，回退关键词粗判：{e}")
            return self._keyword_check(answer, keywords), "keyword"

    def _run_single(self, mode: str, item: dict) -> dict:
        """对单个 QA 项执行指定模式，返回结果字典（按题型分层判定）。"""
        query = item["query"]
        expected_sources = _expected_sources(item)
        answerable = item.get("answerable", bool(expected_sources))
        is_chitchat = item.get("query_type") == "chitchat"
        gold_keywords = item.get("gold_keywords", item.get("expected_answer_keywords", []))
        reference_answer = item.get("reference_answer")

        t0 = time.perf_counter()
        retrieved_sources: list[str] = []
        answer_text = ""
        is_faithful: bool | None = None
        faith_source = "unverified"
        refusal_verdict: str | None = None
        chunk_rank = 0
        error: str | None = None

        try:
            if mode in ("baseline", "linear_hybrid"):
                retriever, llm = self._get_pipeline(mode)
                docs = retriever.search(query, k=EVAL_K)
                retrieved_sources = [
                    _source_filename(getattr(d, "metadata", {}).get("source", ""))
                    for d in docs
                ]
                answer_text = llm.generate(query, docs)
                verify_result: bool | None = None
            else:
                graph = self._get_pipeline(mode)
                final_state = graph.invoke(
                    initial_state(query),
                    config={"recursion_limit": 50},
                )
                retrieved_sources = _doc_sources(final_state)
                answer_text = final_state.get("answer", "")
                docs = final_state.get("documents", []) or []
                verify_result = final_state.get("is_faithful")

            # ── 按题型分层判定 ──
            if expected_sources:
                # 可答题：chunk 级召回 + 忠实度
                chunk_rank = _chunk_hit_rank(docs, expected_sources, gold_keywords)
                if mode != "baseline" and verify_result is not None:
                    is_faithful, faith_source = verify_result, "llm_verify"
                else:
                    # baseline 无 verify 节点 / agent verify 失败：统一裁决器兜底
                    is_faithful, faith_source = self._judge_faithfulness(
                        query, answer_text, docs, gold_keywords
                    )
            elif is_chitchat:
                faith_source = "n/a"  # 寒暄不计忠实度统计
            else:
                # 不可答题（语料外问题）：拒答行为裁决
                refusal_verdict = self._judge_refusal(query, answer_text)
        except Exception as e:  # noqa: BLE001 - 失败样本单独归类，不污染指标
            logger.error(f"[eval] {mode} 执行失败 query='{query}': {e}")
            error = str(e)
            answer_text = f"[ERROR] {e}"

        e2e_ms = (time.perf_counter() - t0) * 1000
        # 统一记录 e2e 延迟（仅成功样本，失败样本不污染分位统计）
        if error is None:
            get_current_collector().record_e2e_latency(e2e_ms)

        answer_f1_score = (
            _answer_f1(answer_text, reference_answer)
            if expected_sources and reference_answer
            else None
        )
        return {
            "query": query,
            "query_type": item.get("query_type", "factual"),
            "answerable": answerable,
            "expected_source": expected_sources[0] if expected_sources else None,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "chunk_rank": chunk_rank,
            "hit": chunk_rank > 0,
            "answer": answer_text,
            "reference_answer": reference_answer,
            "answer_f1": answer_f1_score,
            "is_faithful": is_faithful,
            "faithfulness_source": faith_source,
            "refusal_verdict": refusal_verdict,
            "error": error,
            "e2e_ms": e2e_ms,
        }

    @staticmethod
    def _keyword_check(answer: str, keywords: list[str]) -> bool:
        """关键词命中检查（LLM 裁决失败时的粗判 fallback，口径宽松）。"""
        if not keywords:
            return True
        return any(kw in answer for kw in keywords)

    def _judge_refusal(self, query: str, answer: str) -> str:
        """不可答题行为裁决：refused(诚实拒答) / fabricated(编造) / other。

        幻觉率的真战场：语料外问题若被强行作答（编造具体事实）即计为编造。
        """
        try:
            llm = self._ensure_llm()
            result: RefusalJudgeResult = llm.invoke_structured(
                PromptManager.UNANSWERABLE_JUDGE_PROMPT, RefusalJudgeResult,
                query=query, answer=answer,
            )
            return result.verdict
        except Exception as e:  # noqa: BLE001 - 裁决失败不臆断，记为 other
            logger.warning(f"[eval] 拒答裁决失败，记为 other：{e}")
            return "other"

    def evaluate_mode(self, mode: str) -> dict:
        """对单套模式跑全部数据集，返回汇总指标。"""
        console.print(f"\n[bold cyan]> 运行模式：{mode}[/bold cyan]")
        # 评估为单线程进程，直接使用进程级默认收集器
        metrics = get_current_collector()
        metrics.reset()

        results = []
        for i, item in enumerate(self.dataset, 1):
            console.print(f"  [{i}/{len(self.dataset)}] {item['query'][:30]}...", end=" ")
            r = self._run_single(mode, item)
            results.append(r)
            if r["error"]:
                status = "E"
            elif r["hit"] or not r["expected_sources"]:
                status = "+"
            else:
                status = "-"
            console.print(f"{status} {r['e2e_ms']:.0f}ms")

        # ── 汇总：按题型分层统计，失败/未验证样本单独归类不进分母 ──
        valid = [r for r in results if not r.get("error")]
        answerable = [r for r in valid if r["expected_sources"]]
        unanswerable = [
            r for r in valid
            if not r["expected_sources"] and r["query_type"] != "chitchat"
        ]
        # 检索层：chunk 级召回 + MRR（排名质量，能区分检索策略的价值）
        hits = sum(1 for r in answerable if r["chunk_rank"] > 0)
        recall = hits / len(answerable) if answerable else 0.0
        mrr = (
            sum(1.0 / r["chunk_rank"] for r in answerable if r["chunk_rank"] > 0)
            / len(answerable)
        ) if answerable else 0.0
        # 生成层：可答题忠实度
        verified = [r for r in answerable if r["is_faithful"] is not None]
        faithful = sum(1 for r in verified if r["is_faithful"] is True)
        faith_rate = faithful / len(verified) if verified else 0.0
        # 不可答题：拒答行为
        refusals = sum(1 for r in unanswerable if r.get("refusal_verdict") == "refused")
        fabrications = sum(
            1 for r in unanswerable if r.get("refusal_verdict") == "fabricated"
        )
        refusal_accuracy = refusals / len(unanswerable) if unanswerable else 0.0
        fabrication_rate = fabrications / len(unanswerable) if unanswerable else 0.0
        f1_values = [r["answer_f1"] for r in answerable if r["answer_f1"] is not None]

        summary = metrics.get_summary()
        e2e = summary.get("e2e_latency_ms", {})
        token = summary.get("token_usage", {})

        return {
            "mode": mode,
            "samples": len(results),
            "errors": sum(1 for r in results if r.get("error")),
            "unverified": sum(1 for r in answerable if r["is_faithful"] is None),
            "answerable_count": len(answerable),
            "unanswerable_count": len(unanswerable),
            "recall_top5": round(recall, 4),
            "mrr_at5": round(mrr, 4),
            "hallucination_rate": round(1.0 - faith_rate, 4),
            "faithfulness_rate": round(faith_rate, 4),
            "answer_f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else None,
            "answer_f1_samples": len(f1_values),
            "refusal_accuracy": round(refusal_accuracy, 4),
            "fabrication_rate": round(fabrication_rate, 4),
            "latency_p50_ms": e2e.get("p50", 0.0),
            "latency_max_ms": e2e.get("max", 0.0),
            # P95/P99 保留在 JSON 中；45 题规模下 P99 近似最大值，主表改用 P50 + max
            "latency_p95_ms": e2e.get("p95", 0.0),
            "latency_p99_ms": e2e.get("p99", 0.0),
            "avg_token": round(token.get("avg_per_call", 0.0)) if token else 0,
            "total_token": token.get("total", 0) if token else 0,
            "details": results,
        }

    def run_all(self) -> dict:
        """跑全部 4 套模式，返回完整报告。"""
        report = {
            "modes": [],
            "targets": self._targets(),
            "index": _index_fingerprint(self._chunks, self._vector_store),
        }
        for mode in MODES:
            mode_result = self.evaluate_mode(mode)
            report["modes"].append(mode_result)
        return report

    @staticmethod
    def _targets() -> dict:
        """目标指标门槛（召回/MRR 针对检索层，幻觉/拒答针对生成层）。"""
        return {
            "recall_top5": 0.90,
            "mrr_at5": 0.70,
            "hallucination_rate": 0.05,
            "refusal_accuracy": 0.80,
            "latency_p99_ms": 2000,
            "avg_token": 800,
        }

    def print_report(self, report: dict) -> None:
        """用 rich 表格输出对比矩阵。"""
        targets = report["targets"]
        console.print()
        console.rule("[bold]RAGForge 评估报告 — 4 套 RAG 配置对比[/bold]")

        table = Table(show_lines=True, header_style="bold magenta")
        table.add_column("配置", style="cyan", no_wrap=True)
        table.add_column("Recall@5", justify="right")
        table.add_column("MRR@5", justify="right")
        table.add_column("Ans F1", justify="right")
        table.add_column("幻觉率(可答)", justify="right")
        table.add_column("拒答准确率", justify="right")
        table.add_column("P50 延迟(ms)", justify="right")
        table.add_column("Max 延迟(ms)", justify="right")
        table.add_column("平均 Token", justify="right")
        table.add_column("错误/未验证", justify="right")
        table.add_column("达标", justify="center")

        for m in report["modes"]:
            recall_ok = m["recall_top5"] >= targets["recall_top5"]
            mrr_ok = m["mrr_at5"] >= targets["mrr_at5"]
            halluc_ok = m["hallucination_rate"] <= targets["hallucination_rate"]
            refusal_ok = m["refusal_accuracy"] >= targets["refusal_accuracy"]
            token_ok = 0 < m["avg_token"] <= targets["avg_token"]
            all_ok = recall_ok and mrr_ok and halluc_ok and refusal_ok and token_ok

            table.add_row(
                m["mode"],
                self._fmt(m["recall_top5"], recall_ok, percent=True),
                self._fmt(m["mrr_at5"], mrr_ok, percent=True),
                (
                    f'{m["answer_f1"] * 100:.1f}%'
                    if m.get("answer_f1") is not None
                    else "—"
                ),
                self._fmt(m["hallucination_rate"], halluc_ok, percent=True),
                self._fmt(m["refusal_accuracy"], refusal_ok, percent=True),
                f'{m.get("latency_p50_ms", 0.0):.0f}',
                f'{m.get("latency_max_ms", 0.0):.0f}',
                self._fmt(m["avg_token"], token_ok),
                f"{m.get('errors', 0)}/{m.get('unverified', 0)}",
                "[green]V[/green]" if all_ok else "[yellow]~[/yellow]",
            )

        console.print(table)
        console.print(
            "[dim]图例：V 质量指标全达标 | ~ 部分达标；延迟为 P50 + max 观测值，"
            "样本小不参与达标判定；幻觉率=可答题不忠实占比，"
            "拒答准确率=不可答题诚实拒答占比，编造率见 JSON 报告[/dim]"
        )

    @staticmethod
    def _fmt(value: float, ok: bool, percent: bool = False) -> str:
        """格式化数值，达标绿色、未达标蓝色；无数据显示 —。"""
        if value == 0:
            text = "—"
        elif percent:
            text = f"{value * 100:.1f}%"
        else:
            text = f"{value:.0f}"
        color = "green" if ok else "blue"
        return f"[{color}]{text}[/{color}]"

    def save_report(self, report: dict, path: Path | str = REPORT_PATH) -> Path:
        """保存完整报告为 JSON。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"\n[green]报告已保存：{path}[/green]")
        return path


def run_retrieval_only(
    rebuild: bool = False,
    dataset_path: Path | str = DATASET_PATH,
    corpus_dir: Path | str | None = None,
    skip_reranker: bool = False,
    report_path: Path | str = RETRIEVAL_REPORT_PATH,
) -> None:
    """检索层离线评估：chunk 级 Recall@k + MRR@k，不调用 LLM（免 API key）。

    用于修复期间的快速回归验证；完整评估（忠实度/拒答/延迟/token）仍需完整模式。
    """
    console.rule("[bold]RAGForge 检索层离线评估（不调用 LLM）[/bold]")
    chunks, vs, bm25 = _build_base_artifacts(rebuild, corpus_dir)
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", [])
    items = [it for it in data if _expected_sources(it)]
    if not items:
        console.print("[red]数据集中没有带 expected_source(s) 的样本，无法评估召回[/red]")
        sys.exit(1)

    hybrid = HybridRetriever(vs, bm25)
    reranker = Reranker()
    strategies: dict[str, Callable[[str], list]] = {
        "vector": lambda q: vs.search(q, k=EVAL_K),
        "bm25": lambda q: bm25.search(q, k=EVAL_K),
        "hybrid": lambda q: hybrid.search(q, k=EVAL_K),
        "reranker": lambda q: reranker.rerank(
            q,
            hybrid.search(q, k=max(EVAL_K * 2, settings.reranker_top_n)),
            top_n=EVAL_K,
        ),
    }
    if skip_reranker:
        strategies.pop("reranker")

    table = Table(show_lines=True, header_style="bold magenta")
    table.add_column("检索策略", style="cyan", no_wrap=True)
    table.add_column(f"Recall@{EVAL_K}", justify="right")
    table.add_column(f"MRR@{EVAL_K}", justify="right")
    table.add_column("命中/样本", justify="right")

    modes_report = []
    for name, fn in strategies.items():
        hits = 0
        rr_sum = 0.0
        for it in items:
            docs = fn(it["query"])
            rank = _chunk_hit_rank(docs, _expected_sources(it), it.get("gold_keywords", []))
            if rank > 0:
                hits += 1
                rr_sum += 1.0 / rank
        recall = hits / len(items)
        mrr = rr_sum / len(items)
        modes_report.append(
            {"mode": name, "recall_top5": round(recall, 4), "mrr_at5": round(mrr, 4), "hits": hits}
        )
        table.add_row(name, f"{recall * 100:.1f}%", f"{mrr * 100:.1f}%", f"{hits}/{len(items)}")
        console.print(f"  [{name}] recall={recall * 100:.1f}% mrr={mrr * 100:.1f}% ({hits}/{len(items)})")

    console.print(table)
    report = {
        "retrieval_only": True,
        "dataset": str(dataset_path),
        "corpus_dir": str(corpus_dir) if corpus_dir else str(DATA_DIR),
        "eval_k": EVAL_K,
        "samples": len(items),
        "modes": modes_report,
        "index": _index_fingerprint(chunks, vs),
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[green]报告已保存：{report_path}[/green]")


def main() -> None:
    """评估主入口。

    用法：
        uv run python scripts/eval.py                    # 跑全部 4 套
        uv run python scripts/eval.py --mode agent       # 只跑指定套
        uv run python scripts/eval.py --resume           # 增量补跑
        uv run python scripts/eval.py --rebuild          # 清库重建后评估
        uv run python scripts/eval.py --retrieval-only   # 离线检索评估（免 API）
        uv run python scripts/eval.py --dataset tests/eval_dataset_multiformat.json \
            --corpus-dir data/demo --retrieval-only --skip-reranker
    """
    parser = argparse.ArgumentParser(description="RAGForge 自动化评估")
    parser.add_argument("--mode", choices=MODES, help="只跑指定模式（默认全部）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 条样本（本地 smoke）")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH,
                        help="评测数据集（v1 列表或 v2 schema）")
    parser.add_argument("--corpus-dir", type=Path, default=None,
                        help="评测语料目录；默认 settings.eval_data_dir")
    parser.add_argument("--report", type=Path, default=REPORT_PATH,
                        help="完整评估报告路径")
    parser.add_argument("--retrieval-report", type=Path, default=RETRIEVAL_REPORT_PATH,
                        help="离线检索报告路径")
    parser.add_argument("--resume", action="store_true",
                        help="增量模式：读取已有报告，跳过已完成的模式，只补未完成的")
    parser.add_argument("--rebuild", action="store_true",
                        help="评估前清库重建向量索引（不校验指纹直接重建）")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="只评估各检索策略的召回率，不调用 LLM（免 API key）")
    parser.add_argument("--skip-reranker", action="store_true",
                        help="检索层评估跳过 CrossEncoder 重排")
    args = parser.parse_args()

    # 统一评估口径：检索数量对齐 recall_top5 指标
    if settings.retrieval_k < EVAL_K:
        logger.info(
            f"[eval] retrieval_k {settings.retrieval_k} -> {EVAL_K}（对齐 recall_top5 口径）"
        )
        settings.retrieval_k = EVAL_K

    console.rule("[bold]RAGForge 自动化评估[/bold]")
    console.print(f"数据集：{args.dataset}")
    if args.corpus_dir:
        console.print(f"语料：{args.corpus_dir}")

    if args.retrieval_only:
        run_retrieval_only(
            rebuild=args.rebuild,
            dataset_path=args.dataset,
            corpus_dir=args.corpus_dir,
            skip_reranker=args.skip_reranker,
            report_path=args.retrieval_report,
        )
        return

    evaluator = Evaluator(
        dataset_path=args.dataset,
        rebuild=args.rebuild,
        corpus_dir=args.corpus_dir,
    )
    if args.limit:
        evaluator.dataset = evaluator.dataset[: args.limit]
        console.print(f"样本限制：{len(evaluator.dataset)} 条（--limit {args.limit}）")

    # 决定要跑哪些模式
    existing_report: dict | None = None
    if args.resume and args.report.exists():
        try:
            existing_report = json.loads(args.report.read_text(encoding="utf-8"))
            done_modes = {m["mode"] for m in existing_report.get("modes", [])}
            if args.mode:
                # resume + 指定 mode：只补该模式（若已完成则跳过）
                if args.mode in done_modes:
                    console.print(f"[green]{args.mode} 已在报告中，无需重跑[/green]")
                    evaluator.print_report(existing_report)
                    return
                modes_to_run = [args.mode]
                pending = modes_to_run
            else:
                pending = [m for m in MODES if m not in done_modes]
                modes_to_run = pending
            console.print(f"[yellow]增量模式：已完成 {sorted(done_modes)}，待补 {pending}[/yellow]")
            if not pending:
                console.print("[green]所有模式已完成，无需重跑[/green]")
                evaluator.print_report(existing_report)
                return
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]读取已有报告失败（{e}），从头开始[/red]")
            modes_to_run = [args.mode] if args.mode else MODES
    elif args.mode:
        modes_to_run = [args.mode]
    else:
        modes_to_run = MODES

    console.print(f"模式：{', '.join(modes_to_run)}")

    # 执行评估
    new_results = []
    for mode in modes_to_run:
        new_results.append(evaluator.evaluate_mode(mode))

    # 合并报告（增量模式下保留已有结果）
    if existing_report:
        # 按 MODES 顺序合并：已有的 + 新跑的
        new_by_mode = {m["mode"]: m for m in new_results}
        merged_modes = []
        for mode in MODES:
            for m in existing_report["modes"]:
                if m["mode"] == mode and mode not in new_by_mode:
                    merged_modes.append(m)
            if mode in new_by_mode:
                merged_modes.append(new_by_mode[mode])
        report = {
            "modes": merged_modes,
            "targets": Evaluator._targets(),
            "index": _index_fingerprint(evaluator._chunks, evaluator._vector_store),
        }
    else:
        report = {
            "modes": new_results,
            "targets": Evaluator._targets(),
            "index": _index_fingerprint(evaluator._chunks, evaluator._vector_store),
        }

    evaluator.print_report(report)
    evaluator.save_report(report, args.report)

    # 摘要
    best_recall = max(report["modes"], key=lambda m: m["recall_top5"])
    console.print(
        f"\n[bold]最佳召回率：{best_recall['mode']} "
        f"({best_recall['recall_top5'] * 100:.1f}%)[/bold]"
    )


if __name__ == "__main__":
    main()

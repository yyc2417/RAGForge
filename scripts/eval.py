"""RAGForge 自动化评估脚本

跑 4 套检索配置，计算召回率、幻觉率、延迟、token 消耗，输出对比表。

四套配置：
- baseline：Task 4 线性 RAG（vector-only，无 Agent）
- agent：Task 5 Agent 状态机（vector/bm25 策略，无混合）
- hybrid：Task 6 Agent + HybridRetriever（RRF 融合，无 Reranker）
- reranker：Task 6 Agent + HybridRetriever + Reranker（CrossEncoder 重排序）

运行：
    uv run python scripts/eval.py
输出：
    - 终端 rich 表格对比矩阵
    - reports/eval_report.json 完整报告
"""

import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

# 项目根目录加入 sys.path（脚本独立运行场景）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import build_agent_graph_from_pipeline, initial_state  # noqa: E402
from src.config import PROJECT_ROOT as CFG_ROOT, settings  # noqa: E402
from src.generation.llm_client import LLMClient  # noqa: E402
from src.ingestion import DocumentParser, EmbeddingService, TextChunker  # noqa: E402
from src.retrieval import BM25Retriever, HybridRetriever, Reranker, VectorStore  # noqa: E402
from src.utils.logger import logger  # noqa: E402
from src.utils.metrics import MetricsCollector  # noqa: E402

console = Console()
DATA_DIR = CFG_ROOT / settings.data_dir
DATASET_PATH = PROJECT_ROOT / "tests" / "eval_dataset.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "eval_report.json"
MODES = ["baseline", "agent", "hybrid", "reranker"]


def _build_base_artifacts() -> tuple[list, VectorStore, BM25Retriever]:
    """解析文档、切分、构建向量/BM25 索引（所有模式共享）。"""
    parser = DocumentParser()
    docs = parser.parse_directory(DATA_DIR)
    chunker = TextChunker()
    chunks = chunker.split(docs)
    embedder = EmbeddingService()
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


class Evaluator:
    """自动化评估器：跑 4 套配置，计算召回率/幻觉率/延迟/token。

    Args:
        dataset_path: 评估数据集 JSON 路径
    """

    def __init__(self, dataset_path: Path | str = DATASET_PATH) -> None:
        if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("sk-your"):
            console.print("[red]✗ DEEPSEEK_API_KEY 未配置（请在 .env 中填入有效 key）[/red]")
            console.print("  完整评估需真实调用 DeepSeek，不支持降级。配置后重试。")
            sys.exit(1)

        self.dataset = self._load_dataset(dataset_path)
        self._chunks, self._vector_store, self._bm25 = _build_base_artifacts()
        self._llm = LLMClient()
        self._reranker = Reranker()  # 单例，不可用时自动降级
        # 缓存各模式的执行管道
        self._pipelines: dict[str, object] = {}

    @staticmethod
    def _load_dataset(path: Path | str) -> list[dict]:
        """加载评估数据集。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        logger.info(f"[eval] 数据集加载：{len(data)} 个 QA 对")
        return data

    def _get_pipeline(self, mode: str):
        """按模式惰性构建并缓存执行管道。

        - baseline：返回 (vector_store, llm) 二元组，用 answer() 线性执行
        - agent/hybrid/reranker：返回编译后的 Agent graph
        """
        if mode in self._pipelines:
            return self._pipelines[mode]

        if mode == "baseline":
            # 线性 RAG：直接用 vector_store + llm
            pipeline = (self._vector_store, self._llm)
        else:
            hybrid = None
            reranker = None
            if mode in ("hybrid", "reranker"):
                hybrid = HybridRetriever(self._vector_store, self._bm25)
            if mode == "reranker":
                reranker = self._reranker
            pipeline = build_agent_graph_from_pipeline(
                vector_store=self._vector_store,
                bm25_retriever=self._bm25,
                llm_client=self._llm,
                hybrid_retriever=hybrid,
                reranker=reranker,
            )
        self._pipelines[mode] = pipeline
        return pipeline

    def _run_single(self, mode: str, item: dict) -> dict:
        """对单个 QA 项执行指定模式，返回结果字典。"""
        query = item["query"]
        expected_source = item.get("expected_source")
        expected_keywords = item.get("expected_answer_keywords", [])

        t0 = time.perf_counter()
        retrieved_sources: list[str] = []
        answer_text = ""
        is_faithful = True

        try:
            if mode == "baseline":
                vs, llm = self._get_pipeline(mode)
                docs = vs.search(query)
                retrieved_sources = [
                    _source_filename(getattr(d, "metadata", {}).get("source", ""))
                    for d in docs
                ]
                answer_text = llm.generate(query, docs)
                # baseline 无 verify 节点，用关键词命中粗判忠实度
                is_faithful = self._keyword_check(answer_text, expected_keywords)
            else:
                graph = self._get_pipeline(mode)
                final_state = graph.invoke(
                    initial_state(query),
                    config={"recursion_limit": 50},
                )
                retrieved_sources = _doc_sources(final_state)
                answer_text = final_state.get("answer", "")
                is_faithful = final_state.get("is_faithful", True)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[eval] {mode} 执行失败 query='{query}': {e}")
            answer_text = f"[ERROR] {e}"

        e2e_ms = (time.perf_counter() - t0) * 1000
        # 统一记录 e2e 延迟到 MetricsCollector（baseline/agent 都需要）
        MetricsCollector().record_e2e_latency(e2e_ms)

        # 召回判定：chitchat 无 expected_source，视为跳过
        hit = False
        if expected_source:
            hit = any(expected_source in s for s in retrieved_sources)

        return {
            "query": query,
            "query_type": item.get("query_type", "factual"),
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "hit": hit,
            "answer": answer_text,
            "is_faithful": is_faithful,
            "e2e_ms": e2e_ms,
        }

    @staticmethod
    def _keyword_check(answer: str, keywords: list[str]) -> bool:
        """关键词命中检查（baseline 幻觉判定的 fallback）。"""
        if not keywords:
            return True
        return any(kw in answer for kw in keywords)

    def evaluate_mode(self, mode: str) -> dict:
        """对单套模式跑全部数据集，返回汇总指标。"""
        console.print(f"\n[bold cyan]▶ 运行模式：{mode}[/bold cyan]")
        metrics = MetricsCollector()
        metrics.reset()

        results = []
        for i, item in enumerate(self.dataset, 1):
            console.print(f"  [{i}/{len(self.dataset)}] {item['query'][:30]}...", end=" ")
            r = self._run_single(mode, item)
            results.append(r)
            status = "✓" if r["hit"] or not r["expected_source"] else "✗"
            console.print(f"{status} {r['e2e_ms']:.0f}ms")

        # 汇总
        needs_retrieval = [r for r in results if r["expected_source"]]
        hits = sum(1 for r in needs_retrieval if r["hit"])
        recall = hits / len(needs_retrieval) if needs_retrieval else 0.0
        faithful = sum(1 for r in results if r["is_faithful"])
        faith_rate = faithful / len(results) if results else 0.0
        hallucination_rate = 1.0 - faith_rate

        summary = metrics.get_summary()
        e2e = summary.get("e2e_latency_ms", {})
        token = summary.get("token_usage", {})

        return {
            "mode": mode,
            "samples": len(results),
            "recall_top5": round(recall, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "faithfulness_rate": round(faith_rate, 4),
            "latency_p50_ms": e2e.get("p50", 0.0),
            "latency_p95_ms": e2e.get("p95", 0.0),
            "latency_p99_ms": e2e.get("p99", 0.0),
            "avg_token": round(token.get("avg_per_call", 0.0)) if token else 0,
            "total_token": token.get("total", 0) if token else 0,
            "details": results,
        }

    def run_all(self) -> dict:
        """跑全部 4 套模式，返回完整报告。"""
        report = {"modes": [], "targets": self._targets()}
        for mode in MODES:
            mode_result = self.evaluate_mode(mode)
            report["modes"].append(mode_result)
        return report

    @staticmethod
    def _targets() -> dict:
        """目标指标门槛（来自规划文档）。"""
        return {
            "recall_top5": 0.90,
            "hallucination_rate": 0.05,
            "latency_p99_ms": 2000,
            "avg_token": 800,
        }

    def print_report(self, report: dict) -> None:
        """用 rich 表格输出对比矩阵。"""
        targets = report["targets"]
        console.print()
        console.rule("[bold]RAGForge 评估报告 — 4 套检索配置对比[/bold]")

        table = Table(show_lines=True, header_style="bold magenta")
        table.add_column("配置", style="cyan", no_wrap=True)
        table.add_column("Top-5 召回率", justify="right")
        table.add_column("幻觉率", justify="right")
        table.add_column("P99 延迟(ms)", justify="right")
        table.add_column("平均 Token", justify="right")
        table.add_column("达标", justify="center")

        for m in report["modes"]:
            recall_ok = m["recall_top5"] >= targets["recall_top5"]
            halluc_ok = m["hallucination_rate"] <= targets["hallucination_rate"]
            p99_ok = m["latency_p99_ms"] <= targets["latency_p99_ms"] or m["latency_p99_ms"] == 0
            token_ok = m["avg_token"] <= targets["avg_token"] or m["avg_token"] == 0
            all_ok = recall_ok and halluc_ok and p99_ok and token_ok

            table.add_row(
                m["mode"],
                self._fmt(m["recall_top5"], recall_ok, percent=True),
                self._fmt(m["hallucination_rate"], halluc_ok, percent=True),
                self._fmt(m["latency_p99_ms"], p99_ok),
                self._fmt(m["avg_token"], token_ok),
                "[green]✓[/green]" if all_ok else "[yellow]△[/yellow]",
            )

        console.print(table)
        console.print(
            "[dim]图例：✓ 全指标达标 | △ 部分达标 | 绿色=达标 蓝色=未达标（参考目标见 README）[/dim]"
        )

    @staticmethod
    def _fmt(value: float, ok: bool, percent: bool = False) -> str:
        """格式化数值，达标绿色、未达标蓝色。"""
        if percent:
            text = f"{value * 100:.1f}%"
        elif value == 0:
            text = "N/A"
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
        console.print(f"\n[green]✓ 报告已保存：{path}[/green]")
        return path


def main() -> None:
    """评估主入口。

    用法：
        uv run python scripts/eval.py              # 跑全部 4 套
        uv run python scripts/eval.py --mode agent # 只跑指定套
        uv run python scripts/eval.py --resume     # 跳过报告里已完成的模式，只补未完成的
    """
    import argparse

    parser = argparse.ArgumentParser(description="RAGForge 自动化评估")
    parser.add_argument("--mode", choices=MODES, help="只跑指定模式（默认全部）")
    parser.add_argument("--resume", action="store_true",
                        help="增量模式：读取已有报告，跳过已完成的模式，只补未完成的")
    args = parser.parse_args()

    console.rule("[bold]RAGForge 自动化评估[/bold]")
    console.print(f"数据集：{DATASET_PATH}")

    evaluator = Evaluator()

    # 决定要跑哪些模式
    existing_report: dict | None = None
    if args.resume and REPORT_PATH.exists():
        try:
            existing_report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
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
        report = {"modes": merged_modes, "targets": Evaluator._targets()}
    else:
        report = {"modes": new_results, "targets": Evaluator._targets()}

    evaluator.print_report(report)
    evaluator.save_report(report)

    # 摘要
    best_recall = max(report["modes"], key=lambda m: m["recall_top5"])
    console.print(
        f"\n[bold]最佳召回率：{best_recall['mode']} "
        f"({best_recall['recall_top5'] * 100:.1f}%)[/bold]"
    )


if __name__ == "__main__":
    main()

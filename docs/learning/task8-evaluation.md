> ⚠️ **历史快照（2026-06）**：本文为学习笔记，记录写作当时的机制与数字，部分已被 2026-09 修复取代（双预算终止、LANGSMITH_* 环境变量、45 题数据集、token 化分块等）。现行设计以 `docs/modules/` 与 `docs/adr/`（尤其 006）为准；逐项修复记录见本地 `docs/planning/fix-log-2026-09.md`（私有文档，不入库）。
>
> **本文具体过时点**：数据集 15 题 → 45 题（30 可答 + 10 不可答 + 5 寒暄）；本文实测表为旧基线，最新数字见 README 评估表与 `reports/eval_report.json`。

# Task 8：评估系统与对比实验

## 阶段概述

本阶段实现 RAGForge 的自动化评估系统，跑 4 套检索配置（baseline / agent / hybrid / reranker）的对比实验，输出召回率、幻觉率、延迟、Token 消耗等多维度指标。Task 8 是项目的收尾阶段，验证前 7 个 Task 的实际效果，并为后续优化提供数据支撑。

**前置依赖**：Task 1-7（所有核心功能）  
**后续依赖**：无（项目收尾）

---

## 核心知识点

### 知识点 1：RAG 评估指标体系

RAGForge 的评估指标覆盖 4 个维度：

**1. Top-K 召回率（检索质量）**：
```python
needs_retrieval = [r for r in results if r["expected_source"]]
hits = sum(1 for r in needs_retrieval if r["hit"])
recall = hits / len(needs_retrieval) if needs_retrieval else 0.0
```

定义：检索到的 top-k 文档中，包含 `expected_source`（预期来源文件）的比例。例如 10 个需要检索的问题中，8 个命中了正确来源，召回率为 80%。

**2. 幻觉率（忠实度）**：
```python
faithful = sum(1 for r in results if r["is_faithful"])
faith_rate = faithful / len(results) if results else 0.0
hallucination_rate = 1.0 - faith_rate
```

定义：1 - faithfulness_rate。`is_faithful` 由 verify 节点（LLM 幻觉检测）判断，或 baseline 用关键词命中粗判。幻觉率越低越好。

**3. P50/P95/P99 延迟（响应速度）**：
```python
summary = metrics.get_summary()
e2e = summary.get("e2e_latency_ms", {})
# e2e = {"p50": 2219, "p95": 3237, "p99": 3799, "avg": 2329, "max": 3940}
```

定义：端到端延迟的百分位数。P50 是典型体验，P95 是大多数用户体验，P99 是最差体验（长尾延迟）。

**4. Token 消耗（成本）**：
```python
token = summary.get("token_usage", {})
# token = {"total_prompt": 5000, "total_completion": 3345, "total": 8345, "calls": 12, "avg_per_call": 695}
```

定义：LLM 调用的 token 总量和平均值。`avg_token` 反映单次请求的成本。

**目标门槛**（来自规划文档）：
- 召回率 ≥ 90%
- 幻觉率 < 5%
- P99 延迟 < 2000ms
- 平均 Token < 800

### 知识点 2：评估数据集设计

`tests/eval_dataset.json` 包含 15 个 QA 对，覆盖 4 种 query_type：

```json
[
  {
    "query": "Python 的装饰器怎么用？",
    "expected_answer_keywords": ["装饰器", "@", "函数"],
    "expected_source": "python_basics.md",
    "query_type": "factual",
    "needs_retrieval": true
  },
  {
    "query": "你好",
    "expected_answer_keywords": ["你好", "您", "帮助"],
    "expected_source": null,
    "query_type": "chitchat",
    "needs_retrieval": false
  },
  {
    "query": "对比 Python GIL 的多线程限制与机器学习中的并行训练需求，给出建议",
    "expected_answer_keywords": ["GIL", "并行", "训练"],
    "expected_source": "python_basics.md",
    "query_type": "complex",
    "needs_retrieval": true
  }
]
```

**4 种 query_type 覆盖**：
- **factual**（事实查询，6 条）：如"什么是过拟合？"、"Python 装饰器怎么用？"
- **reasoning**（推理查询，4 条）：如"为什么 Transformer 比 RNN 更适合长序列？"
- **chitchat**（寒暄，3 条）：如"你好"、"谢谢"、"再见"
- **complex**（复杂查询，2 条）：如"对比 GIL 限制与并行训练需求"

**`expected_source` 字段的设计**：
- 需要检索的问题：指定预期来源文件（如 `"python_basics.md"`），用于计算召回率
- 不需要检索的问题（chitchat）：`expected_source: null`，跳过召回率计算

**`expected_answer_keywords` 字段**：关键词列表，用于 baseline 模式的幻觉粗判（baseline 无 verify 节点，用关键词命中代替）。

**为什么 10-15 条 QA 就够**：
1. **覆盖性优先于数量**：15 条覆盖 4 种 query_type、2 个知识文件、多种难度，比 100 条同质化问题更有价值
2. **成本控制**：4 套配置 × 15 条 = 60 次 LLM 调用，约 10 分钟完成。100 条需要 40 分钟，成本和时间都过高
3. **快速迭代**：评估数据集小有助于快速跑完实验、发现问题、调整策略。生产环境建议扩到 30+ 条

### 知识点 3：4 套对比配置

评估脚本跑 4 套配置，逐步叠加复杂度：

```python
MODES = ["baseline", "agent", "hybrid", "reranker"]

def _get_pipeline(self, mode: str):
    if mode == "baseline":
        # 线性 RAG：vector-only，无 Agent
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
    return pipeline
```

**4 套配置对比**：

| 配置 | 检索方式 | Agent | 混合检索 | 重排序 |
|------|---------|-------|---------|--------|
| baseline | vector-only | ❌ | ❌ | ❌ |
| agent | vector/bm25 策略 | ✅ | ❌ | ❌ |
| hybrid | RRF 融合 | ✅ | ✅ | ❌ |
| reranker | RRF + CrossEncoder | ✅ | ✅ | ✅ |

**逐步叠加复杂度的设计**：
1. **baseline**：Task 4 线性 RAG，作为基准线
2. **agent**：加入 Task 5 状态机，验证自适应能力
3. **hybrid**：加入 Task 6 混合检索，验证 RRF 融合效果
4. **reranker**：加入 Task 6 重排序，验证 CrossEncoder 修正能力

**实测结果**（来自 `reports/eval_report.json`）：

| 配置 | 召回率 | 幻觉率 | P99 延迟 | 平均 Token |
|------|:------:|:-----:|:-------:|:---------:|
| baseline | **100%** | **0%** | 3.8s | 695 |
| agent | **100%** | 16.7% | 35.3s | 617 |
| hybrid | 70% | 16.7% | 58.8s | 641 |
| reranker | **100%** | 16.7% | 89.6s | 626 |

**关键洞察**：
1. **hybrid 召回率下降的原因**：小型知识库（仅 10 chunks）上 RRF 融合引入 BM25 噪声，稀释向量检索的高质量结果
2. **reranker 的修正效果**：CrossEncoder 重排序把 70% 拉回 100%，证明 reranker 能有效修正 RRF 的缺陷
3. **幻觉率 16.7% 的真实来源**：baseline 用关键词命中粗判（宽松，假阴性 0%）；agent 系用 LLM verify 严格判定（更真实）。两例幻觉均发生在 verify 后因达 `max_iterations=3` 被强制结束的边界场景
4. **延迟随复杂度递增**：baseline 单题 2-4s；agent 因 reformulate 循环升至 35s；reranker 额外叠加 CrossEncoder 推理

### 知识点 4：rich 终端输出

评估脚本用 `rich` 库输出美观的终端表格：

```python
from rich.console import Console
from rich.table import Table

console = Console()

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
    console.print("[dim]图例：✓ 全指标达标 | △ 部分达标 | 绿色=达标 蓝色=未达标[/dim]")

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
```

**Table API 的列定义**：
- `add_column("配置", style="cyan", no_wrap=True)`：列名、样式（青色）、禁止换行
- `justify="right"`：右对齐（数字列）
- `justify="center"`：居中对齐（状态列）

**行添加**：`table.add_row(...)` 接受任意数量的参数，每个参数对应一列。

**样式标记**：`[green]✓[/green]` 用 BBCode 风格标记颜色，`[bold]...[/bold]` 标记粗体，`[dim]...[/dim]` 标记暗淡。

**Console.rule**：`console.rule("[bold]标题[/bold]")` 输出带分隔线的标题，自动填充 `─` 字符。

### 知识点 5：增量评估模式

全量 4 套 × 15 题首次运行超过 10 分钟（agent 模式单题 15-35s），增加 `--mode` 和 `--resume` 参数支持增量评估：

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="RAGForge 自动化评估")
    parser.add_argument("--mode", choices=MODES, help="只跑指定模式（默认全部）")
    parser.add_argument("--resume", action="store_true",
                        help="增量模式：读取已有报告，跳过已完成的模式，只补未完成的")
    args = parser.parse_args()
    
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
        except Exception as e:
            console.print(f"[red]读取已有报告失败（{e}），从头开始[/red]")
            modes_to_run = [args.mode] if args.mode else MODES
    elif args.mode:
        modes_to_run = [args.mode]
    else:
        modes_to_run = MODES
```

**`--mode X`**：只跑指定模式（如 `--mode agent` 只跑 agent 模式）

**`--resume`**：读取已有 `reports/eval_report.json`，计算 `done_modes`，只跑未完成的模式

**合并逻辑**：增量模式下，新跑的结果与已有报告按 `MODES` 固定顺序合并：

```python
# 合并报告（增量模式下保留已有结果）
if existing_report:
    new_by_mode = {m["mode"]: m for m in new_results}
    merged_modes = []
    for mode in MODES:
        for m in existing_report["modes"]:
            if m["mode"] == mode and mode not in new_by_mode:
                merged_modes.append(m)
        if mode in new_by_mode:
            merged_modes.append(new_by_mode[mode])
    report = {"modes": merged_modes, "targets": Evaluator._targets()}
```

**使用场景**：
```bash
# 首次全量跑（超时）
uv run python scripts/eval.py

# 分 4 次增量跑（每次 < 10 分钟）
uv run python scripts/eval.py --mode baseline
uv run python scripts/eval.py --resume --mode agent
uv run python scripts/eval.py --resume --mode hybrid
uv run python scripts/eval.py --resume --mode reranker

# 查看完整报告
uv run python scripts/eval.py --resume
```

### 知识点 6：指标分析方法论

**如何解读对比实验结果**：

1. **召回率对比**：
   - baseline 100% vs hybrid 70%：说明 RRF 在小知识库上引入噪声
   - hybrid 70% vs reranker 100%：说明 CrossEncoder 重排序能修正 RRF 缺陷
   - 结论：小知识库优先用向量检索，大知识库用混合检索 + 重排序

2. **幻觉率对比**：
   - baseline 0% vs agent 16.7%：baseline 用关键词命中粗判（宽松），agent 用 LLM verify 严格判定（更真实）
   - 16.7% 发生在 verify 后因达 `max_iterations` 被强制结束的边界场景
   - 结论：verify 节点的严格判定更可靠，但需要调优 `max_iterations` 或 VERIFY_PROMPT

3. **延迟对比**：
   - baseline 3.8s vs agent 35.3s：Agent 状态机的多轮 LLM 调用（analyze/evaluate/reformulate/verify）显著增加延迟
   - agent 35.3s vs hybrid 58.8s vs reranker 89.6s：混合检索和重排序进一步叠加延迟
   - 结论：自适应质量与延迟是权衡关系，生产环境可通过调低 `max_iterations`、缓存、SSE 流式缓解

4. **Token 对比**：
   - baseline 695 vs agent 617：Agent 系平均 token 反而下降，因为 reformulate 用更短的诊断 prompt，且循环时复用上下文
   - 结论：Agent 的多轮调用不一定增加 token 消耗

**hybrid 召回率下降的原因分析**：
- 知识库只有 10 个 chunks（2 个 Markdown 文件切分后）
- BM25 在小库上关键词匹配不够精准（如"过拟合"查询匹配到包含"拟合"但不相关的文档）
- RRF 融合时，BM25 的低质量结果稀释了向量检索的高质量结果
- Reranker 能修正：CrossEncoder 对 (query, document) pair 做交叉注意力评分，过滤掉不相关文档

**reranker 的修正效果**：
- hybrid 70% → reranker 100%：CrossEncoder 重排序把被 RRF 错误排序的正确文档重新提权
- 证明 reranker 是混合检索的必要补强，尤其在知识库较小或 BM25 质量不稳定时

---

## 设计模式与架构决策

**对比实验模式**：4 套配置逐步叠加复杂度，控制变量（相同数据集、相同查询、相同 LLM），只变化检索策略，得出可靠结论。

**增量评估模式**：`--mode` 和 `--resume` 参数支持分次运行，避免全量超时。合并逻辑按固定顺序保证报告一致性。

**rich 终端输出**：Table API 输出美观的表格，样式标记（颜色/粗体）提升可读性。

**关键词粗判（baseline 幻觉检测）**：baseline 无 verify 节点，用 `expected_answer_keywords` 关键词命中粗判忠实度。这是简化设计，agent 系用 LLM verify 更严格。

---

## 关键代码解读

### eval.py：评估脚本

```python
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

# 项目根目录加入 sys.path（脚本独立运行场景）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import build_agent_graph_from_pipeline, initial_state
from src.config import PROJECT_ROOT as CFG_ROOT, settings
from src.generation.llm_client import LLMClient
from src.ingestion import DocumentParser, EmbeddingService, TextChunker
from src.retrieval import BM25Retriever, HybridRetriever, Reranker, VectorStore
from src.utils.logger import logger
from src.utils.metrics import MetricsCollector

console = Console()
DATA_DIR = CFG_ROOT / settings.data_dir
DATASET_PATH = PROJECT_ROOT / "tests" / "eval_dataset.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "eval_report.json"
MODES = ["baseline", "agent", "hybrid", "reranker"]
```

**设计意图**：`sys.path.insert(0, str(PROJECT_ROOT))` 确保脚本独立运行时能 import `src` 模块（无需设置 `PYTHONPATH`）。`PROJECT_ROOT` 用 `Path(__file__).resolve().parent.parent` 动态计算，确保路径正确。

```python
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
```

**设计意图**：所有模式共享相同的基础工件（文档、向量索引、BM25 索引），避免重复构建。`EmbeddingService` 单例确保模型只加载一次。

```python
class Evaluator:
    def __init__(self, dataset_path: Path | str = DATASET_PATH) -> None:
        if not settings.deepseek_api_key or settings.deepseek_api_key.startswith("sk-your"):
            console.print("[red]✗ DEEPSEEK_API_KEY 未配置（请在 .env 中填入有效 key）[/red]")
            console.print("  完整评估需真实调用 DeepSeek，不支持降级。配置后重试。")
            sys.exit(1)
        
        self.dataset = self._load_dataset(dataset_path)
        self._chunks, self._vector_store, self._bm25 = _build_base_artifacts()
        self._llm = LLMClient()
        self._reranker = Reranker()  # 单例，不可用时自动降级
        self._pipelines: dict[str, object] = {}
```

**设计意图**：构造函数检查 API key 是否有效，无效时提前退出（避免跑完才发现 key 错误）。`_pipelines` 缓存各模式的管道，惰性构建（首次调用 `_get_pipeline(mode)` 时构建）。

```python
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
            retrieved_sources = [_source_filename(getattr(d, "metadata", {}).get("source", "")) for d in docs]
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
    except Exception as e:
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
```

**设计意图**：`_run_single` 对单个 QA 项执行指定模式。baseline 直接调用 `vector_store.search()` + `llm.generate()`，用关键词命中粗判忠实度；agent 系调用 `graph.invoke()`，从 `final_state` 提取结果和 verify 节点的 `is_faithful`。`MetricsCollector().record_e2e_latency(e2e_ms)` 统一记录延迟（baseline 和 agent 都需要）。

---

## 踩坑记录

### 问题 1：baseline 延迟统计为 0

**问题现象**：eval.py 冒烟测试时，baseline 模式 `latency_p50/p99` 都是 0，但 `e2e_ms` 有值（1716/1238ms）。

**排查过程**：
1. 检查 `MetricsCollector` 发现 `e2e_latency_ms` 列表为空
2. 检查 `_run_single` 发现 baseline 直接调 `llm.generate()`，绕过了 `ask_agent()`
3. `ask_agent()` 中有 `metrics.record_e2e_latency(e2e)`，但 baseline 不调用它

**根因**：baseline 模式直接调 `llm.generate()`，绕过了 `ask_agent()`，所以 `record_e2e_latency()` 从未被调用。

**解决方案**：在 `_run_single()` 里统一加 `MetricsCollector().record_e2e_latency(e2e_ms)`，对 baseline 和 agent 模式都生效。

### 问题 2：全量评估首次超时

**问题现象**：`uv run python scripts/eval.py` 后台运行，10 分钟后在 reranker 模式附近超时（exit code 1），`reports/eval_report.json` 未生成。

**排查过程**：
1. 计算耗时：agent 模式单题 15-35s × 15 题 = 225-525s ≈ 4-9 分钟
2. 4 套配置总计约 20-40 分钟，远超 10 分钟超时限制

**根因**：全量 4 套 × 15 题，agent 系每题多轮 LLM 调用，总耗时远超 10 分钟。

**解决方案**：增加 `--mode` 和 `--resume` 参数，分 4 次运行（baseline 前台 ~30s，agent/hybrid/reranker 各后台 ~5 分钟），通过 JSON 报告累积结果。

### 问题 3：结构化输出解析偶发失败

**问题现象**：eval 日志中多次出现 `[llm] 结构化输出解析失败` 警告，如 `Expecting value: line 1 column 1 (char 0)`（空回复）。

**排查过程**：
1. 打印 LLM 原始回复发现有时返回空字符串
2. 检查 Prompt 发现 JSON 指令在末尾，LLM 可能未完整生成

**根因**：DeepSeek 不支持 `response_format=json_schema`，用"prompt 末尾追加 JSON 指令 + Pydantic 解析"的替代方案。LLM 偶尔不严格遵守 schema。

**影响**：**无**。每个 LLM 节点都有降级逻辑——`evaluate` 节点 catch 后回退规则层，`analyze` 节点降级到 `factual + needs_retrieval=True`。这是设计内的容错，不影响整体流程，只记录为 warning。

---

## 与其他模块的交互

**输入接口**：
- `build_agent_pipeline()`：从 `src/pipeline.py` 注入，构建完整 Agent 管道
- `initial_state(query)`：从 `src/agent/state.py` 注入，构造初始状态
- `VectorStore` / `BM25Retriever` / `HybridRetriever` / `Reranker`：从 Task 3/6 注入
- `LLMClient`：从 Task 4 注入
- `MetricsCollector`：从 Task 4 注入，记录性能指标
- `tests/eval_dataset.json`：评估数据集
- `settings.data_dir`：从 Task 1 配置读取数据目录

**输出接口**：
- `reports/eval_report.json`：完整评估报告（4 套配置的所有指标和详细结果）
- 终端 rich 表格：对比矩阵（召回率/幻觉率/延迟/Token/达标状态）

**衔接方式**：
- `_build_base_artifacts()` 调用 Task 2 的 parser / chunker / embedder
- `_get_pipeline(mode)` 调用 Task 3/6 的检索器和 Task 5 的 Agent 图
- `_run_single(mode, item)` 调用 `graph.invoke()` 或 `llm.generate()`
- `evaluate_mode(mode)` 调用 `MetricsCollector.get_summary()` 提取指标
- `print_report(report)` 用 rich 输出终端表格
- `save_report(report)` 导出 JSON 报告

---

## 本阶段收获总结

1. **RAG 评估需覆盖多维度指标**：召回率（检索质量）、幻觉率（忠实度）、延迟（响应速度）、Token（成本），单一指标无法全面评估
2. **评估数据集设计需覆盖多种 query_type**：factual / reasoning / chitchat / complex，10-15 条足够（覆盖性优先于数量）
3. **对比实验需控制变量**：相同数据集、相同查询、相同 LLM，只变化检索策略，才能得出可靠结论
4. **增量评估模式应对超时问题**：`--mode` 和 `--resume` 参数支持分次运行，合并逻辑按固定顺序保证一致性
5. **hybrid 召回率下降的原因是小知识库上 BM25 引入噪声**：reranker 能有效修正，证明 CrossEncoder 重排序的价值
6. **如实记录未达标指标比粉饰数据更重要**：hybrid 70% 召回率、16.7% 幻觉率是真实数据，README 中如实填入并分析原因
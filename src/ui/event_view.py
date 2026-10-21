"""astream_events 事件流蒸馏器：把原始事件字典流转成可视化视图状态

纯逻辑实现，不依赖 Streamlit——与 UI 框架解耦，可离线单测。

消费 graph.astream_events(version="v2") 产出的事件 dict：
- on_chain_start / on_chain_end（name ∈ 8 业务节点）：节点三态与执行次数
- 节点 output（on_chain_end 的 data.output）：诊断字段提取
- on_chain_start 的 data.input（已实测确认携带节点入参状态）：逐轮归集的策略与查询
- on_chat_model_stream：流式 token 缓冲——**必须过滤 langgraph_node == "generate"**，
  否则 analyze/evaluate 等结构化调用的 JSON 碎片会混入答案流（实测确认）
- on_chain_end 且 output 含 answer：最终状态捕获

逐轮归集（rounds）：诊断字段只保留终态，第二轮 evaluate 会覆盖第一轮的
failure_mode / diagnosis_reason，自愈因果链因此丢失。本模块以 retrieve 的
start/end 为轮次边界，把每轮的查询、策略、命中、诊断结论、后续自愈动作
各自归档为一条 Round，供界面呈现完整时间线。
"""

from dataclasses import dataclass, field
from typing import Any

# 8 个业务节点（7 节点主链路 + clarify 终止分支）
NODES = ["analyze", "clarify", "decide", "retrieve", "evaluate",
         "reformulate", "generate", "verify"]
_NODE_SET = set(NODES)

# 节点状态三态
PENDING = "pending"
RUNNING = "running"
DONE = "done"

# 从各节点 output 中提取展示的诊断字段（诊断面板逐项刷新）
_DIAGNOSTIC_FIELDS = (
    "query_type",
    "needs_retrieval",
    "needs_clarification",
    "ambiguity_type",
    "clarification_question",
    "retrieval_strategy",
    "failure_mode",
    "suggested_action",
    "diagnosis_reason",
    "reformulated_query",
    "rewrite_effective",
    "iteration_count",
    "verify_failures",
    "is_faithful",
    "verification_reason",
)

# 自愈动作节点（界面上高亮提示）
_SELF_HEAL_NODES = {"reformulate"}

# 事件引起的可见变更级别：界面据此选择重画范围（流式 token 仅刷答案区）
LEVEL_FULL = "full"
LEVEL_ANSWER = "answer"

# 固定 hybrid 的分数语义（详见 ADR-006）：RRF 分只能在本轮内比较排序
SCORE_SEMANTICS = {
    "hybrid": "RRF 融合分上限约 2/(rrf_k+1)，只能在本轮内比较排序",
    "none": "未检索",
}


@dataclass
class Round:
    """一次真实检索的完整归档（自愈时间线的一行）。"""

    index: int
    query: str = ""
    strategy: str = "unknown"
    hits: int = 0
    scores: list[float] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    failure_mode: str | None = None
    suggested_action: str | None = None
    diagnosis_reason: str = ""
    # 本轮之后依次触发的自愈动作：[("reformulate", 改写后的查询)]
    # 用列表而非单字段：幻觉重试路径可能在同一轮上追加第二个动作，覆盖会丢信息
    actions: list[tuple[str, str]] = field(default_factory=list)
    rewrite_effective: bool | None = None

    @property
    def top_score(self) -> float:
        """本轮最高分（无命中为 0.0）。"""
        return max(self.scores, default=0.0)

    @property
    def healed(self) -> bool:
        """本轮是否触发了自愈动作（时间线高亮依据）。"""
        return bool(self.actions)

    @property
    def score_semantics(self) -> str:
        """本轮分数的语义说明（避免跨策略误读）。"""
        return SCORE_SEMANTICS.get(self.strategy, "分数语义未知")


class EventView:
    """消费 astream_events 事件，维护可视化所需的全部视图状态。"""

    def __init__(self) -> None:
        self.node_status: dict[str, str] = {name: PENDING for name in NODES}
        self.node_runs: dict[str, int] = {name: 0 for name in NODES}
        self.diagnostic: dict[str, Any] = {}
        self.tokens: list[str] = []
        self.sources: list[dict[str, Any]] = []  # [{content, source, score}]
        self.rounds: list[Round] = []  # 逐轮时间线：每轮检索一条归档
        self.final_state: dict[str, Any] = {}
        self.error: str | None = None
        self._pending_round: Round | None = None

    # ── 事件分发 ───────────────────────────────────────────────────
    def consume(self, event: dict) -> str | None:
        """处理单个 astream_events 事件（容错：异常事件记入 self.error）。

        Returns:
            可见变更级别 LEVEL_FULL / LEVEL_ANSWER；None 表示本事件无可见变化。
            界面侧靠它把「每个 token 重画全页」降为「节点变更重画、token 只刷答案」。
        """
        kind = event.get("event")
        name = event.get("name", "")

        if kind == "on_chain_start" and name in _NODE_SET:
            self.node_runs[name] += 1
            self.node_status[name] = RUNNING
            if name == "retrieve":
                self._open_round(event.get("data", {}).get("input"))
            return LEVEL_FULL

        if kind == "on_chain_end":
            output = event.get("data", {}).get("output")
            if not isinstance(output, dict):
                return None
            if name in _NODE_SET:
                self.node_status[name] = DONE
                self._absorb_output(output)
                self._update_rounds(name, output)
                return LEVEL_FULL
            if "answer" in output:
                # 根图最终输出：按「含 answer 键」判定而非硬编码链路名
                # （根图最后完成，覆盖 generate 节点的部分输出）
                self.final_state = output
                return LEVEL_FULL
            return None

        if kind == "on_chat_model_stream":
            # 只采纳 generate 节点的 token：结构化节点的 JSON 碎片必须排除
            node = event.get("metadata", {}).get("langgraph_node")
            if node != "generate":
                return None
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", "") or ""
            if not content:
                return None
            self.tokens.append(content)
            return LEVEL_ANSWER

        if kind == "on_chain_error":
            err = event.get("data", {}).get("error")
            if not err:
                return None
            self.error = str(err)
            return LEVEL_FULL

        return None

    # ── 派生视图 ───────────────────────────────────────────────────
    @property
    def answer(self) -> str:
        """当前答案文本：优先流式累积，兜底最终状态。"""
        if self.tokens:
            return "".join(self.tokens)
        return str(self.final_state.get("answer", ""))

    @property
    def self_heal_triggered(self) -> bool:
        """是否触发过自愈动作（reformulate 至少执行一次）。"""
        return any(self.node_runs[n] > 0 for n in _SELF_HEAL_NODES)

    @property
    def total_llm_rounds(self) -> int:
        """LLM 相关节点的最大执行次数（自愈循环轮次的直观展示）。"""
        return max((self.node_runs[n] for n in NODES), default=0)

    def summary(self) -> dict[str, Any]:
        """运行摘要面板所需字段。"""
        return {
            "answer": self.answer,
            "is_faithful": self.final_state.get("is_faithful"),
            "verification_reason": self.final_state.get("verification_reason", ""),
            "iteration_count": self.final_state.get("iteration_count", len(self.rounds)),
            "verify_failures": self.final_state.get("verify_failures", 0),
            "self_heal_triggered": self.self_heal_triggered,
            "round_count": len(self.rounds),
            "max_iterations": self.final_state.get("max_iterations"),
            "error": self.error,
        }

    # ── 内部 ───────────────────────────────────────────────────────
    def _absorb_output(self, output: dict) -> None:
        """从业务节点 output 提取诊断字段与来源表格数据。"""
        for key in _DIAGNOSTIC_FIELDS:
            if key in output:
                self.diagnostic[key] = output[key]
        # retrieve 节点输出携带 documents + retrieval_scores → 来源表格
        if "documents" in output:
            self.sources = self._extract_sources(output)

    def _open_round(self, input_state: Any) -> None:
        """retrieve 启动时按节点入参状态开一轮（入参缺失时回退到已捕获的诊断值）。"""
        state = input_state if isinstance(input_state, dict) else {}
        strategy = (
            state.get("retrieval_strategy")
            or self.diagnostic.get("retrieval_strategy")
            or "unknown"
        )
        query = (
            state.get("reformulated_query")
            or state.get("query")
            or self.diagnostic.get("reformulated_query")
            or ""
        )
        self._pending_round = Round(
            index=len(self.rounds) + 1, query=str(query), strategy=str(strategy)
        )

    def _update_rounds(self, name: str, output: dict) -> None:
        """把 retrieve / evaluate / reformulate 的产出归到对应轮次。"""
        if name == "retrieve":
            if self._pending_round is None:
                return
            rnd = self._pending_round
            # iteration_count 是检索次数的唯一递增点，作为轮号权威来源
            rnd.index = int(output.get("iteration_count") or rnd.index)
            rnd.sources = list(self.sources)
            rnd.hits = len(rnd.sources)
            rnd.scores = [float(s) for s in output.get("retrieval_scores") or []]
            self.rounds.append(rnd)
            self._pending_round = None
            return

        if not self.rounds:
            # 寒暄路径无检索轮次，不归档
            return
        rnd = self.rounds[-1]
        if name == "evaluate" and rnd.failure_mode is None:
            rnd.failure_mode = output.get("failure_mode")
            rnd.suggested_action = output.get("suggested_action")
            rnd.diagnosis_reason = str(output.get("diagnosis_reason", ""))
        elif name == "reformulate":
            rnd.rewrite_effective = output.get("rewrite_effective")
            rnd.actions.append(
                ("reformulate", str(output.get("reformulated_query", "")))
            )

    @staticmethod
    def _extract_sources(output: dict) -> list[dict[str, Any]]:
        """把 retrieve 节点输出的 documents/scores 转成来源数据。

        content 保留全文：界面默认折叠、展开时才阅读，截断只会把句子切在半词上。
        """
        docs = output.get("documents", []) or []
        scores = output.get("retrieval_scores") or []
        rows: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            if not hasattr(doc, "page_content"):
                continue
            meta = getattr(doc, "metadata", {})
            score = scores[i] if i < len(scores) else 0.0
            rows.append({
                "content": doc.page_content,
                "source": meta.get("source", "unknown"),
                "score": round(float(score), 4),
            })
        return rows

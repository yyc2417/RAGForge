"""astream_events 事件流蒸馏器：把原始事件字典流转成可视化视图状态

纯逻辑实现，不依赖 Streamlit——与 UI 框架解耦，可离线单测。

消费 graph.astream_events(version="v2") 产出的事件 dict：
- on_chain_start / on_chain_end（name ∈ 8 业务节点）：节点三态与执行次数
- 节点 output（on_chain_end 的 data.output）：诊断字段提取
- on_chat_model_stream：流式 token 缓冲——**必须过滤 langgraph_node == "generate"**，
  否则 analyze/evaluate 等结构化调用的 JSON 碎片会混入答案流（实测确认）
- on_chain_end 且 output 含 answer：最终状态捕获
"""

from typing import Any

# 8 个业务节点（顺序即主流程拓扑）
NODES = ["analyze", "decide", "retrieve", "evaluate",
         "reformulate", "switch_strategy", "generate", "verify"]
_NODE_SET = set(NODES)

# 节点状态三态
PENDING = "pending"
RUNNING = "running"
DONE = "done"

# 从各节点 output 中提取展示的诊断字段（诊断面板逐项刷新）
_DIAGNOSTIC_FIELDS = (
    "query_type",
    "needs_retrieval",
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
_SELF_HEAL_NODES = {"reformulate", "switch_strategy"}


class EventView:
    """消费 astream_events 事件，维护可视化所需的全部视图状态。"""

    def __init__(self) -> None:
        self.node_status: dict[str, str] = {name: PENDING for name in NODES}
        self.node_runs: dict[str, int] = {name: 0 for name in NODES}
        self.diagnostic: dict[str, Any] = {}
        self.tokens: list[str] = []
        self.sources: list[dict[str, Any]] = []  # [{content, source, score}]
        self.final_state: dict[str, Any] = {}
        self.error: str | None = None

    # ── 事件分发 ───────────────────────────────────────────────────
    def consume(self, event: dict) -> None:
        """处理单个 astream_events 事件（容错：异常事件记入 self.error）。"""
        kind = event.get("event")
        name = event.get("name", "")

        if kind == "on_chain_start" and name in _NODE_SET:
            self.node_runs[name] += 1
            self.node_status[name] = RUNNING

        elif kind == "on_chain_end":
            output = event.get("data", {}).get("output")
            if not isinstance(output, dict):
                return
            if name in _NODE_SET:
                self.node_status[name] = DONE
                self._absorb_output(output)
            elif "answer" in output:
                # 根图最终输出：按「含 answer 键」判定而非硬编码链路名
                # （根图最后完成，覆盖 generate 节点的部分输出）
                self.final_state = output

        elif kind == "on_chat_model_stream":
            # 只采纳 generate 节点的 token：结构化节点的 JSON 碎片必须排除
            node = event.get("metadata", {}).get("langgraph_node")
            if node != "generate":
                return
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", "") or ""
            if content:
                self.tokens.append(content)

        elif kind == "on_chain_error":
            err = event.get("data", {}).get("error")
            self.error = str(err) if err else self.error

    # ── 派生视图 ───────────────────────────────────────────────────
    @property
    def answer(self) -> str:
        """当前答案文本：优先流式累积，兜底最终状态。"""
        if self.tokens:
            return "".join(self.tokens)
        return str(self.final_state.get("answer", ""))

    @property
    def self_heal_triggered(self) -> bool:
        """是否触发过自愈动作（reformulate / switch_strategy 至少执行一次）。"""
        return any(self.node_runs[n] > 0 for n in _SELF_HEAL_NODES)

    @property
    def total_llm_rounds(self) -> int:
        """LLM 相关节点的最大执行次数（自愈循环轮次的直观展示）。"""
        return max((self.node_runs[n] for n in NODES), default=0)

    def summary(self) -> dict[str, Any]:
        """收尾指标面板所需字段。"""
        return {
            "answer": self.answer,
            "is_faithful": self.final_state.get("is_faithful"),
            "verification_reason": self.final_state.get("verification_reason", ""),
            "iteration_count": self.final_state.get("iteration_count", 0),
            "verify_failures": self.final_state.get("verify_failures", 0),
            "self_heal_triggered": self.self_heal_triggered,
            "error": self.error,
        }

    # ── 内部 ───────────────────────────────────────────────────────
    def _absorb_output(self, output: dict) -> None:
        """从业务节点 output 提取诊断字段与来源表格数据。"""
        for field in _DIAGNOSTIC_FIELDS:
            if field in output:
                self.diagnostic[field] = output[field]
        # retrieve 节点输出携带 documents + retrieval_scores → 来源表格
        if "documents" in output:
            self.sources = self._extract_sources(output)

    @staticmethod
    def _extract_sources(output: dict) -> list[dict[str, Any]]:
        """把 retrieve 节点输出的 documents/scores 转成来源表格数据。"""
        docs = output.get("documents", []) or []
        scores = output.get("retrieval_scores") or []
        rows: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            if not hasattr(doc, "page_content"):
                continue
            meta = getattr(doc, "metadata", {})
            score = scores[i] if i < len(scores) else 0.0
            rows.append({
                "content": doc.page_content[:200],
                "source": meta.get("source", "unknown"),
                "score": round(float(score), 4),
            })
        return rows

"""Agent 决策节点集合

7 个节点主链路，统一签名：node(state, *, llm, prompts, **deps) -> dict（返回部分状态更新）。
LLM 节点用结构化输出 + 本地 JSON 校验，确保输出可靠。

节点流向（固定 hybrid）：
  analyze → decide ─┬(none)→ generate → verify
                    └(hybrid)→ retrieve → evaluate ─┬(proceed)→ generate → verify
                                                     └(reformulate)→ reformulate → retrieve

拓扑口径：旧 8 节点含 switch_strategy；本模块为固定 hybrid 后的 7 节点主链路。
"""

from src.agent.nodes.analyze import analyze_node
from src.agent.nodes.decide import decide_node
from src.agent.nodes.evaluate import evaluate_node
from src.agent.nodes.generate import generate_node
from src.agent.nodes.reformulate import reformulate_node
from src.agent.nodes.retrieve import retrieve_node
from src.agent.nodes.verify import verify_node

__all__ = [
    "analyze_node",
    "decide_node",
    "retrieve_node",
    "evaluate_node",
    "reformulate_node",
    "generate_node",
    "verify_node",
]

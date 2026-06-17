"""Agent 决策节点集合

8 个节点，统一签名：node(state, *, llm, prompts, **deps) -> dict（返回部分状态更新）
LLM 节点用 with_structured_output 确保结构化输出可靠。

节点流向：
  analyze → decide ─┬(none)→ generate → verify
                    └(else)→ retrieve → evaluate ─┬(proceed)→ generate → verify
                                                  ├(reformulate)→ reformulate → retrieve
                                                  └(switch_strategy)→ switch_strategy → retrieve
"""

from src.agent.nodes.analyze import analyze_node
from src.agent.nodes.decide import decide_node
from src.agent.nodes.evaluate import evaluate_node
from src.agent.nodes.generate import generate_node
from src.agent.nodes.reformulate import reformulate_node
from src.agent.nodes.retrieve import retrieve_node
from src.agent.nodes.switch_strategy import switch_strategy_node
from src.agent.nodes.verify import verify_node

__all__ = [
    "analyze_node",
    "decide_node",
    "retrieve_node",
    "evaluate_node",
    "reformulate_node",
    "switch_strategy_node",
    "generate_node",
    "verify_node",
]

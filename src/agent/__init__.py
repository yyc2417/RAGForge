"""Agent 决策层 — LangGraph 状态机驱动

核心：8 节点状态机（analyze→decide→retrieve→evaluate→[reformulate/switch]→generate→verify）
含诊断式评估（failure_mode 驱动策略切换/查询改写）与幻觉检测闭环。
"""

from src.agent.graph import build_agent_graph, build_agent_graph_from_pipeline
from src.agent.state import AgentState, initial_state

__all__ = ["build_agent_graph", "build_agent_graph_from_pipeline", "AgentState", "initial_state"]

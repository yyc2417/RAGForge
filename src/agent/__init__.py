"""Agent 决策层 — LangGraph 状态机驱动

核心：8 节点状态机（analyze→decide→retrieve→evaluate→[reformulate/switch]→generate→verify）
含诊断式评估（failure_mode 驱动策略切换/查询改写）与幻觉检测闭环。

build_agent_graph* 惰性导入（PEP 562）：graph → nodes → generation.schemas
→ src.agent.state 与包 __init__ 的急切导入构成循环（schemas 依赖 state 的
枚举类型），任何先触达 generation.schemas 的入口都会炸出部分初始化错误，
因此 graph工厂由 __getattr__ 按需转发。
"""

from typing import Any

from src.agent.state import AgentState, initial_state

__all__ = ["build_agent_graph", "build_agent_graph_from_pipeline", "AgentState", "initial_state"]

_GRAPH_EXPORTS = ("build_agent_graph", "build_agent_graph_from_pipeline")


def __getattr__(name: str) -> Any:
    """按需转发 graph 工厂，规避与 generation.schemas 的循环导入。"""
    if name in _GRAPH_EXPORTS:
        from src.agent.graph import build_agent_graph, build_agent_graph_from_pipeline

        return {
            "build_agent_graph": build_agent_graph,
            "build_agent_graph_from_pipeline": build_agent_graph_from_pipeline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

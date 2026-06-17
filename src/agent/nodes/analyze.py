"""查询分析节点：用 LLM 分析查询意图与是否需要检索

输入状态：query
输出状态：query_type, needs_retrieval, messages
"""

from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import AnalyzeResult
from src.utils.logger import logger


def analyze_node(
    state: AgentState,
    *,
    llm: LLMClient,
    prompts: PromptManager,
) -> dict:
    """分析查询意图，判断 query_type 和 needs_retrieval。

    Args:
        state: 全局状态（读 query）
        llm: LLM 客户端
        prompts: Prompt 管理器

    Returns:
        部分状态更新：query_type, needs_retrieval, messages
    """
    query = state["query"]
    logger.info(f"[analyze] 分析查询：{query}")

    try:
        result: AnalyzeResult = llm.invoke_structured(
            prompts.ANALYZE_PROMPT, AnalyzeResult, query=query
        )
        query_type = result.query_type
        needs_retrieval = result.needs_retrieval
    except Exception as e:  # noqa: BLE001 - LLM 失败时降级为需要检索的事实查询
        logger.warning(f"[analyze] LLM 分析失败，降级为 factual+需检索：{e}")
        query_type = "factual"
        needs_retrieval = True

    msg = f"analyze: query_type={query_type}, needs_retrieval={needs_retrieval}"
    logger.info(f"[analyze] → {msg}")
    return {
        "query_type": query_type,
        "needs_retrieval": needs_retrieval,
        "messages": [msg],
    }

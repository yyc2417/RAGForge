"""查询改写节点：根据 failure_mode 用对应策略改写查询

输入状态：query, failure_mode, documents
输出状态：reformulated_query, rewrite_strategy, iteration_count, messages

改写策略选择（规则提示 LLM）：
- irrelevant → specify（具体化，把模糊查询变明确）
- low_recall → synonym_replace（同义替换，换关键词）
（LLM 最终决定 rewrite_strategy 字段）
"""

from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.generation.schemas import ReformulateResult
from src.utils.logger import logger


def reformulate_node(
    state: AgentState,
    *,
    llm: LLMClient,
    prompts: PromptManager,
) -> dict:
    """改写查询并递增 iteration_count。

    Args:
        state: 全局状态（读 query, failure_mode, documents）
        llm: LLM 客户端
        prompts: Prompt 管理器

    Returns:
        部分状态更新：reformulated_query, rewrite_strategy, iteration_count, messages
    """
    query = state["query"]
    failure_mode = state.get("failure_mode", "irrelevant")
    documents = state.get("documents", [])

    # 根据失败模式给 LLM 改写策略提示
    if failure_mode == "irrelevant":
        rewrite_hint = "specify（具体化：把模糊指代补全为明确术语）"
    elif failure_mode == "low_recall":
        rewrite_hint = "synonym_replace（同义替换：换用更常见的同义关键词）"
    else:
        rewrite_hint = "generalize（泛化）"

    context = _format_context(documents)

    try:
        result: ReformulateResult = llm.invoke_structured(
            prompts.REFORMULATE_PROMPT, ReformulateResult,
            query=query,
            failure_mode=failure_mode,
            rewrite_hint=rewrite_hint,
            context=context,
        )
        new_query = result.reformulated_query or query
        rewrite_strategy = result.rewrite_strategy
    except Exception as e:  # noqa: BLE001 - LLM 失败时简单兜底
        logger.warning(f"[reformulate] LLM 改写失败，保留原查询：{e}")
        new_query = query
        rewrite_strategy = "specify"

    # 递增迭代计数（防死循环的关键状态）
    iteration_count = state.get("iteration_count", 0) + 1

    msg = (
        f"reformulate: '{query}' → '{new_query}' "
        f"(strategy={rewrite_strategy}, iter={iteration_count})"
    )
    logger.info(f"[reformulate] {msg}")
    return {
        "reformulated_query": new_query,
        "rewrite_strategy": rewrite_strategy,
        "iteration_count": iteration_count,
        "messages": [msg],
    }


def _format_context(documents) -> str:
    """把检索文档片段格式化为 prompt 文本（供改写参考）。"""
    if not documents:
        return "（无参考文档）"
    snippets = [doc.page_content[:120].replace("\n", " ") for doc in documents[:3]]
    return "\n".join(f"- {s}..." for s in snippets)

"""答案生成节点：调用 LLMClient 基于 context 生成答案

输入状态：query, documents
输出状态：answer, messages
"""

import time

from src.agent.state import AgentState
from src.generation.llm_client import LLMClient
from src.utils.metrics import MetricsCollector
from src.utils.logger import logger


def generate_node(
    state: AgentState,
    *,
    llm: LLMClient,
    prompts=None,
) -> dict:
    """基于检索上下文生成回答。

    Args:
        state: 全局状态（读 query, documents）
        llm: LLM 客户端
        prompts: Prompt 管理器（generate 复用 LLMClient 内置 QA_PROMPT，此参数保留以对齐节点签名）

    Returns:
        部分状态更新：answer, messages
    """
    query = state["query"]
    documents = state.get("documents", [])

    t0 = time.perf_counter()
    answer_text = llm.generate(query, documents)
    latency_ms = (time.perf_counter() - t0) * 1000

    # 端到端延迟在最终汇总处记录，这里只记录生成耗时（不入指标，避免与 e2e 重复）
    msg = f"generate: answer_len={len(answer_text)}, latency={latency_ms:.0f}ms"
    logger.info(f"[generate] {msg}")
    return {
        "answer": answer_text,
        "messages": [msg],
    }

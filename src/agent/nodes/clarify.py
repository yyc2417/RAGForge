"""clarify 终止节点：返回澄清问题并结束本轮

输入状态：needs_clarification, ambiguity_type, clarification_question
输出状态：answer（空）, clarification_question, messages

该节点不检索、不生成；下一轮用户回答后由会话层带历史重新进图。
"""

from src.agent.state import AgentState
from src.utils.logger import logger


def clarify_node(state: AgentState) -> dict:
    """把 analyze 产出的澄清问题作为本轮唯一输出。"""
    question = str(state.get("clarification_question") or "").strip()
    if not question:
        question = "能否补充更具体的信息？例如你指的是哪个对象或哪个场景。"
    ambiguity_type = state.get("ambiguity_type")
    msg = f"clarify: ambiguity_type={ambiguity_type}, question='{question}'"
    logger.info(f"[clarify] {msg}")
    return {
        # answer 留空：澄清问题不是最终回答，避免 verify / 评估误判
        "answer": "",
        "clarification_question": question,
        "messages": [msg],
    }

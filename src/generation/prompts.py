"""Prompt 模板集中管理

Task 4：实现 QA_PROMPT（问答）。
Task 5 将在此补充 ANALYZE_PROMPT / EVALUATE_PROMPT / REFORMULATE_PROMPT / VERIFY_PROMPT。
"""

from langchain_core.prompts import ChatPromptTemplate


class PromptManager:
    """Prompt 模板集中管理，供 LLMClient 与 Agent 节点复用。"""

    # ── Task 4：基础问答 Prompt ────────────────────────────────────
    QA_PROMPT = ChatPromptTemplate.from_template(
        """你是一个知识库问答助手。请根据以下检索到的文档内容回答用户的问题。

回答规则：
1. 优先使用检索到的文档内容来组织回答
2. 如果文档中包含相关信息，即使需要推理或总结，也应尽力回答
3. 只有当文档完全不包含任何相关信息时，才说明无法从知识库中找到答案
4. 回答时引用文档中的关键信息，但不要逐字照搬
5. 不要编造文档中没有的具体数据或事实

检索到的文档内容：
{context}

用户问题：{input}

请用中文回答："""
    )

    # 以下 Prompt 在 Task 5 补充：
    # ANALYZE_PROMPT / EVALUATE_PROMPT / REFORMULATE_PROMPT / VERIFY_PROMPT
    ANALYZE_PROMPT = ChatPromptTemplate.from_template(
        """你是一个查询意图分析器。请分析用户查询的类型和是否需要检索知识库。

查询类型说明：
- factual：事实性知识查询（如"什么是过拟合""Python 装饰器怎么用"）
- reasoning：需要推理或多步分析的查询
- chitchat：寒暄、打招呼、闲聊（如"你好""谢谢"）——此类不需要检索
- complex：复杂查询，可能需要多角度信息

用户查询：{query}

请输出 query_type 和 needs_retrieval（寒暄类 needs_retrieval=False）。"""
    )

    EVALUATE_PROMPT = ChatPromptTemplate.from_template(
        """你是一个检索质量诊断器。请诊断以下检索结果是否能支撑回答用户问题。

用户查询：{query}
检索到的文档（含相关性分数，越高越相关）：
{retrieved_docs}

诊断维度（三选一）：
- sufficient：检索结果与问题相关且信息充足，可 proceed 直接生成
- low_recall：相关性分数普遍偏低（如 < 0.3），召回不足，建议 switch_strategy 切换检索策略
- irrelevant：检索结果与问题无关（语义不匹配），建议 reformulate 改写查询

请输出 failure_mode、suggested_action 和 reason。"""
    )

    REFORMULATE_PROMPT = ChatPromptTemplate.from_template(
        """你是一个查询改写器。当前检索质量不佳，需要改写查询以提升召回。

用户原始查询：{query}
诊断的失败原因：{failure_mode}
（可选）改写策略建议：{rewrite_hint}
（可选）当前检索到的文档片段（供参考）：
{context}

改写策略：
- specify：将模糊查询具体化（"那个机器学习的东西" → "机器学习中的过拟合现象"）
- generalize：将过窄查询泛化
- synonym_replace：用同义词替换关键词

请输出改写后的 reformulated_query 和所用的 rewrite_strategy。改写后的查询应保留中文。"""
    )

    VERIFY_PROMPT = ChatPromptTemplate.from_template(
        """你是一个幻觉检测器。请判断以下回答是否忠实于检索到的文档内容。

用户查询：{query}
生成的回答：
{answer}

检索到的文档（回答应基于这些内容）：
{context}

判断标准：
- is_faithful=True：回答的核心信息可由文档内容推导或直接得出，允许合理的总结、归纳和推理
- is_faithful=False：回答包含文档中完全没有提及且无法合理推导的事实性信息（幻觉）

注意：
- 如果回答是对文档内容的合理总结或推理，应判为忠实
- 只有当回答编造了具体的事实、数据或概念时才判为幻觉
- 礼貌性回复（如问候、感谢）不涉及事实声明，应判为忠实

请输出 is_faithful 和 reason。"""
    )

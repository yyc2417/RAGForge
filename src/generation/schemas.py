"""结构化输出 Pydantic schema

配合 LLMClient.invoke_structured(prompt, schema) 使用：
prompt 引导 LLM 输出 JSON → 手工解析 → 本模块 Pydantic 校验
（DeepSeek 不支持 response_format=json_schema，见 ADR-003）。

枚举字段经 BeforeValidator 归一化（strip/lower/常见中文变体映射），
LLM 输出 "Factual"/"低召回" 等变体不再直接落入降级分支。
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from src.agent.state import (
    AmbiguityType,
    FailureMode,
    QueryType,
    RewriteStrategy,
    SuggestedAction,
)

# LLM 输出枚举值的常见变体 → 标准值映射（小写比较）
_ENUM_ALIASES: dict[str, str] = {
    "事实": "factual",
    "事实查询": "factual",
    "推理": "reasoning",
    "寒暄": "chitchat",
    "闲聊": "chitchat",
    "复杂": "complex",
    "复杂查询": "complex",
    "fact": "factual",
    "factoid": "factual",
    "question": "factual",
    "query": "factual",
    "reasoning_query": "reasoning",
    "greeting": "chitchat",
    "chat": "chitchat",
    "complex_query": "complex",
    "缺槽位": "missing_slot",
    "缺失槽位": "missing_slot",
    "缺少槽位": "missing_slot",
    "缺少实体": "missing_slot",
    "实体缺失": "missing_slot",
    "指代不明": "missing_slot",
    "缺少指代": "missing_slot",
    "missing_entity": "missing_slot",
    "missing_reference": "missing_slot",
    "missing_slot_value": "missing_slot",
    "missing_key_slot": "missing_slot",
    "missing_information": "missing_slot",
    "missing_detail": "missing_slot",
    "missing_context": "missing_slot",
    "underspecified": "missing_slot",
    "ambiguous_reference": "missing_slot",
    "多解释": "multiple_interpretations",
    "多种解释": "multiple_interpretations",
    "歧义": "multiple_interpretations",
    "multiple_interpretation": "multiple_interpretations",
    "ambiguous": "multiple_interpretations",
    "ambiguity": "multiple_interpretations",
    "约束冲突": "conflicting_constraints",
    "constraint_conflict": "conflicting_constraints",
    "conflicting_constraint": "conflicting_constraints",
    "conflict": "conflicting_constraints",
    "低召回": "low_recall",
    "召回不足": "low_recall",
    "insufficient": "low_recall",
    "not_enough": "low_recall",
    "不相关": "irrelevant",
    "无关": "irrelevant",
    "irrelevant_results": "irrelevant",
    "充足": "sufficient",
    "足够": "sufficient",
    "adequate": "sufficient",
    "改写": "reformulate",
    "改写查询": "reformulate",
    "rewrite": "reformulate",
    "retry": "reformulate",
    # 固定 hybrid 后不再有 switch_strategy；旧模型输出统一映射为改写
    "切换策略": "reformulate",
    "切策略": "reformulate",
    "switch_strategy": "reformulate",
    "继续": "proceed",
    "直接生成": "proceed",
    "generate": "proceed",
    "continue": "proceed",
    "具体化": "specify",
    "specific": "specify",
    "specificity": "specify",
    "泛化": "generalize",
    "general": "generalize",
    "同义替换": "synonym_replace",
    "synonym": "synonym_replace",
    "synonym_replacement": "synonym_replace",
    "paraphrase": "synonym_replace",
    "拒答": "refused",
    "诚实拒答": "refused",
    "编造": "fabricated",
    "幻觉": "fabricated",
    "其他": "other",
}


def _normalize_enum(value: object) -> object:
    """归一化 LLM 输出的枚举值：strip + lower + 中文变体映射。"""
    if isinstance(value, str):
        value = value.strip().lower()
        value = _ENUM_ALIASES.get(value, value)
    return value


NormalizedQueryType = Annotated[QueryType, BeforeValidator(_normalize_enum)]
NormalizedAmbiguityType = Annotated[AmbiguityType, BeforeValidator(_normalize_enum)]
NormalizedFailureMode = Annotated[FailureMode, BeforeValidator(_normalize_enum)]
NormalizedSuggestedAction = Annotated[SuggestedAction, BeforeValidator(_normalize_enum)]
NormalizedRewriteStrategy = Annotated[RewriteStrategy, BeforeValidator(_normalize_enum)]


class AnalyzeResult(BaseModel):
    """analyze 节点输出：查询意图分析。"""

    query_type: NormalizedQueryType = Field(
        ..., description="查询类型：factual(事实查询) / reasoning(推理) / chitchat(寒暄) / complex(复杂)"
    )
    needs_retrieval: bool = Field(
        ..., description="是否需要检索知识库"
    )
    needs_clarification: bool = Field(
        default=False,
        description="是否需要先向用户澄清（缺槽位 / 多解释 / 约束冲突）",
    )
    ambiguity_type: NormalizedAmbiguityType | None = Field(
        default=None,
        description=(
            "歧义类型：missing_slot(缺槽位) / "
            "multiple_interpretations(多解释) / "
            "conflicting_constraints(约束冲突)"
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description="需要澄清时向用户提出的一个具体问题；不需要时为空",
    )
    reason: str = Field(default="", description="判断理由（简要）")


class EvaluateResult(BaseModel):
    """evaluate 节点输出：诊断式检索质量评估。

    failure_mode 区分三种失败原因，suggested_action 给出对应动作。
    """

    failure_mode: NormalizedFailureMode = Field(
        ...,
        description=(
            "检索质量诊断："
            "low_recall(召回不足，关键信息缺失) / "
            "irrelevant(检索结果与问题无关) / "
            "sufficient(已足够回答)"
        ),
    )
    suggested_action: NormalizedSuggestedAction = Field(
        ...,
        description=(
            "建议动作："
            "reformulate(改写查询重试) / "
            "proceed(直接生成)"
        ),
    )
    reason: str = Field(default="", description="诊断理由")


class CondenseResult(BaseModel):
    """多轮会话层输出：把追问改写成独立问题。"""

    standalone_query: str = Field(..., description="不依赖上下文、可独立检索的问题")
    reason: str = Field(default="", description="改写理由（简要）")


class ReformulateResult(BaseModel):
    """reformulate 节点输出：查询改写。"""

    reformulated_query: str = Field(..., description="改写后的查询")
    rewrite_strategy: NormalizedRewriteStrategy = Field(
        ...,
        description=(
            "改写策略："
            "specify(具体化) / "
            "generalize(泛化) / "
            "synonym_replace(同义替换)"
        ),
    )


class VerifyResult(BaseModel):
    """verify 节点输出：幻觉检测。"""

    is_faithful: bool = Field(
        ..., description="回答是否忠实于检索文档（True=无幻觉，False=存在幻觉）"
    )
    reason: str = Field(default="", description="检测原因说明")


class RefusalJudgeResult(BaseModel):
    """不可答题裁决输出：RAG 系统对语料外问题的行为判定（评估脚本专用）。"""

    verdict: Annotated[
        Literal["refused", "fabricated", "other"],
        BeforeValidator(_normalize_enum),
    ] = Field(
        ...,
        description=(
            "refused(诚实承认无法回答) / "
            "fabricated(编造了具体的事实性内容) / "
            "other(答非所问等其他情况)"
        ),
    )
    reason: str = Field(default="", description="判定理由")

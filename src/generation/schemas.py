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
    "低召回": "low_recall",
    "召回不足": "low_recall",
    "不相关": "irrelevant",
    "无关": "irrelevant",
    "充足": "sufficient",
    "足够": "sufficient",
    "改写": "reformulate",
    "改写查询": "reformulate",
    "切换策略": "switch_strategy",
    "切策略": "switch_strategy",
    "继续": "proceed",
    "直接生成": "proceed",
    "具体化": "specify",
    "泛化": "generalize",
    "同义替换": "synonym_replace",
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
            "switch_strategy(切换检索策略) / "
            "proceed(直接生成)"
        ),
    )
    reason: str = Field(default="", description="诊断理由")


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

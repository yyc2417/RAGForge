"""结构化输出 Pydantic schema

配合 LLMClient.invoke_structured(prompt, schema) 使用，
通过 with_structured_output 确保 LLM 返回符合 schema 的对象。
"""

from pydantic import BaseModel, Field

from src.agent.state import (
    FailureMode,
    QueryType,
    RewriteStrategy,
    SuggestedAction,
)


class AnalyzeResult(BaseModel):
    """analyze 节点输出：查询意图分析。"""

    query_type: QueryType = Field(
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

    failure_mode: FailureMode = Field(
        ...,
        description=(
            "检索质量诊断："
            "low_recall(召回不足，分数普遍偏低) / "
            "irrelevant(检索结果与问题无关) / "
            "sufficient(已足够回答)"
        ),
    )
    suggested_action: SuggestedAction = Field(
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
    rewrite_strategy: RewriteStrategy = Field(
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

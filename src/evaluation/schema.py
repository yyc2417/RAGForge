"""评测 v2 统一 schema 与校验

设计目标：让自定义集、CRUD-RAG、RGB、MultiHop-RAG 等都能落到同一套结构：
- 检索：expected_sources / evidence
- 行为：expected_behavior = answer / refuse / clarify / chitchat
- 生成：reference_answer（可选）
- 划分：split = dev / test
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Behavior = Literal["answer", "refuse", "clarify", "chitchat"]


class EvalItem(BaseModel):
    """单条评测样本。"""

    id: str = ""
    query: str = Field(..., min_length=1)
    query_type: Literal["factual", "reasoning", "complex", "chitchat"] = "factual"
    difficulty: Literal["L1", "L2", "L3", "L4"] = "L1"
    answerable: bool = True
    expected_behavior: Behavior | None = None
    expected_source: str | None = None
    expected_sources: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    gold_keywords: list[str] = Field(default_factory=list)
    # clarify 样本的澄清问题应包含的实体/槽位关键词
    clarification_keywords: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    split: Literal["dev", "test"] = "dev"
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> "EvalItem":
        if self.expected_source and not self.expected_sources:
            self.expected_sources = [self.expected_source]
        if self.expected_sources and not self.expected_source:
            self.expected_source = self.expected_sources[0]

        if self.expected_behavior is None:
            if self.query_type == "chitchat":
                self.expected_behavior = "chitchat"
            else:
                self.expected_behavior = "answer" if self.answerable else "refuse"

        if self.expected_behavior == "answer":
            self.answerable = True
        elif self.expected_behavior == "clarify":
            # clarify 不期望生成答案；expected_sources 可保留为澄清后的参考答案
            self.answerable = False
        elif self.expected_behavior in ("refuse", "chitchat"):
            self.answerable = False
            # 不在这里清空 sources：保留原始输入，交给 validate_dataset 报错
        return self


def load_dataset(path: Path | str) -> list[EvalItem]:
    """加载 list 或 {"items": [...]} 格式的评测集。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", [])
    if not isinstance(data, list):
        raise ValueError("评测集必须是 JSON 数组或包含 items 数组的对象")
    return [EvalItem.model_validate(item) for item in data]


def validate_dataset(items: list[EvalItem], corpus_dir: Path | str | None = None) -> list[str]:
    """返回错误列表；空列表表示通过。"""
    errors: list[str] = []
    seen_queries: dict[str, int] = {}
    seen_ids: dict[str, int] = {}

    corpus_names: set[str] = set()
    if corpus_dir is not None:
        root = Path(corpus_dir)
        if not root.exists():
            errors.append(f"corpus_dir 不存在：{root}")
        else:
            corpus_names = {p.name for p in root.rglob("*") if p.is_file()}

    for index, item in enumerate(items):
        label = item.id or f"#{index}"
        normalized = " ".join(item.query.split()).lower()
        if normalized in seen_queries:
            errors.append(f"{label}: query 与第 {seen_queries[normalized]} 条重复")
        else:
            seen_queries[normalized] = index

        if item.id:
            if item.id in seen_ids:
                errors.append(f"{label}: id 与第 {seen_ids[item.id]} 条重复")
            seen_ids[item.id] = index

        if item.expected_behavior == "answer":
            if not item.expected_sources:
                errors.append(f"{label}: 可答题缺少 expected_sources")
            if not item.gold_keywords and not item.reference_answer:
                errors.append(f"{label}: 可答题缺少 gold_keywords 或 reference_answer")
        elif item.expected_behavior == "clarify":
            if not item.clarification_keywords:
                errors.append(f"{label}: clarify 样本缺少 clarification_keywords")
            # clarify 的 expected_sources / gold_keywords 可作为澄清后的参考答案，
            # 不参与当前召回与忠实度统计，因此允许保留。
        elif item.expected_behavior in ("refuse", "chitchat") and item.expected_sources:
            errors.append(f"{label}: {item.expected_behavior} 样本不应带 expected_sources")

        if corpus_dir is not None and item.expected_sources:
            for source in item.expected_sources:
                if Path(source).name not in corpus_names:
                    errors.append(f"{label}: corpus 中找不到来源 {source}")

    return errors


def dataset_hash(items: list[EvalItem]) -> str:
    """稳定的数据集指纹（用于报告元数据）。

    新增的可选字段为空时从 payload 中省略，避免仅升级 schema 就改变旧数据集
    的 hash（例如 clarification_keywords 全为空时不应影响 CRUD/v1 报告）。
    """
    payload: list[dict] = []
    for item in items:
        data = item.model_dump(mode="json")
        if not data.get("clarification_keywords"):
            data.pop("clarification_keywords", None)
        payload.append(data)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

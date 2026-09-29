# ADR-003：DeepSeek 不支持 structured output，选用 prompt+JSON 解析的替代方案

> **状态**：已采纳
> **日期**：2026-06-14
> **决策者**：宇诚

## 背景

RAGForge 的 Agent 状态机中有 4 个节点（analyze、evaluate、reformulate、verify）需要 LLM 返回结构化数据，对应各自的 Pydantic schema。LangChain 的标准做法是调用 `llm.with_structured_output(schema)` 让模型直接返回 JSON Schema 约束的结构化输出。然而 DeepSeek API 不支持 `response_format=json_schema` 参数，调用时返回 400 错误。需要一种兼容方案。

## 备选方案

### 方案 A：Prompt 指令 + JSON 容错解析 — 选定
- **概述**：在 system prompt 末尾追加 JSON 输出指令，要求模型以 markdown 代码块或纯 JSON 格式返回；通过 `_extract_json()` 容错解析 + Pydantic `model_validate` 校验
- **优点**：
  - 兼容所有 LLM，不依赖任何 API 特性
  - 解析层统一，切换模型零成本
  - 每个节点有 catch + 规则层兜底，即使解析失败也不中断流程
- **缺点**：
  - LLM 偶尔不遵守 schema，需要降级逻辑
  - 需要维护 prompt 模板和解析函数
- **决定**：选，最大兼容性，可控风险

### 方案 B：`with_structured_output`
- **概述**：LangChain 标准 API，底层使用 `response_format=json_schema`
- **优点**：
  - 框架原生支持，代码最简洁
- **缺点**：
  - DeepSeek 不支持该参数，返回 400 错误
- **决定**：不选，API 不兼容

### 方案 C：Function Calling / tool_use
- **概述**：利用 DeepSeek 的 function calling 能力，将 schema 定义为工具
- **优点**：
  - DeepSeek 支持 function calling
- **缺点**：
  - 改变了节点签名，每个节点需要注册工具
  - 增加了 Agent 编排的复杂度
  - function calling 的输出格式仍需解析
- **决定**：不选，复杂度过高，收益不大

### 方案 D：换用支持 structured output 的模型（如 GPT-4o）
- **概述**：直接更换模型绕过兼容性问题
- **优点**：
  - 代码最简洁，框架原生支持
- **缺点**：
  - 增加 API 调用成本
  - 不符合项目"低成本开源模型"的定位
- **决定**：不选，成本和定位不符

## 决定的理由

方案 A 是唯一同时满足兼容性、低成本和低复杂度的方案。核心实现如下：

```python
def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON，支持 markdown 代码块和普通 JSON"""
    # 尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # 回退：直接解析整个文本
    return json.loads(text)

# 节点中的使用模式
try:
    raw = llm.invoke(prompt)
    data = _extract_json(raw.content)
    result = AnalysisSchema.model_validate(data)
except Exception:
    # 规则层兜底
    result = AnalysisSchema(query_type="factual", needs_retrieval=True)
```

> **2026-09 补记**：现行实现已从本快照的两候选正则版演进为**多候选解析**——依次尝试代码围栏整体内容（非贪婪正则会在第一个 `}` 处截断嵌套 JSON，故取整段再校验）→ 最外层 `{...}` 块 → 原文，返回第一个能通过 `json.loads` 的候选；并增加「解析失败带纠错提示重试 1 次」。以 `src/generation/llm_client.py` 的 `_extract_json` 与 [generation.md](../modules/generation.md) 为准。

## 风险与缓解

| 风险 | 可能性 | 缓解措施 |
|------|--------|----------|
| LLM 不遵守 JSON schema | 中 | 每个节点有降级逻辑：analyze→factual+needs_retrieval，evaluate→规则层分数阈值 |
| JSON 解析失败（格式错误） | 低 | `_extract_json()` 支持多种格式 + try/except 兜底 |
| Prompt 模板需随 schema 变化维护 | 低 | Schema 变更频率低，集中在 prompts.py 管理 |

## 参考资料

- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [LangChain `with_structured_output` 文档](https://python.langchain.com/docs/how_to/structured_output/)
- [Pydantic model_validate 文档](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_validate)

# 生成层（Generation Layer）

## 1. 模块定位与设计动机

生成层（`src/generation/`）在 RAGForge 五层架构中承担**非结构化 LLM 输出到结构化 Agent 状态的适配桥梁**角色。它的核心设计动机源于一个工程约束：

> DeepSeek 不支持 `response_format=json_schema`（返回 400），无法使用 LangChain 的 `with_structured_output`。因此需要在 prompt 层面引导 JSON 输出，再用本地解析器完成类型校验。

这一约束决定了整个模块的架构走向：LLMClient 不仅是 ChatOpenAI 的封装，更是一套**结构化合约执行器**。

---

## 2. 架构与类设计

### 2.1 LLMClient — 多模态调用封装

LLMClient 对外暴露四种调用语义，覆盖 Agent 状态机的全部 LLM 需求：

```python
class LLMClient:
    def generate(self, query: str, context: list[Document]) -> str
    def generate_stream(self, query: str, context: list[Document]) -> Iterator[str]
    def invoke_with_prompt(self, prompt: ChatPromptTemplate, **kwargs) -> str
    def invoke_structured(self, prompt: ChatPromptTemplate, schema: type, **kwargs) -> BaseModel
```

| 调用方式 | 消费者 | 输出类型 |
|----------|--------|----------|
| `generate` | generate 节点 / SSE 端点 | `str` / `Iterator[str]` |
| `invoke_structured` | analyze / evaluate / reformulate / verify | Pydantic schema 实例 |

**关键设计取舍**：`generate` 挂载 `tenacity` 重试（3 次指数退避），`invoke_structured` 则不挂——结构化调用失败后各节点有规则兜底（见第 5 节），重试应交给调用方控制，避免在节点内盲目重试浪费延迟预算。

### 2.2 invoke_structured — JSON 合约执行流程

本模块最核心的方法。执行链路：从 `schema.model_fields` 提取字段名构建 field_hints → 追加到 prompt 末尾 → `chain = prompt | llm` invoke → `_extract_json` 三级降级解析 → `json.loads` → `schema.model_validate` → 返回实例。

**_extract_json 三级降级**：优先匹配 ` ```json {...}``` ` 代码块 → 匹配首个 `{...}` 块 → `strip()` 兜底。确保 LLM 输出格式不稳定时仍能提取有效 JSON。

### 2.3 PromptManager — 无状态模板注册表

```python
class PromptManager:
    QA_PROMPT: ChatPromptTemplate           # {context, input}
    ANALYZE_PROMPT: ChatPromptTemplate      # {query}
    EVALUATE_PROMPT: ChatPromptTemplate     # {query, retrieved_docs}
    REFORMULATE_PROMPT: ChatPromptTemplate  # {query, failure_mode, rewrite_hint, context}
    VERIFY_PROMPT: ChatPromptTemplate       # {query, answer, context}
```

所有模板定义为类属性（非实例属性），本质上将 PromptManager 当作**命名空间**使用。在 `build_agent_graph` 中通过 `partial(node_fn, llm=llm_client, prompts=prompt_manager)` 注入各节点。这个设计使得新增 prompt 只需在类中加一个类属性，无需修改任何构造函数或依赖注入链。

### 2.4 Schemas — Agent 状态机的类型合约

四个 schema 类定义了 LLM 输出与 AgentState 之间的**类型边界**：

```python
class AnalyzeResult(BaseModel):    # → query_type: Literal["factual","reasoning","chitchat","complex"]
class EvaluateResult(BaseModel):   # → failure_mode: Literal["low_recall","irrelevant","sufficient"]
class ReformulateResult(BaseModel):# → rewrite_strategy: Literal["specify","generalize","synonym_replace"]
class VerifyResult(BaseModel):     # → is_faithful: bool
```

所有枚举字段使用 `Literal` 类型（从 `src.agent.state` 导入），这意味着 `model_validate` 会在解析阶段拒绝非法值——即使 LLM 输出了一个"看起来合理"但不在状态机路由表中的值，也会被 Pydantic 拦截为校验错误，而非悄悄进入条件边导致不可预测的路由行为。

---

## 3. 数据流

```mermaid
graph TD
    A[Agent 节点] -->|query + prompt + schema| B[invoke_structured]
    B --> C[追加 JSON 字段指令到 prompt 末尾]
    C --> D[chain = prompt | ChatOpenAI]
    D --> E[DeepSeek API]
    E --> F[_extract_json 三级降级解析]
    F --> G[json.loads]
    G --> H[schema.model_validate]
    H --> I[类型安全的 schema 实例]
    I --> J[节点提取字段写入 AgentState]

    K[generate 节点] -->|query + documents| L[generate]
    L --> M[_format_context 拼接文档]
    M --> N[QA_PROMPT | ChatOpenAI]
    N --> O[自由文本回答]

    P[SSE 端点] -->|query + documents| Q[generate_stream]
    Q --> R[chain.stream 逐 token yield]
```

---

## 4. 设计模式

| 模式 | 体现 | 设计意图 |
|------|------|----------|
| **Facade** | LLMClient 封装 ChatOpenAI 初始化、链式调用、token 统计 | 节点不直接接触 LangChain 管道细节 |
| **Template Method** | invoke_structured 固定"追加指令→调用→解析→校验"流程 | 四个节点共享执行骨架，仅替换 prompt/schema |
| **Contract** | Pydantic + Literal 约束 | 非法值在 schema 层拦截，不进入状态机路由 |

**关键决策**：生成层只负责"尽力解析"，失败后 raise 给调用节点，由各节点实现降级逻辑，避免生成层耦合不同节点的降级策略。

---

## 5. 错误处理与降级链路

整个调用链形成三层防御体系：

```
第 1 层：LLM 调用 ─── tenacity 重试（仅 generate 方法，3 次指数退避）
              │
第 2 层：解析容错 ─── _extract_json 三级降级（markdown → 裸 JSON → strip）
              │       model_validate 校验 Literal 值域
              │
第 3 层：节点降级 ─── 每个节点 catch + 规则兜底
              │
              ├── analyze 失败 → factual + needs_retrieval=True（宁可多检索）
              ├── evaluate 失败 → 规则层分数阈值兜底（avg_score < 0.3 → switch）
              ├── reformulate 失败 → 保留原查询 + strategy=specify
              └── verify 失败 → is_faithful=True（宁可放过不误判，避免触发无效循环）
```

**设计洞察**：每个节点的降级策略都偏向"保守安全"方向——analyze 宁可多检索、verify 宁可放过。这是 Agentic RAG 系统的核心容错哲学：**宁可多花一次检索成本，也不要因 LLM 故障导致死循环或错误回答**。

`_record_usage` 使用 `except Exception` 静默吞掉所有异常（仅 `logger.debug`），确保 token 统计这条"旁路观测"永远不会影响主流程执行。

---

## 6. 集成上下文

LLMClient 在 `build_agent_graph` 中作为单例创建，通过 `functools.partial` 注入 5 个节点（analyze/evaluate/reformulate/generate/verify）。generate 节点使用 `generate`（自由文本），其余使用 `invoke_structured`（结构化）。SSE 流式端点通过 `generate_stream` 对接 `astream_events(v2)`。

---

## 7. 配置项

| 配置键 | 默认值 | 来源 |
|--------|--------|------|
| `deepseek_model` | `deepseek-v4-flash` | `src.config.Settings` |
| `deepseek_base_url` | `https://api.deepseek.com` | `src.config.Settings` |
| `deepseek_api_key` | `""` | `.env` 环境变量 |
| `temperature` | `0.3` | LLMClient 构造时硬编码 |

temperature 硬编码为 0.3 是一个有意识的取舍：偏低温度保证结构化 JSON 输出的稳定性，但也牺牲了 generate 节点回答的多样性。若需分离控制，未来可将 temperature 提升为 `invoke_structured` 和 `generate` 的独立参数。

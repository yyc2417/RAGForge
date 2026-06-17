# Agent 决策层（Agent）

## 模块职责

Agent 决策层基于 LangGraph 构建 8 节点状态机，自主决策检索策略，包含诊断式评估（`failure_mode` + `suggested_action` 驱动策略切换/查询改写）和幻觉检测闭环。

## 状态设计

### AgentState

全局状态容器，`TypedDict(total=False)` 允许节点只更新部分字段：

```python
class AgentState(TypedDict, total=False):
    query: str                                    # 用户输入
    query_type: QueryType                         # factual | reasoning | chitchat | complex
    needs_retrieval: bool                         # 是否需要检索
    retrieval_strategy: RetrievalStrategy          # vector | bm25 | hybrid | none
    documents: list[Document]                     # 检索结果
    retrieval_scores: list[float]                 # 检索分数
    failure_mode: FailureMode                     # low_recall | irrelevant | sufficient
    suggested_action: SuggestedAction              # reformulate | switch_strategy | proceed
    diagnosis_reason: str                         # 诊断原因
    reformulated_query: str                       # 改写后的查询
    rewrite_strategy: RewriteStrategy              # specify | generalize | synonym_replace
    answer: str                                   # 生成的回答
    is_faithful: bool                             # 幻觉检测结果
    verification_reason: str                      # 校验原因
    previous_strategy: RetrievalStrategy           # 策略切换前的策略
    iteration_count: int                          # 重检索迭代计数
    max_iterations: int                           # 最大迭代数（默认 3）
    messages: Annotated[list[str], add]           # 决策轨迹日志（自动追加）
```

**关键字段**：
- `failure_mode` + `suggested_action`：诊断式评估核心，驱动策略切换/查询改写
- `iteration_count` / `max_iterations`：防死循环（默认最多 3 轮重检索）
- `messages: Annotated[list, add]`：LangGraph 自动追加，记录完整决策轨迹

### 初始状态工厂

```python
def initial_state(query: str, max_iterations: int = 3) -> AgentState:
    """构造初始状态，包含 query、iteration_count=0、max_iterations、空 messages。"""
```

## 状态机拓扑

```mermaid
graph TD
    START --> Analyze[analyze]
    Analyze --> Decide[decide]
    Decide -->|strategy=none| Generate[generate]
    Decide -->|strategy!=none| Retrieve[retrieve]
    Retrieve --> Evaluate[evaluate]
    Evaluate -->|proceed| Generate
    Evaluate -->|switch_strategy| Switch[switch_strategy]
    Evaluate -->|reformulate| Reformulate[reformulate]
    Switch --> Retrieve
    Reformulate --> Retrieve
    Generate -->|strategy=none| END
    Generate -->|strategy!=none| Verify[verify]
    Verify -->|is_faithful| END
    Verify -->|not faithful & iter<max| Reformulate
```

## 节点说明

| 节点 | 需要 LLM | 职责 |
|------|----------|------|
| `analyze` | 是 | 分析查询意图，判断 `query_type` 和 `needs_retrieval`，LLM 失败时降级为 factual+需检索 |
| `decide` | 否 | 纯规则决策：chitchat→none，factual→vector，reasoning→bm25，complex→hybrid |
| `retrieve` | 否 | 按 `retrieval_strategy` 调用对应检索器，优先使用 `reformulated_query` |
| `evaluate` | 是 | 诊断检索质量：规则层（分数阈值） + LLM 层（语义判断），输出 `failure_mode` + `suggested_action` |
| `reformulate` | 是 | 根据 `failure_mode` 改写查询（specify/synonym_replace/generalize），递增 `iteration_count` |
| `switch_strategy` | 否 | 纯规则升级策略：vector→bm25→hybrid，单向升级不回退 |
| `generate` | 是 | 基于检索上下文调用 LLM 生成回答 |
| `verify` | 是 | 幻觉检测：检验回答是否忠实于检索文档，不通过则递增 `iteration_count` |

## 条件边（路由函数）

### route_after_decide

```python
def route_after_decide(state: AgentState) -> str:
    """strategy=none → generate（跳过检索），否则 → retrieve。"""
```

### route_after_evaluate

```python
def route_after_evaluate(state: AgentState) -> str:
    """三路分支：proceed → generate / switch_strategy → switch_strategy / reformulate → reformulate。
    达上限（iteration_count >= max_iterations）强制 generate（防死循环）。"""
```

### route_after_generate

```python
def route_after_generate(state: AgentState) -> str:
    """strategy=none（chitchat）→ END（跳过 verify），其他 → verify。"""
```

### route_after_verify

```python
def route_after_verify(state: AgentState) -> str:
    """is_faithful → END / 幻觉 & 未达上限 → reformulate / 达上限 → END。"""
```

## 安全机制

**双重保护防死循环**：

| 机制 | 配置 | 说明 |
|------|------|------|
| `max_iterations` | 3（状态级） | `evaluate` 和 `verify` 路由函数检查，达上限强制走向 `generate`/`END` |
| `recursion_limit` | 50（调用级） | LangGraph 编译后图的执行上限，第二道防线 |

**计数递增点**：
- `reformulate_node`：每次改写查询时 `iteration_count += 1`
- `verify_node`：检测到幻觉时 `iteration_count += 1`

## 依赖注入

`build_agent_graph()` 通过 `functools.partial` 将外部依赖注入到各节点函数：

```python
def build_agent_graph(
    vector_store: VectorStore,
    bm25_retriever: BM25Retriever,
    llm_client: LLMClient,
    prompt_manager: PromptManager,
    hybrid_retriever=None,  # 可选，Task 6 注入
    reranker=None,          # 可选，Task 6 注入
):
    graph.add_node("analyze", partial(analyze_node, llm=llm_client, prompts=prompt_manager))
    graph.add_node("decide", decide_node)  # 纯规则，无依赖
    graph.add_node("retrieve", partial(retrieve_node,
        vector_store=vector_store, bm25=bm25_retriever,
        hybrid_retriever=hybrid_retriever, reranker=reranker))
    # ... 其他节点
```

**节点统一签名**：`node(state: AgentState, *, llm, prompts, **deps) -> dict`，返回部分状态更新。

## 数据流图

```mermaid
graph LR
    Q[用户查询] --> A[analyze]
    A --> D[decide]
    D --> R[retrieve]
    R --> E[evaluate]
    E -->|proceed| G[generate]
    E -->|reformulate| RF[reformulate]
    E -->|switch| SS[switch_strategy]
    RF --> R
    SS --> R
    G --> V[verify]
    V -->|faithful| ANS[最终回答]
    V -->|hallucination| RF
```

# Task 5：Agent 状态机（LangGraph）

## 阶段概述

本阶段是 RAGForge 的核心——用 LangGraph 实现 Agentic RAG 状态机。Task 5 将线性的"检索→生成"升级为自适应的 8 节点状态机，支持查询分析、策略决策、检索质量评估、查询改写、幻觉检测等智能行为。这是系统从"基础 RAG"到"智能 RAG"的关键跃迁。

**前置依赖**：Task 1-4（配置、文档处理、检索、生成）  
**后续依赖**：Task 6（混合检索集成）、Task 7（API 服务）、Task 8（评估）

---

## 核心知识点

### 知识点 1：LangGraph 核心概念

LangGraph 是基于状态图的 Agent 框架，核心模型是 **StateGraph（状态图）+ 节点 + 边 + 条件边**：

```python
from langgraph.graph import END, START, StateGraph

def build_agent_graph(...):
    """组装 LangGraph 状态机。"""
    graph = StateGraph(AgentState)  # 用 AgentState 定义全局状态
    
    # 添加节点（partial 注入依赖）
    graph.add_node("analyze", partial(analyze_node, llm=llm_client, prompts=prompt_manager))
    graph.add_node("decide", decide_node)
    graph.add_node("retrieve", partial(retrieve_node, vector_store=vector_store, ...))
    graph.add_node("evaluate", partial(evaluate_node, llm=llm_client, prompts=prompt_manager))
    graph.add_node("reformulate", partial(reformulate_node, llm=llm_client, prompts=prompt_manager))
    graph.add_node("switch_strategy", switch_strategy_node)
    graph.add_node("generate", partial(generate_node, llm=llm_client))
    graph.add_node("verify", partial(verify_node, llm=llm_client, prompts=prompt_manager))
    
    # 边定义
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "decide")
    
    # 条件边：decide 后是否检索
    graph.add_conditional_edges(
        "decide",
        route_after_decide,  # 路由函数
        {"retrieve": "retrieve", "generate": "generate"},
    )
    
    # ... 更多边定义
    
    compiled = graph.compile()  # 编译为可执行图
    return compiled
```

**StateGraph 模型**：
- **节点（Node）**：执行具体操作的函数，接收当前状态、返回部分状态更新
- **边（Edge）**：`add_edge("A", "B")` 表示 A 执行完后无条件进入 B
- **条件边（Conditional Edge）**：`add_conditional_edges("A", route_func, {...})` 根据 `route_func` 返回值决定下一步

**`compile()` 生成可执行图**：`compile()` 将图定义编译为 LangGraph 的 `CompiledGraph` 对象，提供 `invoke()` / `stream()` / `astream_events()` 等执行方法。

**`invoke()` 执行状态机**：`graph.invoke(initial_state, config)` 从初始状态开始，按图定义的路径执行节点，直到到达 `END`。`config={"recursion_limit": 50}` 设置最大节点执行次数（防死循环的第二道防线）。

### 知识点 2：TypedDict 状态设计

`AgentState` 是全局状态容器，用 `TypedDict` 定义字段类型：

```python
from operator import add
from typing import Annotated, Literal, TypedDict

# 字面量类型（与各节点输出严格对应）
QueryType = Literal["factual", "reasoning", "chitchat", "complex"]
RetrievalStrategy = Literal["vector", "bm25", "hybrid", "none"]
FailureMode = Literal["low_recall", "irrelevant", "sufficient"]
SuggestedAction = Literal["reformulate", "switch_strategy", "proceed"]

class AgentState(TypedDict, total=False):
    """LangGraph 全局状态。total=False 允许节点只更新部分字段。"""
    
    # 输入
    query: str
    
    # analyze 节点输出
    query_type: QueryType
    needs_retrieval: bool
    
    # decide 节点输出
    retrieval_strategy: RetrievalStrategy
    
    # retrieve 节点输出
    documents: list[Document]
    retrieval_scores: list[float]
    
    # evaluate 节点输出（诊断式评估）
    failure_mode: FailureMode
    suggested_action: SuggestedAction
    diagnosis_reason: str
    
    # reformulate 节点输出
    reformulated_query: str
    rewrite_strategy: RewriteStrategy
    
    # generate 节点输出
    answer: str
    
    # verify 节点输出
    is_faithful: bool
    verification_reason: str
    
    # 流程控制
    iteration_count: int
    max_iterations: int
    messages: Annotated[list[str], add]  # reducer 语义：自动追加
```

**`total=False` 允许部分更新**：`TypedDict` 默认要求所有字段都必须存在，`total=False` 放宽这个约束。节点只需返回自己负责的字段（如 `analyze_node` 只返回 `query_type` 和 `needs_retrieval`），无需填充整个状态。

**`Annotated[list, add]` 的 reducer 语义**：`messages` 字段的类型是 `Annotated[list[str], add]`，其中 `add` 是 `operator.add`。这是 LangGraph 的 reducer 机制：当节点返回 `{"messages": ["新消息"]}` 时，不是覆盖整个列表，而是追加到现有列表。这使得 `messages` 成为"决策轨迹日志"，自动累积所有节点的输出。

**Literal 类型约束**：`QueryType = Literal["factual", "reasoning", "chitchat", "complex"]` 限制字段只能取这 4 个字符串值。这提供了类型安全（IDE 会提示可选值），也让状态机逻辑更清晰（`if query_type == "chitchat"` 是确定性的）。

### 知识点 3：条件边（conditional_edges）

条件边根据路由函数的返回值决定下一步走向：

```python
def route_after_decide(state: AgentState) -> str:
    """decide 后：strategy=none 跳过检索直接生成，否则检索。"""
    return "generate" if state.get("retrieval_strategy") == "none" else "retrieve"

def route_after_evaluate(state: AgentState) -> str:
    """evaluate 后：根据诊断结果三路分支（达上限强制生成）。"""
    action = state.get("suggested_action", "proceed")
    iter_count = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    
    if action == "proceed":
        return "generate"
    if iter_count >= max_iter:
        logger.warning(f"[route] iter={iter_count}≥{max_iter}，强制 generate")
        return "generate"
    if action == "switch_strategy":
        return "switch_strategy"
    return "reformulate"

def route_after_verify(state: AgentState) -> str:
    """verify 后：忠实则结束，幻觉则改写重检索（达上限强制结束）。"""
    if state.get("is_faithful", True):
        return "end"
    iter_count = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    if iter_count >= max_iter:
        logger.warning(f"[route] verify 后 iter={iter_count}≥{max_iter}，强制 END")
        return "end"
    return "reformulate"

# 注册条件边
graph.add_conditional_edges(
    "decide",
    route_after_decide,
    {"retrieve": "retrieve", "generate": "generate"},  # 可能的返回值 → 目标节点
)
```

**路由函数签名**：接收 `state`（当前状态），返回字符串（下一步节点名或 `"end"`）。返回值必须在 `add_conditional_edges` 的映射字典中有对应项。

**与 `add_edge` 的区别**：
- `add_edge("A", "B")`：A 执行完后无条件进入 B
- `add_conditional_edges("A", route_func, {...})`：A 执行完后，根据 `route_func(state)` 返回值决定进入哪个节点

**三个路由函数的设计**：
1. **`route_after_decide`**：chitchat 跳过检索直接生成，其他进入检索
2. **`route_after_evaluate`**：三路分支（proceed → generate / switch_strategy → switch_strategy / reformulate → reformulate），达上限强制生成
3. **`route_after_verify`**：忠实则结束，幻觉则改写，达上限强制结束

### 知识点 4：诊断式评估设计

evaluate 节点不是简单打分，而是输出 **failure_mode（失败模式）+ suggested_action（建议动作）**：

```python
def evaluate_node(state, *, llm, prompts):
    """诊断检索质量，决定后续动作。"""
    query = state["query"]
    documents = state.get("documents", [])
    scores = state.get("retrieval_scores", [])
    
    # 规则层：先看召回是否充足
    if not documents:
        failure_mode, action = "low_recall", "switch_strategy"
        reason = "检索返回 0 条文档，召回严重不足"
        ...
    
    # 平均相关性分数
    valid_scores = [s for s in scores if s > 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else SCORE_NO_HITS
    
    # LLM 层：语义诊断
    retrieved_docs = _format_docs(documents, scores)
    try:
        result = llm.invoke_structured(
            prompts.EVALUATE_PROMPT, EvaluateResult,
            query=query, retrieved_docs=retrieved_docs,
        )
        llm_failure = result.failure_mode
        llm_action = result.suggested_action
        reason = result.reason
    except Exception as e:
        # 规则兜底：分数普遍低 → low_recall+switch，否则 sufficient+proceed
        if avg_score >= 0 and avg_score < SCORE_LOW_RECALL_THRESHOLD:
            llm_failure, llm_action = "low_recall", "switch_strategy"
            reason = f"平均相关性分数 {avg_score:.3f} < {SCORE_LOW_RECALL_THRESHOLD}（规则兜底）"
        else:
            llm_failure, llm_action = "sufficient", "proceed"
            reason = "LLM 诊断失败，默认 sufficient（规则兜底）"
    
    # 融合：规则层的 low_recall 信号优先（分数硬证据）
    if avg_score >= 0 and avg_score < SCORE_LOW_RECALL_THRESHOLD and llm_failure != "irrelevant":
        failure_mode, action = "low_recall", "switch_strategy"
        reason = f"规则层触发：平均分数 {avg_score:.3f} 过低 → 切换策略 | LLM: {reason}"
    else:
        failure_mode, action = llm_failure, llm_action
    ...
```

**failure_mode 三种分类**：
- `low_recall`：召回不足，相关性分数普遍偏低（< 0.3）
- `irrelevant`：检索结果与问题无关（语义不匹配）
- `sufficient`：检索结果已足够回答

**suggested_action 三种动作**：
- `switch_strategy`：切换检索策略（vector → bm25 → hybrid）
- `reformulate`：改写查询（specify / generalize / synonym_replace）
- `proceed`：直接生成回答

**规则层+LLM 层融合策略**：
1. **规则层**：检查文档数量和平均分数（硬证据）
2. **LLM 层**：语义诊断（软判断）
3. **融合**：规则层的 `low_recall` 信号优先（分数是客观证据），LLM 的 `irrelevant` 信号优先（语义判断更准确）

**为什么不是简单打分**：简单打分（如 0-10 分）无法指导后续动作。诊断式评估输出"失败原因 + 建议动作"，让状态机能做出智能决策（切策略 vs 改写查询）。

### 知识点 5：防死循环机制

状态机包含循环（evaluate → reformulate → retrieve → evaluate），需要防止无限循环：

```python
# state.py
def initial_state(query: str, max_iterations: int = 3) -> AgentState:
    """构造一个初始 AgentState（含默认流程控制字段）。"""
    return {
        "query": query,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "messages": [],
    }

# reformulate.py
def reformulate_node(state, *, llm, prompts):
    # 递增迭代计数（防死循环的关键状态）
    iteration_count = state.get("iteration_count", 0) + 1
    ...
    return {"reformulated_query": ..., "iteration_count": iteration_count, ...}

# verify.py
def verify_node(state, *, llm, prompts):
    # 幻觉检测不通过时递增迭代计数
    if not is_faithful:
        iteration_count = state.get("iteration_count", 0) + 1
    else:
        iteration_count = state.get("iteration_count", 0)
    ...
    return {"is_faithful": ..., "iteration_count": iteration_count, ...}

# graph.py 路由函数
def route_after_evaluate(state):
    iter_count = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)
    if iter_count >= max_iter:
        logger.warning(f"[route] iter={iter_count}≥{max_iter}，强制 generate")
        return "generate"
    ...
```

**双重保护机制**：
1. **状态机级 `max_iterations`**：`iteration_count` 在 `reformulate_node` 和 `verify_node` 中递增，路由函数检查 `iter_count >= max_iter` 时强制走向 `generate` / `END`
2. **框架级 `recursion_limit`**：`graph.invoke(..., config={"recursion_limit": 50})` 限制节点执行总次数，超过则抛出 `GraphRecursionError`

**为什么需要双重保护**：
- `max_iterations` 控制"重检索循环"次数（业务逻辑层）
- `recursion_limit` 控制节点执行总次数（框架安全层），防止 `max_iterations` 逻辑 bug 导致无限循环

---

## 设计模式与架构决策

**状态机模式（State Machine）**：LangGraph 的 StateGraph 是显式状态机，节点是状态转换的"动作"，边是"转换路径"。相比隐式状态机（如 while 循环 + if-else），显式状态机更易理解和调试（可以可视化图结构）。

**依赖注入模式（Dependency Injection）**：节点函数接收依赖作为参数（`llm`, `prompts`, `vector_store`），用 `functools.partial` 预绑定。这比全局变量更易测试（可以注入 Mock 对象）。

**诊断式评估模式**：evaluate 节点输出"失败原因 + 建议动作"而非简单分数，让状态机能做出智能决策。这种模式适合需要多轮重试的自适应系统。

**优雅降级模式**：每个 LLM 节点都有 try-except 降级逻辑，LLM 失败时用规则兜底。这确保系统不会因为单次 LLM 调用失败就崩溃。

---

## 关键代码解读

### state.py：状态定义

```python
from operator import add
from typing import Annotated, Literal, TypedDict

QueryType = Literal["factual", "reasoning", "chitchat", "complex"]
RetrievalStrategy = Literal["vector", "bm25", "hybrid", "none"]
FailureMode = Literal["low_recall", "irrelevant", "sufficient"]
SuggestedAction = Literal["reformulate", "switch_strategy", "proceed"]
RewriteStrategy = Literal["specify", "generalize", "synonym_replace"]
```

**设计意图**：`Literal` 类型定义所有可能的枚举值，IDE 会提示可选值（如 `if query_type == "factual"`），避免拼写错误。这些类型在 `schemas.py` 中被 Pydantic 模型复用，确保 LLM 输出符合约束。

### graph.py：状态机组装

```python
from functools import partial

def build_agent_graph(
    vector_store: VectorStore,
    bm25_retriever: BM25Retriever,
    llm_client: LLMClient,
    prompt_manager: PromptManager,
    hybrid_retriever=None,
    reranker=None,
):
    """组装 LangGraph 状态机。"""
    graph = StateGraph(AgentState)
    
    # 添加节点（partial 注入依赖）
    graph.add_node("analyze", partial(analyze_node, llm=llm_client, prompts=prompt_manager))
    graph.add_node("decide", decide_node)
    graph.add_node(
        "retrieve",
        partial(
            retrieve_node,
            vector_store=vector_store,
            bm25=bm25_retriever,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
        ),
    )
    ...
```

**设计意图**：`partial` 预绑定依赖参数，节点函数签名保持统一（第一个参数始终是 `state`）。`hybrid_retriever` 和 `reranker` 是可选参数（Task 5 阶段为 None，Task 6 注入），实现了向前兼容。

### analyze.py：查询分析节点

```python
def analyze_node(state: AgentState, *, llm: LLMClient, prompts: PromptManager) -> dict:
    """分析查询意图，判断 query_type 和 needs_retrieval。"""
    query = state["query"]
    logger.info(f"[analyze] 分析查询：{query}")
    
    try:
        result: AnalyzeResult = llm.invoke_structured(
            prompts.ANALYZE_PROMPT, AnalyzeResult, query=query
        )
        query_type = result.query_type
        needs_retrieval = result.needs_retrieval
    except Exception as e:
        logger.warning(f"[analyze] LLM 分析失败，降级为 factual+需检索：{e}")
        query_type = "factual"
        needs_retrieval = True
    
    msg = f"analyze: query_type={query_type}, needs_retrieval={needs_retrieval}"
    logger.info(f"[analyze] → {msg}")
    return {
        "query_type": query_type,
        "needs_retrieval": needs_retrieval,
        "messages": [msg],
    }
```

**设计意图**：`analyze_node` 用 LLM 分析查询意图，输出 `query_type` 和 `needs_retrieval`。降级逻辑确保 LLM 失败时仍能继续（默认"事实查询+需要检索"，最保守的假设）。`messages` 字段记录决策轨迹，便于调试。

### decide.py：策略决策节点

```python
def decide_node(state: AgentState) -> dict:
    """根据查询类型规则化决策检索策略（无 LLM 调用）。"""
    query_type = state.get("query_type", "factual")
    needs_retrieval = state.get("needs_retrieval", True)
    
    if not needs_retrieval or query_type == "chitchat":
        strategy = "none"
    elif query_type == "factual":
        strategy = "vector"
    elif query_type == "reasoning":
        strategy = "bm25"
    else:  # complex 或未知
        strategy = "hybrid"
    
    msg = f"decide: query_type={query_type} → strategy={strategy}"
    logger.info(f"[decide] {msg}")
    return {
        "retrieval_strategy": strategy,
        "messages": [msg],
    }
```

**设计意图**：`decide_node` 是纯规则决策（无 LLM 调用），快速且确定性。决策规则基于经验：
- chitchat → none（跳过检索）
- factual → vector（语义匹配适合事实查询）
- reasoning → bm25（关键词匹配适合精确术语）
- complex → hybrid（多角度覆盖）

### retrieve.py：检索执行节点

```python
def retrieve_node(state: AgentState, *, vector_store, bm25, hybrid_retriever=None, reranker=None) -> dict:
    """按策略执行检索，返回文档与分数。"""
    query = state.get("reformulated_query") or state["query"]  # 优先使用改写后的查询
    strategy = state.get("retrieval_strategy", "vector")
    
    t0 = time.perf_counter()
    
    if strategy == "vector":
        docs, scores = vector_store.search_with_scores(query)
    elif strategy == "bm25":
        docs, scores = bm25.search_with_scores(query)
    elif strategy == "hybrid":
        if hybrid_retriever is None:
            # Task 5 阶段 hybrid 未注入时，退化为向量+BM25 简单拼接去重
            v_docs, v_scores = vector_store.search_with_scores(query)
            b_docs, b_scores = bm25.search_with_scores(query)
            docs, scores = _merge_dedupe(v_docs, v_scores, b_docs, b_scores)
        else:
            docs, scores = hybrid_retriever.search_with_scores(query)
            if reranker is not None and getattr(reranker, "is_available", lambda: False)():
                docs = reranker.rerank(query, docs)
                scores = [0.0] * len(docs)  # rerank 后分数失去含义
    ...
```

**设计意图**：`retrieve_node` 根据 `strategy` 选择检索器。`query = state.get("reformulated_query") or state["query"]` 优先使用改写后的查询（如果有）。Task 5 阶段 `hybrid_retriever` 为 None，用简单合并去重兜底；Task 6 注入真正的 RRF 混合检索器。

---

## 踩坑记录

### 问题 1：verify 回退导致无限循环

**问题现象**：hybrid 模式下某些查询触发 `Recursion limit of 50 reached` 错误。

**排查过程**：
1. 查看日志发现 `verify → reformulate → retrieve → evaluate → generate → verify` 循环
2. 检查 `iteration_count` 发现只在 `reformulate_node` 递增，`verify_node` 不递增
3. 循环路径 `verify → reformulate` 中，verify 不递增计数，导致增长过慢

**根因**：`iteration_count` 仅在 `reformulate_node` 递增，`verify_node` 检测幻觉后回退 reformulate 但不递增计数，导致 `max_iterations` 防线失效。

**解决方案**：在 `verify_node` 中，当 `is_faithful=False` 时也递增 `iteration_count`：

```python
if not is_faithful:
    iteration_count = state.get("iteration_count", 0) + 1
else:
    iteration_count = state.get("iteration_count", 0)
```

### 问题 2：chitchat 误判幻觉

**问题现象**："你好"、"谢谢"等寒暄被 verify 节点判为幻觉（`is_faithful=False`）。

**排查过程**：
1. 检查 verify 输入发现 `documents=[]`（chitchat 跳过检索，无文档）
2. VERIFY_PROMPT 要求"回答应基于检索文档"，无文档时 LLM 判为幻觉

**根因**：chitchat（strategy=none）走 `generate → verify`，但 documents 为空，verify 误判。

**解决方案**：新增 `route_after_generate` 函数，chitchat 跳过 verify 直接结束：

```python
def route_after_generate(state: AgentState) -> str:
    """generate 后：strategy=none（chitchat）跳过 verify 直接结束。"""
    if state.get("retrieval_strategy") == "none":
        return "end"
    return "verify"
```

---

## 与其他模块的交互

**输入接口**：
- `LLMClient`：从 Task 4 注入，提供 generate / invoke_structured 方法
- `PromptManager`：从 Task 4 注入，提供 ANALYZE / EVALUATE / REFORMULATE / VERIFY PROMPT
- `VectorStore` / `BM25Retriever`：从 Task 3 注入
- `HybridRetriever` / `Reranker`：从 Task 6 注入（Task 5 阶段为 None）

**输出接口**：
- `build_agent_graph(...)` → `CompiledGraph`：供 Task 7 API 服务调用
- `build_agent_graph_from_pipeline(...)` → `CompiledGraph`：便捷封装，内部构造默认 PromptManager
- `initial_state(query, max_iterations)` → `AgentState`：构造初始状态

**衔接方式**：
- Task 6 的 `HybridRetriever` 和 `Reranker` 通过 `build_agent_graph` 参数注入
- Task 7 的 `get_graph()` 调用 `build_agent_pipeline()` → `build_agent_graph_from_pipeline()`
- Task 8 的 evaluator 调用 `graph.invoke(initial_state(query))` 执行状态机

---

## 本阶段收获总结

1. **LangGraph 的 StateGraph 是显式状态机**：节点是动作，边是路径，条件边是分支，比隐式 while 循环更易理解和调试
2. **TypedDict + total=False 允许部分状态更新**：节点只返回自己负责的字段，无需填充整个状态
3. **Annotated[list, add] 实现 reducer 语义**：messages 字段自动追加而非覆盖，成为决策轨迹日志
4. **诊断式评估输出"失败原因+建议动作"**：比简单打分更能指导后续智能决策
5. **双重防死循环机制**：状态机级 `max_iterations` + 框架级 `recursion_limit`，确保安全性
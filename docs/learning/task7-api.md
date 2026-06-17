# Task 7：API 服务（FastAPI + SSE）

## 阶段概述

本阶段将 RAGForge 的 Agent 状态机封装为 HTTP API 服务，提供同步对话和 SSE 流式对话两种端点。Task 7 是系统的对外接口层，让前端、移动端或其他服务可以通过 HTTP 调用 RAGForge。同时集成了 LangSmith 可观测性追踪。

**前置依赖**：Task 1-6（所有核心功能）  
**后续依赖**：Task 8（评估系统可通过 API 测试）

---

## 核心知识点

### 知识点 1：FastAPI 应用工厂模式

应用工厂模式用函数构建 FastAPI 实例，而非全局 `app = FastAPI()`：

```python
from fastapi import FastAPI

_app: FastAPI | None = None

def create_app() -> FastAPI:
    """构建 FastAPI 应用实例。"""
    _configure_langsmith()
    
    application = FastAPI(
        title="RAGForge",
        description="Agentic RAG — LangGraph 状态机驱动的自适应检索增强生成",
        version="0.1.0",
    )
    application.include_router(router)
    logger.info("[api] FastAPI 应用构建完成（/health /chat /chat/stream）")
    return application

def get_app() -> FastAPI:
    """返回模块级 FastAPI 单例（供 uvicorn `--factory` 或直接引用）。"""
    global _app
    if _app is None:
        _app = create_app()
    return _app
```

**为什么不用全局 `app = FastAPI()`**：
1. **测试友好**：测试可以调用 `create_app()` 构建独立实例，避免测试间状态污染
2. **配置灵活**：`create_app()` 可以在构建前设置环境变量（如 LangSmith 配置），全局 `app` 在 import 时已构建，无法干预
3. **延迟初始化**：`get_app()` 用模块级单例，首次调用时才构建，避免 import 时就触发重型初始化

**`include_router` 注册路由**：`application.include_router(router)` 将 `routes.py` 中定义的路由注册到应用。Router 是独立的路由集合，可以跨应用复用。

### 知识点 2：SSE（Server-Sent Events）协议

SSE 是 HTTP 的单向流式推送协议，与 WebSocket 的区别：

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 方向 | 单向（服务端→客户端） | 双向 |
| 协议 | HTTP | WS/WSS |
| 重连 | 浏览器自动重连 | 需手动实现 |
| 复杂度 | 低（文本流） | 高（二进制帧） |

```python
from sse_starlette.sse import EventSourceResponse

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, graph=Depends(get_graph)):
    """SSE 流式对话端点：逐 token + 节点事件推送。"""
    
    async def event_generator():
        """SSE 事件生成器。"""
        final_state: dict = {}
        try:
            async for event in graph.astream_events(
                initial_state(request.query),
                config={"recursion_limit": 50},
                version="v2",
            ):
                kind = event.get("event")
                name = event.get("name", "")
                
                # 节点完成事件
                if kind == "on_chain_end" and name in _AGENT_NODES:
                    yield {"event": "node", "data": json.dumps({"node": name})}
                
                # LLM 逐 token 事件
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is not None:
                        content = getattr(chunk, "content", "") or ""
                        if content:
                            yield {"event": "token", "data": json.dumps({"text": content})}
            
            # 流结束后推送来源文档
            sources = _extract_sources(final_state)
            if sources:
                yield {"event": "sources", "data": json.dumps([s.model_dump() for s in sources])}
            yield {"event": "done", "data": json.dumps({"session_id": session_id})}
        
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
    
    return EventSourceResponse(event_generator())
```

**EventSourceResponse 的事件格式**：每个 `yield` 返回 `{"event": "类型", "data": "内容"}`，SSE 协议将其格式化为：

```
event: token
data: {"text": "你好"}

event: node
data: {"node": "generate"}

event: done
data: {"session_id": "abc123"}
```

**前端如何消费**：浏览器的 `EventSource` API 或 `fetch` + `ReadableStream`：

```javascript
const response = await fetch("/chat/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: "什么是过拟合？"})
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    const text = decoder.decode(value);
    // 解析 SSE 格式：event: xxx\ndata: {...}\n\n
}
```

### 知识点 3：LangGraph astream_events vs astream

LangGraph 提供两种流式 API：

**`astream()`**：只返回节点级状态快照，每个节点执行完后输出一次完整状态：

```python
async for state in graph.astream(initial_state):
    # state 是节点执行后的完整 AgentState
    print(state.get("answer"))  # 只有 generate 节点后才有值
```

**`astream_events(version="v2")`**：能捕获底层 LLM 的逐 token 事件：

```python
async for event in graph.astream_events(initial_state, version="v2"):
    kind = event.get("event")
    # kind 可能是：
    # "on_chain_start" / "on_chain_end" — 节点开始/结束
    # "on_chat_model_stream" — LLM 逐 token 输出
    # "on_llm_start" / "on_llm_end" — LLM 调用开始/结束
    
    if kind == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        print(chunk.content)  # 逐 token 打印
```

**为什么 SSE 必须用 `astream_events`**：
- `astream()` 只在节点完成时输出一次，无法实现"逐 token 推送"
- `astream_events(v2)` 能捕获 `on_chat_model_stream`，这是 LLM 生成的每个 token 片段

**事件类型分类**：
- `on_chain_start` / `on_chain_end`：LangGraph 节点级事件
- `on_chat_model_stream`：LLM 逐 token 事件（仅 `astream_events` 能捕获）
- `on_llm_start` / `on_llm_end`：LLM 调用级事件

**关键决策：不加 `streaming=True`**：`ChatOpenAI` 构造函数有 `streaming` 参数，但 `astream_events` 会自动通过 LCEL chain 的 `.stream()` 触发底层模型的流式分发，**不需要** 手动设置 `streaming=True`。设置反而可能影响 `usage_metadata` 的返回方式。

### 知识点 4：依赖注入惰性初始化

FastAPI 的 `Depends` 机制实现依赖注入，配合 double-check locking 实现惰性单例：

```python
import threading

_graph_lock = threading.Lock()
_graph = None

def get_graph():
    """获取编译后的 LangGraph Agent 单例（惰性初始化）。"""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:  # double-check
                from src.pipeline import build_agent_pipeline
                logger.info("[api] 首次请求，正在构建 Agent 管道...")
                _graph = build_agent_pipeline()
                logger.info("[api] Agent 管道构建完成，复用单例")
    return _graph

def get_metrics() -> MetricsCollector:
    """获取 MetricsCollector 单例。"""
    return MetricsCollector()

# routes.py 中使用
@router.post("/chat")
async def chat(request: ChatRequest, graph=Depends(get_graph)):
    final_state = await graph.ainvoke(initial_state(request.query))
    ...
```

**延迟导入避免启动时构建重型管道**：`from src.pipeline import build_agent_pipeline` 放在函数内部，而非模块顶部。这样 import `api` 模块时不会触发管道构建（解析文档 + 构建索引 + 加载 embedding + 组装状态机，约需 5-10 秒）。

**double-check locking 确保线程安全**：多线程环境下，两个请求可能同时进入 `if _graph is None`，`with _graph_lock` 互斥锁确保只有一个线程执行构建，第二个线程在锁内再次检查 `if _graph is None` 时发现已构建，直接返回。

**`Depends(get_graph)` 的工作方式**：FastAPI 在每次请求时调用 `get_graph()`，返回值注入到 `graph` 参数。由于 `get_graph()` 内部是单例，只有首次请求触发构建，后续请求复用。

### 知识点 5：LangSmith 零代码集成

LangSmith 是 LangChain 的可观测性平台，通过环境变量自动启用：

```python
def _configure_langsmith() -> None:
    """根据配置启用 LangSmith 追踪（仅设置环境变量，LangChain 自动读取）。"""
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info(f"[api] LangSmith 追踪已启用，项目={settings.langsmith_project}")
    elif settings.langsmith_tracing:
        logger.warning("[api] LANGSMITH_TRACING=true 但未配置 LANGSMITH_API_KEY，跳过")
```

**零代码改动**：只需设置环境变量，LangChain/LangGraph 内部自动检测并上报 trace。无需在业务代码中手动埋点。

**环境变量说明**：
- `LANGCHAIN_TRACING_V2=true`：启用追踪
- `LANGCHAIN_API_KEY`：LangSmith API key
- `LANGCHAIN_PROJECT`：项目名称（在 LangSmith 面板中分组显示）

**追踪内容**：每次 `graph.invoke()` / `graph.astream_events()` 调用自动上报：
- 每个节点的输入/输出
- 每个 LLM 调用的 prompt/response/token 使用量
- 延迟和错误信息

---

## 设计模式与架构决策

**应用工厂模式**：`create_app()` 构建 FastAPI 实例，`get_app()` 返回单例。测试友好、配置灵活、延迟初始化。

**依赖注入模式**：`Depends(get_graph)` 将 Agent 图注入到路由函数，路由函数无需关心图的构建和生命周期。

**SSE 单向推送**：相比 WebSocket 更简单，适合"查询→流式回答"的场景（客户端只需发送一次请求，服务端持续推送）。

**惰性初始化 + double-check locking**：Agent 管道构建成本高，首次请求时构建，后续复用。线程安全设计确保并发请求不会重复构建。

---

## 关键代码解读

### app.py：应用工厂

```python
from fastapi import FastAPI
from src.api.routes import router
from src.config import settings

_app: FastAPI | None = None

def create_app() -> FastAPI:
    """构建 FastAPI 应用实例。"""
    _configure_langsmith()
    
    application = FastAPI(
        title="RAGForge",
        description="Agentic RAG — LangGraph 状态机驱动的自适应检索增强生成",
        version="0.1.0",
    )
    application.include_router(router)
    logger.info("[api] FastAPI 应用构建完成（/health /chat /chat/stream）")
    return application
```

**设计意图**：`_configure_langsmith()` 在构建应用前设置环境变量，确保 LangChain 在 import 时就能读取到配置。`include_router(router)` 注册所有路由。FastAPI 的 `title` / `description` / `version` 用于自动生成 OpenAPI 文档（`/docs` 端点）。

### routes.py：路由定义

```python
from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

_AGENT_NODES = {"analyze", "decide", "retrieve", "evaluate",
                "reformulate", "switch_strategy", "generate", "verify"}

_INTERNAL_NAMES = {"LangGraph", "WriteDocuments", "RunnableSequence",
                   "ChatOpenAI", "RunnableLambda", "ToolCallingLLM"}

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查端点。"""
    return HealthResponse(status="ok", version="0.1.0")
```

**设计意图**：`APIRouter()` 是路由集合，可以跨应用复用。`_AGENT_NODES` 白名单过滤 `on_chain_end` 事件，只推送 8 个业务节点（忽略 LangGraph 内部节点）。`_INTERNAL_NAMES` 黑名单忽略 LangGraph 内部链路名，避免噪声。

```python
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, graph=Depends(get_graph)) -> ChatResponse:
    """异步对话端点：完整执行 Agent 状态机后返回结构化响应。"""
    session_id = request.session_id or str(uuid.uuid4())[:8]
    metrics = MetricsCollector()
    metrics.reset()  # 单次请求隔离指标
    
    t0 = time.perf_counter()
    final_state = await graph.ainvoke(
        initial_state(request.query),
        config={"recursion_limit": 50},
    )
    e2e_ms = (time.perf_counter() - t0) * 1000
    
    answer_text = final_state.get("answer", "")
    sources = _extract_sources(final_state)
    resp_metrics = _extract_metrics(metrics, e2e_ms)
    
    return ChatResponse(
        answer=answer_text,
        sources=sources,
        agent_trace=final_state.get("messages", []),
        metrics=resp_metrics,
        session_id=session_id,
    )
```

**设计意图**：`async def chat` 是异步函数，`await graph.ainvoke()` 避免阻塞事件循环（如果用同步 `graph.invoke()`，会阻塞整个 FastAPI worker）。`metrics.reset()` 确保每次请求的指标独立（不与其他请求混淆）。

```python
@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, graph=Depends(get_graph)):
    """SSE 流式对话端点：逐 token + 节点事件推送。"""
    
    async def event_generator():
        final_state: dict = {}
        try:
            async for event in graph.astream_events(
                initial_state(request.query),
                config={"recursion_limit": 50},
                version="v2",
            ):
                kind = event.get("event")
                name = event.get("name", "")
                
                # 节点完成事件：只推送 8 个业务节点
                if kind == "on_chain_end" and name in _AGENT_NODES:
                    yield {"event": "node", "data": json.dumps({"node": name})}
                
                # 根 graph 完成：捕获 LangGraph 根链路的 on_chain_end
                elif kind == "on_chain_end" and name == "LangGraph":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict) and "answer" in output:
                        final_state = output
                
                # LLM 逐 token 事件
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is not None:
                        content = getattr(chunk, "content", "") or ""
                        if content:
                            yield {"event": "token", "data": json.dumps({"text": content})}
            
            # 流结束后推送来源文档
            sources = _extract_sources(final_state)
            if sources:
                yield {"event": "sources", "data": json.dumps([s.model_dump() for s in sources])}
            yield {"event": "done", "data": json.dumps({"session_id": session_id, "e2e_latency_ms": round(e2e_ms, 2)})}
        
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
    
    return EventSourceResponse(event_generator())
```

**设计意图**：`astream_events(version="v2")` 捕获 `on_chat_model_stream` 实现逐 token 推送。`name == "LangGraph"` 捕获根链路的 `on_chain_end`，获取 `final_state`（包含 `documents` 用于提取来源）。`EventSourceResponse` 将 async generator 包装为 SSE 响应。

### schemas.py：请求/响应模型

```python
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """对话请求。"""
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(None, description="会话 ID（可选，用于追踪）")

class SourceDoc(BaseModel):
    """检索到的来源文档。"""
    content: str = Field(..., description="文档片段内容")
    source: str = Field("unknown", description="来源文件名")
    score: float = Field(0.0, description="相关性分数")

class ChatResponse(BaseModel):
    """对话响应（同步端点）。"""
    answer: str = Field(..., description="生成的回答")
    sources: list[SourceDoc] = Field(default_factory=list, description="检索到的来源文档")
    agent_trace: list[str] = Field(default_factory=list, description="Agent 决策轨迹")
    metrics: ResponseMetrics = Field(default_factory=ResponseMetrics, description="性能指标")
    session_id: str | None = Field(None, description="会话 ID（回传）")
```

**设计意图**：Pydantic 模型提供类型验证和自动序列化。`Field(...)` 表示必填字段，`Field(default_factory=list)` 表示默认为空列表。FastAPI 自动用这些模型生成 OpenAPI 文档（`/docs` 端点）。

---

## 踩坑记录

### 问题 1：`include_router` 后 `app.routes` 不显示端点

**问题现象**：`create_app()` 后遍历 `app.routes` 只看到 `/docs`、`/openapi.json` 等默认路由，看不到 `/health`、`/chat`、`/chat/stream`。

**排查过程**：
1. 检查 `router.routes` 确认 router 内部有 3 个路由
2. 怀疑 `include_router` 没生效

**根因**：FastAPI 0.137 的 `app.routes` 遍历方式对 `APIRouter` 的呈现与预期不同（router 作为整体挂载，遍历父 routes 看不到子 path）。

**解决方案**：改用 `app.openapi()["paths"]` 验证，确认 3 个端点全部注册。这不是 bug，是验证方式选错了。

### 问题 2：curl 在 PowerShell 中 JSON 转义破坏

**问题现象**：`curl -d '{"query":"你好"}'` 在 PowerShell 中报 `JSON decode error`。

**排查过程**：
1. 打印请求体发现双引号被吃掉
2. 检查 PowerShell 引号规则

**根因**：cmd → PowerShell → curl 的多层引号转义把双引号吃掉了。

**解决方案**：改用 `--data-binary '@file.json'` 从文件读取请求体，避免引号转义问题。

### 问题 3：SSE final_state 捕获失败

**问题现象**：`/chat/stream` 推送的 `sources` 事件为空数组，但同步 `/chat` 端点有来源文档。

**排查过程**：
1. 打印 `astream_events` 的所有事件，发现 `on_chain_end` 事件很多
2. 检查发现 `name == "LangGraph"` 的事件包含 `final_state`

**根因**：原实现用排除法匹配 `on_chain_end`（排除 `_INTERNAL_NAMES`），但 LangGraph 内部链路名在不同版本可能变化，导致匹配不稳定。

**解决方案**：改用正向匹配 `name == "LangGraph"`，只捕获根链路的 `on_chain_end`，获取 `final_state`。

### 问题 4：/chat 端点阻塞事件循环

**问题现象**：并发请求 `/chat` 时，第二个请求要等第一个完成后才开始处理。

**排查过程**：
1. 检查路由函数发现是 `def chat`（同步函数）
2. 同步函数中调用 `graph.invoke()`（同步阻塞调用）

**根因**：FastAPI 对同步路由函数会用线程池执行，但如果线程池满（默认 40 个线程），后续请求会等待。

**解决方案**：改为 `async def chat` + `await graph.ainvoke()`，避免阻塞事件循环。LangGraph 的 `ainvoke()` 是异步实现，不会阻塞。

---

## 与其他模块的交互

**输入接口**：
- `build_agent_pipeline()`：从 `src/pipeline.py` 注入，构建完整 Agent 管道
- `initial_state(query)`：从 `src/agent/state.py` 注入，构造初始状态
- `MetricsCollector`：从 `src/utils/metrics.py` 注入，记录性能指标
- `settings.api_host` / `settings.api_port`：从 Task 1 配置读取服务地址端口

**输出接口**：
- `GET /health` → `HealthResponse`：健康检查
- `POST /chat` → `ChatResponse`：同步对话（完整回答 + 来源 + 决策轨迹 + 指标）
- `POST /chat/stream` → SSE 流：逐 token + 节点事件 + 来源 + 完成信号

**衔接方式**：
- `get_graph()` 调用 `build_agent_pipeline()` 构建 Agent 图（首次请求时）
- 路由函数调用 `graph.ainvoke()` / `graph.astream_events()` 执行状态机
- `_extract_sources()` 从 `final_state["documents"]` 提取来源文档
- `_extract_metrics()` 从 `MetricsCollector` 提取性能指标

---

## 本阶段收获总结

1. **应用工厂模式比全局 app 更灵活**：测试友好、配置灵活、延迟初始化
2. **SSE 比 WebSocket 更简单，适合单向推送场景**：浏览器自动重连、协议简单、无需二进制帧处理
3. **`astream_events(v2)` 是实现逐 token 推送的关键**：`astream()` 只返回节点级快照，无法捕获 LLM 的 `on_chat_model_stream`
4. **依赖注入 + 惰性初始化避免启动时构建重型管道**：首次请求时构建，后续复用，线程安全
5. **LangSmith 零代码集成**：只需设置环境变量，LangChain 内部自动上报 trace，无需手动埋点
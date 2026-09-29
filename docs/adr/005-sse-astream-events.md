# ADR-005：SSE 流式端点选用 `astream_events(v2)` 而非 `astream()`

> **状态**：已采纳
> **日期**：2026-06-14
> **决策者**：宇诚

## 背景

RAGForge 的 `/chat/stream` 端点需要通过 SSE（Server-Sent Events）逐 token 推送 LLM 生成内容，同时向前端报告 Agent 节点的执行进度（如"正在分析查询"、"正在检索文档"）。LangGraph 提供了 `graph.astream()` 和 `graph.astream_events(version="v2")` 两种异步流式 API，需要选择合适的方案。

## 备选方案

### 方案 A：`astream_events(version="v2")` — 选定
- **概述**：捕获 `on_chat_model_stream` 事件实现真正逐 token 推送，同时捕获 `on_chain_end` 获取节点完成事件
- **优点**：
  - 能同时获取 token 级和节点级两种粒度的事件
  - 事件类型丰富，可区分模型流、工具调用、链完成等
  - LCEL chain 的 `.stream()` 自动触发底层模型流式分发，无需手动设置 `streaming=True`
- **缺点**：
  - 事件流较复杂，需要过滤内部链路名称
  - 事件 payload 结构较深，需要逐层提取
- **决定**：选，唯一同时满足逐 token 和节点进度的方案

### 方案 B：`astream()`
- **概述**：LangGraph 的标准流式 API，按节点输出状态快照
- **优点**：
  - API 简单，返回的就是状态字典的增量
- **缺点**：
  - 只返回节点级状态快照，每次输出是一整段文本而非逐 token
  - 无法实现打字机效果，用户体验差
- **决定**：不选，无法逐 token 推送

### 方案 C：`invoke()` + WebSocket 轮询
- **概述**：后端完整执行图，前端通过 WebSocket 轮询获取中间状态
- **优点**：
  - 前端可完全控制轮询频率
- **缺点**：
  - 复杂度高，需维护 WebSocket 连接和状态缓存
  - 轮询延迟大，实时性差
  - 服务端需要额外的状态存储
- **决定**：不选，复杂度和延迟不合理

### 方案 D：`LLMClient.generate_stream()` 单独调用
- **概述**：绕过 Agent 状态机，直接调用 LLM 的流式生成接口
- **优点**：
  - 实现最简单，直接拿到 token 流
- **缺点**：
  - 完全绕过 Agent 状态机，丢失查询分析、检索、评估等节点进度信息
  - 丧失了 Agentic RAG 的核心价值
- **决定**：不选，丢失 Agent 进度信息

## 决定的理由

`astream_events(v2)` 是唯一同时满足逐 token 推送和节点进度报告的方案。关键发现：**不需要**在 `ChatOpenAI` 构造时设置 `streaming=True`，LCEL chain 的 `.stream()` 会自动触发底层模型的流式分发。

```python
@app.post("/chat/stream")
async def chat_stream(query: str):
    async def event_generator():
        async for event in graph.astream_events(
            {"query": query}, version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                # 逐 token 推送
                token = event["data"]["chunk"].content
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            elif kind == "on_chain_end" and event["name"] not in _INTERNAL_NAMES:
                # 节点完成事件
                yield f"data: {json.dumps({'type': 'node', 'name': event['name']})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

> **2026-09 补记**：上图示例为决策期示意代码。生产实现（`src/api/routes.py`）已演进为：`sse-starlette` 的 `EventSourceResponse`（替代裸 `StreamingResponse`）、`POST` + JSON body（`ChatRequest`）而非 query 参数、结束事件为携带 `session_id` 与 `e2e_latency_ms` 的 JSON `done` 事件（替代 `[DONE]` 标记）、节点事件按 8 个业务节点**白名单**过滤（替代内部链路黑名单）。详见 [api.md](../modules/api.md)。

## 风险与缓解

| 风险 | 可能性 | 缓解措施 |
|------|--------|----------|
| 事件流复杂，内部链路产生大量噪声事件 | 高 | 维护 `_INTERNAL_NAMES` 黑名单，过滤 LangChain 内部链路 |
| 事件 payload 结构随 LangGraph 版本变化 | 低 | 锁定 langgraph 版本，升级时回归测试 |
| 前端未正确处理 SSE 格式 | 低 | 遵循 W3C SSE 规范，使用标准 `EventSource` API |

## 参考资料

- [LangGraph `astream_events` 文档](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/)
- [LangChain 事件类型参考](https://python.langchain.com/docs/concepts/callbacks/#event-types)
- [W3C Server-Sent Events 规范](https://html.spec.whatwg.org/multipage/server-sent-events.html)

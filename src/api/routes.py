"""FastAPI 路由定义

三个端点：
- GET  /health       健康检查（ready 反映管道是否就绪）
- POST /chat         同步对话（返回完整 ChatResponse）
- POST /chat/stream  SSE 流式对话（逐 token + 节点事件推送）

SSE 关键：使用 graph.astream_events(version="v2")，不是 astream()。
astream() 只返回节点级状态快照；astream_events() 捕获 LLM on_chat_model_stream
实现真正的逐 token 推送。

事件时序契约：node(retrieve) → sources → node(evaluate) → token* → done。
指标注入：每请求创建独立 MetricsCollector 并经 ContextVar 注入，
LangGraph 在 executor 线程中以 copy_context 执行同步节点，
节点内 get_current_collector() 取到的即本请求实例，并发请求互不污染。
"""

import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src import __version__
from src.agent.state import initial_state
from src.api.dependencies import get_graph
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse, ResponseMetrics, SourceDoc
from src.utils.logger import logger
from src.utils.metrics import MetricsCollector, reset_current_collector, set_current_collector

router = APIRouter()

# 8 个 Agent 节点名（用于过滤 on_chain_end 事件，只推送业务节点）
_AGENT_NODES = {"analyze", "decide", "retrieve", "evaluate",
                "reformulate", "switch_strategy", "generate", "verify"}


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """健康检查端点：status=ok 表示管道已就绪，starting 表示仍在构建。"""
    ready = getattr(request.app.state, "ready", False)
    return HealthResponse(status="ok" if ready else "starting", version=__version__)


def _extract_sources(state: dict) -> list[SourceDoc]:
    """从状态（完整状态或 retrieve 节点输出）提取来源文档列表。"""
    docs = state.get("documents", []) or []
    scores = state.get("retrieval_scores") or []
    sources = []
    for i, doc in enumerate(docs):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        score = scores[i] if i < len(scores) else 0.0
        sources.append(SourceDoc(
            content=doc.page_content[:200] if hasattr(doc, "page_content") else str(doc)[:200],
            source=meta.get("source", "unknown"),
            score=float(score) if score is not None else 0.0,
        ))
    return sources


def _extract_metrics(metrics: MetricsCollector, e2e_ms: float) -> ResponseMetrics:
    """从请求级 MetricsCollector 提取单次响应指标。"""
    summary = metrics.get_summary()
    retrieval = summary.get("retrieval_latency_ms", {})
    token = summary.get("token_usage", {})
    return ResponseMetrics(
        retrieval_latency_ms=retrieval.get("max", 0.0) if retrieval else 0.0,
        e2e_latency_ms=round(e2e_ms, 2),
        token_usage=token,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, graph=Depends(get_graph)) -> ChatResponse:
    """异步对话端点：完整执行 Agent 状态机后返回结构化响应。

    使用 graph.ainvoke() 避免阻塞事件循环。

    Args:
        request: 含 query 和可选 session_id
        graph: 注入的 Agent graph 单例

    Returns:
        ChatResponse：答案 + 来源 + 决策轨迹 + 指标

    Raises:
        HTTPException: 502，Agent 执行失败（LLM 重试耗尽等）
    """
    session_id = request.session_id or str(uuid.uuid4())
    # 请求级指标收集器：注入当前上下文，节点内记录落到本实例
    metrics = MetricsCollector()
    context_token = set_current_collector(metrics)
    try:
        t0 = time.perf_counter()
        try:
            final_state = await graph.ainvoke(
                initial_state(request.query),
                config={"recursion_limit": 50},
            )
        except Exception as e:  # noqa: BLE001 - 转换为结构化 502，与流式端点错误契约对齐
            logger.error(f"[chat] Agent 执行失败 query='{request.query[:100]}': {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Agent 执行失败：{type(e).__name__}: {e}",
            ) from e
        e2e_ms = (time.perf_counter() - t0) * 1000
    finally:
        reset_current_collector(context_token)

    answer_text = final_state.get("answer", "")
    sources = _extract_sources(final_state)
    resp_metrics = _extract_metrics(metrics, e2e_ms)

    logger.info(f"[chat] query='{request.query[:100]}' | e2e={e2e_ms:.0f}ms | "
                f"sources={len(sources)} | trace={len(final_state.get('messages', []))}")

    return ChatResponse(
        answer=answer_text,
        sources=sources,
        agent_trace=final_state.get("messages", []),
        metrics=resp_metrics,
        session_id=session_id,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, graph=Depends(get_graph)):
    """SSE 流式对话端点：逐 token + 节点事件推送。

    事件类型与时序：
    - {"event":"node","data":"<节点名>"}     Agent 节点执行完成
    - {"event":"sources","data":"[...]"}     来源文档（retrieve 节点完成后、生成前推送一次）
    - {"event":"token","data":"<token>"}     LLM 生成的 token 片段（仅 generate 节点）
    - {"event":"done","data":"[DONE]"}       流结束（含 session_id 与 e2e 延迟）
    - {"event":"error","data":"{...}"}       执行失败（流中断时）

    必须使用 graph.astream_events(version="v2") 才能捕获 on_chat_model_stream。
    """
    session_id = request.session_id or str(uuid.uuid4())
    # 请求级指标收集器：与 /chat 相同的隔离机制
    metrics = MetricsCollector()
    context_token = set_current_collector(metrics)

    t0 = time.perf_counter()

    async def event_generator():
        """SSE 事件生成器。"""
        final_state: dict = {}
        sources_sent = False
        try:
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
                        yield {"event": "node", "data": json.dumps({"node": name}, ensure_ascii=False)}
                        # sources 契约：生成前推送一次——retrieve 节点输出即含
                        # documents + retrieval_scores，在此时点推送
                        if name == "retrieve" and not sources_sent:
                            output = event.get("data", {}).get("output")
                            if isinstance(output, dict) and output.get("documents"):
                                yield {
                                    "event": "sources",
                                    "data": json.dumps(
                                        [s.model_dump() for s in _extract_sources(output)],
                                        ensure_ascii=False,
                                    ),
                                }
                                sources_sent = True

                    # 捕获最终状态：任一含 answer 键的链路输出（根图最后完成，覆盖节点输出）
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output")
                        if isinstance(output, dict) and "answer" in output:
                            final_state = output

                    # LLM 逐 token 事件：仅采纳 generate 节点的输出。
                    # analyze/evaluate 等结构化调用的 token 事件是 JSON 碎片，
                    # 不过滤会混入答案流（实测确认，必须按节点过滤）
                    elif kind == "on_chat_model_stream":
                        if event.get("metadata", {}).get("langgraph_node") == "generate":
                            chunk = event.get("data", {}).get("chunk")
                            if chunk is not None:
                                content = getattr(chunk, "content", "") or ""
                                if content:
                                    yield {"event": "token", "data": json.dumps({"text": content}, ensure_ascii=False)}
            finally:
                reset_current_collector(context_token)

            # 兜底：retrieve 事件未携带文档时（如 chitchat 跳过检索），从 final_state 提取
            if not sources_sent:
                sources = _extract_sources(final_state)
                if sources:
                    yield {
                        "event": "sources",
                        "data": json.dumps(
                            [s.model_dump() for s in sources], ensure_ascii=False
                        ),
                    }
                    sources_sent = True

            if not final_state:
                logger.warning("[SSE] 未捕获到 final_state（LangGraph 输出结构可能已变化）")

            # 流结束：推送 done 事件
            e2e_ms = (time.perf_counter() - t0) * 1000
            yield {
                "event": "done",
                "data": json.dumps({
                    "session_id": session_id,
                    "e2e_latency_ms": round(e2e_ms, 2),
                }, ensure_ascii=False),
            }
        except Exception as e:  # noqa: BLE001 - 流中断以 error 事件收尾
            logger.error(f"[chat/stream] 流式生成失败：{e}")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())

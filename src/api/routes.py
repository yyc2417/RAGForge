"""FastAPI 路由定义

三个端点：
- GET  /health       健康检查
- POST /chat         同步对话（返回完整 ChatResponse）
- POST /chat/stream  SSE 流式对话（逐 token + 节点事件推送）

SSE 关键：使用 graph.astream_events(version="v2")，不是 astream()。
astream() 只返回节点级状态快照；astream_events() 捕获 LLM on_chat_model_stream
实现真正的逐 token 推送。
"""

import json
import time
import uuid

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from src.agent.state import initial_state
from src.api.dependencies import get_graph, get_metrics
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse, ResponseMetrics, SourceDoc
from src.utils.logger import logger
from src.utils.metrics import MetricsCollector

router = APIRouter()

# 8 个 Agent 节点名（用于过滤 on_chain_end 事件，只推送业务节点）
_AGENT_NODES = {"analyze", "decide", "retrieve", "evaluate",
                "reformulate", "switch_strategy", "generate", "verify"}

# LangGraph 内部链路名（忽略，避免噪声）
_INTERNAL_NAMES = {"LangGraph", "WriteDocuments", "RunnableSequence",
                   "ChatOpenAI", "RunnableLambda", "ToolCallingLLM"}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """健康检查端点。"""
    return HealthResponse(status="ok", version="0.1.0")


def _extract_sources(state: dict) -> list[SourceDoc]:
    """从 final_state 提取来源文档列表。"""
    docs = state.get("documents", [])
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
    """从 MetricsCollector 提取单次响应指标。"""
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
    """
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

    logger.info(f"[chat] query='{request.query}' | e2e={e2e_ms:.0f}ms | "
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

    事件类型：
    - {"event":"node","data":"<节点名>"}     Agent 节点执行完成
    - {"event":"token","data":"<token>"}     LLM 生成的 token 片段（仅 generate 节点）
    - {"event":"sources","data":"[...]"}     最终来源文档（生成前推送一次）
    - {"event":"done","data":"[DONE]"}       流结束

    必须使用 graph.astream_events(version="v2") 才能捕获 on_chat_model_stream。
    """
    session_id = request.session_id or str(uuid.uuid4())[:8]
    metrics = MetricsCollector()
    metrics.reset()

    t0 = time.perf_counter()

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

                # 节点完成事件：只推送 8 个业务节点
                if kind == "on_chain_end" and name in _AGENT_NODES:
                    yield {"event": "node", "data": json.dumps({"node": name}, ensure_ascii=False)}

                # 根 graph 完成：仅捕获 LangGraph 根链路的 on_chain_end（最可靠的 final state）
                elif kind == "on_chain_end" and name == "LangGraph":
                    output = event.get("data", {}).get("output")
                    if isinstance(output, dict) and "answer" in output:
                        final_state = output
                        logger.debug("[SSE] 捕获 final_state via LangGraph root chain")

                # LLM 逐 token 事件（generate 节点产生）
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk is not None:
                        content = getattr(chunk, "content", "") or ""
                        if content:
                            yield {"event": "token", "data": json.dumps({"text": content}, ensure_ascii=False)}

            # 流结束后推送来源文档（从 final_state 提取）
            e2e_ms = (time.perf_counter() - t0) * 1000
            sources = _extract_sources(final_state)
            if sources:
                yield {
                    "event": "sources",
                    "data": json.dumps(
                        [s.model_dump() for s in sources], ensure_ascii=False
                    ),
                }
            yield {
                "event": "done",
                "data": json.dumps({
                    "session_id": session_id,
                    "e2e_latency_ms": round(e2e_ms, 2),
                }, ensure_ascii=False),
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"[chat/stream] 流式生成失败：{e}")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())

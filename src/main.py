"""RAGForge 主入口模块

保留 Task 4 兼容入口（build_rag_pipeline, answer），
CLI/Server 和 Agent 管道逻辑已拆分到 cli.py 和 pipeline.py。

向后兼容 re-export：
- from src.cli import run_cli, run_server, main
- from src.pipeline import build_agent_pipeline, ask_agent
"""

from typing import TYPE_CHECKING

from src.cli import main, run_cli, run_server  # noqa: F401
from src.pipeline import ask_agent, build_agent_pipeline  # noqa: F401

if TYPE_CHECKING:  # 仅类型检查用，运行时延迟导入保持启动轻量
    from src.generation.llm_client import LLMClient
    from src.retrieval import VectorStore


def build_rag_pipeline() -> tuple["VectorStore", "LLMClient"]:
    """构建基础管道：ingestion → retrieval(VectorStore) → generation。

    Task 4 兼容入口（test_stage1.py 使用）；重型导入延迟到函数内。
    """
    from src.config import PROJECT_ROOT, settings
    from src.generation.llm_client import LLMClient
    from src.ingestion import DocumentParser, EmbeddingService, TextChunker
    from src.retrieval import VectorStore
    from src.utils.logger import logger

    parser = DocumentParser()
    docs = parser.parse_directory(PROJECT_ROOT / settings.data_dir)
    chunks = TextChunker().split(docs)
    embedder = EmbeddingService()
    vector_store = VectorStore.load_or_build(chunks, embedder)
    llm = LLMClient()
    logger.info("[pipeline] RAG 基础管道就绪")
    return vector_store, llm


def answer(
    vector_store: "VectorStore", llm: "LLMClient", question: str
) -> tuple[str, float]:
    """Task 4 线性管道的单问题回答（检索→生成），返回 (回答, 端到端延迟ms)。"""
    import time

    from src.utils.logger import logger
    from src.utils.metrics import MetricsCollector

    metrics = MetricsCollector()
    t0 = time.perf_counter()

    tr = time.perf_counter()
    docs = vector_store.search(question)
    metrics.record_retrieval_latency((time.perf_counter() - tr) * 1000)

    answer_text = llm.generate(question, docs)
    e2e = (time.perf_counter() - t0) * 1000
    metrics.record_e2e_latency(e2e)
    return answer_text, e2e


if __name__ == "__main__":
    main()

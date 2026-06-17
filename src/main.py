"""RAGForge 主入口模块

保留 Task 4 兼容入口（build_rag_pipeline, answer），
CLI/Server 和 Agent 管道逻辑已拆分到 cli.py 和 pipeline.py。

向后兼容 re-export：
- from src.cli import run_cli, run_server, main
- from src.pipeline import build_agent_pipeline, ask_agent
"""

import time

from src.cli import main, run_cli, run_server  # noqa: F401
from src.config import PROJECT_ROOT, settings
from src.generation.llm_client import LLMClient
from src.ingestion import DocumentParser, TextChunker, EmbeddingService
from src.pipeline import ask_agent, build_agent_pipeline  # noqa: F401
from src.retrieval import VectorStore
from src.utils.logger import logger
from src.utils.metrics import MetricsCollector

DATA_DIR = PROJECT_ROOT / settings.data_dir


def build_rag_pipeline() -> tuple[VectorStore, LLMClient]:
    """构建基础管道：ingestion → retrieval(VectorStore) → generation。

    Task 4 兼容入口（test_stage1.py 使用）。
    """
    parser = DocumentParser()
    docs = parser.parse_directory(DATA_DIR)
    chunker = TextChunker()
    chunks = chunker.split(docs)
    embedder = EmbeddingService()
    vector_store = VectorStore.load_or_build(chunks, embedder)
    llm = LLMClient()
    logger.info("[pipeline] RAG 基础管道就绪")
    return vector_store, llm


def answer(vector_store: VectorStore, llm: LLMClient, question: str) -> tuple[str, float]:
    """Task 4 线性管道的单问题回答（检索→生成），返回 (回答, 端到端延迟ms)。"""
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

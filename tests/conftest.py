"""pytest 共享 fixtures

测试分层约定：
- 单元测试（默认全部）：LLM 用 FakeLLM stub、Chroma 索引写入 pytest 临时目录，
  零网络、可重复运行（HF 离线模式强制，embedding 模型走本地缓存）
- 集成测试（tests/test_integration.py，integration marker）：真实调用 DeepSeek
  API，由环境变量 RUN_INTEGRATION=1 门控，默认跳过
"""

import os

# 真·零网络铁律：强制 HF 离线，杜绝套件隐性访问 huggingface.co。
# 必须在导入 src.*（间接导入 transformers）之前设置——huggingface_hub
# 在 import 时读取该开关。
# 首次在新环境运行测试前，一次性预缓存 embedding 模型（约 90MB）：
#   uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# 需要联网调试时，显式导出 HF_HUB_OFFLINE=0 即可绕过。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

import pytest
from langchain_core.documents import Document

# pytest 以 prepend 模式将本目录加入 sys.path，可直接导入 fakes
from fakes import FakeLLM, FakeRetriever, make_script  # noqa: F401
from src.agent.graph import build_agent_graph
from src.config import settings
from src.generation.prompts import PromptManager
from src.ingestion import DocumentParser, EmbeddingService, TextChunker
from src.retrieval import BM25Retriever, VectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 真实语料 / 本地模型（session 级，避免重复加载）──────────────────


@pytest.fixture(scope="session")
def sample_docs() -> list[Document]:
    """解析 data/sample 真实语料。"""
    return DocumentParser().parse_directory(PROJECT_ROOT / settings.data_dir)


@pytest.fixture(scope="session")
def chunks(sample_docs) -> list[Document]:
    """真实语料的切分结果（session 内共享）。"""
    return TextChunker().split(sample_docs)


@pytest.fixture(scope="session")
def embedder() -> EmbeddingService:
    """本地 embedding 模型（session 内单次加载）。"""
    return EmbeddingService()


@pytest.fixture(scope="session")
def vector_store(chunks, embedder, tmp_path_factory) -> VectorStore:
    """写入 pytest 临时目录的 VectorStore（不污染真实 chroma_db）。"""
    persist_dir = tmp_path_factory.mktemp("chroma_session") / "chroma_db"
    old = settings.chroma_persist_dir
    settings.chroma_persist_dir = str(persist_dir)
    try:
        store = VectorStore(embedder)
        store.build_index(chunks)
        yield store
    finally:
        settings.chroma_persist_dir = old


@pytest.fixture(scope="session")
def bm25(chunks) -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.build_index(chunks)
    return retriever


# ── 函数级 fixtures ────────────────────────────────────────────────


@pytest.fixture()
def chroma_dir(tmp_path, monkeypatch) -> Path:
    """把 chroma_persist_dir 指向本测试专属的临时目录（测试结束自动恢复）。"""
    target = tmp_path / "chroma_db"
    monkeypatch.setattr(settings, "chroma_persist_dir", str(target))
    return target


@pytest.fixture()
def fake_llm() -> FakeLLM:
    """默认脚本的 Fake LLM（factual → 检索充足 → 生成 → 验证通过）。"""
    return FakeLLM(make_script())


@pytest.fixture()
def make_graph():
    """构造「真实状态机 + Fake 依赖」的工厂，可传入自定义脚本。"""

    def _make(script: dict | None = None):
        llm = FakeLLM(script if script is not None else make_script())
        return build_agent_graph(
            vector_store=FakeRetriever(),
            bm25_retriever=FakeRetriever(),
            llm_client=llm,
            prompt_manager=PromptManager(),
            hybrid_retriever=None,
            reranker=None,
        )

    return _make

"""优质库归档与加权 RRF 优质通道测试（离线）"""

import uuid

from langchain_core.documents import Document

import src.knowledge.service as kb_service_module
from src.config import settings
from src.ingestion import EmbeddingService
from src.knowledge import KnowledgeBaseService
from src.knowledge.quality import QualityArchive
from src.retrieval import VectorStore
from src.retrieval.hybrid import HybridRetriever


class FakeVector:
    def __init__(self, quality_score: float = 0.9) -> None:
        self.quality_score = quality_score
        self.quality_doc = Document(
            page_content="优质库：远程办公每周最多两天。",
            metadata={"source": "quality/abc.md", "source_type": "qa_archive"},
        )
        self.normal_doc = Document(
            page_content="普通文档：公司制度。", metadata={"source": "handbook.md"}
        )

    def search_with_scores(self, query, k=None, filter=None):
        if filter and filter.get("source_type") == "qa_archive":
            return [self.quality_doc], [self.quality_score]
        return [self.normal_doc], [0.9]


class FakeBM25:
    def search_with_scores(self, query, k=None):
        return [Document(page_content="BM25 文档", metadata={"source": "bm25.md"})], [12.0]


def test_quality_archive_roundtrip(tmp_path):
    archive = QualityArchive(tmp_path / "quality")
    mid = str(uuid.uuid4())
    path = archive.archive(
        message_id=mid,
        question="远程办公每周几天？",
        original_query="那远程办公呢？",
        answer="每周最多两天。",
        sources=[{"source": "handbook.md", "page_number": 1, "chunk_index": 2, "score": 0.9, "content": "每周两天"}],
        session_id="s1",
        model="test-model",
    )
    assert path is not None and path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Q: 远程办公每周几天？")
    assert "每周最多两天" in text
    assert "handbook.md" in text
    assert archive.list()[0]["message_id"] == mid
    # 幂等
    assert archive.archive(
        message_id=mid, question="q", original_query="q", answer="a",
        sources=[], session_id="s", model="m",
    ) == path
    assert archive.remove(mid) is True
    assert archive.list() == []


def test_quality_archive_rejects_empty_and_bad_id(tmp_path):
    archive = QualityArchive(tmp_path / "quality")
    assert archive.archive(
        message_id="bad", question="q", original_query="q", answer="a",
        sources=[], session_id="s", model="m",
    ) is None
    assert archive.archive(
        message_id=str(uuid.uuid4()), question="q", original_query="q", answer="",
        sources=[], session_id="s", model="m",
    ) is None


def test_quality_channel_boosts_archive_doc():
    hybrid = HybridRetriever(
        FakeVector(), FakeBM25(),
        quality_weight=1.3, quality_top_k=3,
        quality_min_relevance=0.3, quality_max_per_source=2,
    )
    docs, scores = hybrid.search_with_scores("远程办公", k=3)
    assert docs[0].metadata.get("source_type") == "qa_archive"
    assert scores[0] > scores[1]


def test_quality_channel_respects_relevance_threshold():
    hybrid = HybridRetriever(
        FakeVector(quality_score=0.1), FakeBM25(),
        quality_weight=1.3, quality_top_k=3,
        quality_min_relevance=0.3, quality_max_per_source=2,
    )
    docs, _ = hybrid.search_with_scores("远程办公", k=3)
    assert all(d.metadata.get("source_type") != "qa_archive" for d in docs)


def test_vector_store_metadata_filter(tmp_path, monkeypatch):
    """Chroma metadata filter 必须真的只返回 qa_archive 文档。"""
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma_filter"))
    store = VectorStore(EmbeddingService())
    store.build_index([
        Document(
            page_content="优质库：远程办公每周两天。",
            metadata={"source": "q.md", "source_type": "qa_archive", "chunk_index": 0},
        ),
        Document(
            page_content="普通文档：远程办公制度。",
            metadata={"source": "d.md", "source_type": "document", "chunk_index": 0},
        ),
    ])
    docs, _ = store.search_with_scores(
        "远程办公", k=5, filter={"source_type": "qa_archive"}
    )
    assert docs
    assert all(d.metadata.get("source_type") == "qa_archive" for d in docs)


def test_knowledge_service_archive_and_remove(tmp_path, monkeypatch):
    monkeypatch.setattr(kb_service_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    demo = tmp_path / "data" / "demo"
    library = tmp_path / "data" / "library"
    demo.mkdir(parents=True)
    (demo / "handbook.md").write_text("# 手册\n\n普通制度。", encoding="utf-8")
    kb = KnowledgeBaseService(demo_dir=demo, library_dir=library)

    mid = str(uuid.uuid4())
    path = kb.archive_quality(
        message_id=mid,
        question="远程办公每周最多几天？",
        original_query="远程办公呢？",
        answer="每周最多两天。",
        sources=[{"source": "handbook.md", "score": 1.0, "content": "普通制度"}],
        session_id="s1",
    )
    assert path is not None
    assert kb.list_quality()[0]["message_id"] == mid
    docs, _ = kb.hybrid.search_with_scores("远程办公每周最多几天", k=3)
    assert any(d.metadata.get("source_type") == "qa_archive" for d in docs)
    assert kb.remove_quality(mid) is True
    assert kb.list_quality() == []

"""运行时知识库服务测试（离线，使用临时 Chroma 目录）"""

import src.knowledge.service as kb_service_module
from src.config import settings
from src.knowledge import KnowledgeBaseService


def _make_service(tmp_path, monkeypatch) -> KnowledgeBaseService:
    monkeypatch.setattr(kb_service_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    demo = tmp_path / "data" / "demo"
    library = tmp_path / "data" / "library"
    demo.mkdir(parents=True)
    (demo / "handbook.md").write_text(
        "# 员工手册\n\n## 远程办公\n\n每周最多两天远程办公。", encoding="utf-8"
    )
    return KnowledgeBaseService(demo_dir=demo, library_dir=library)


def test_rebuild_and_hybrid_search(tmp_path, monkeypatch):
    kb = _make_service(tmp_path, monkeypatch)
    assert kb.chunks_count >= 1
    docs, scores = kb.hybrid.search_with_scores("远程办公", k=3)
    assert docs
    assert any("远程办公" in d.page_content for d in docs)
    assert len(scores) == len(docs)


def test_ingest_files_dedup_and_remove(tmp_path, monkeypatch):
    kb = _make_service(tmp_path, monkeypatch)
    payload = "# FAQ\n\n退款三天到账。".encode()
    first = kb.ingest_files([("faq.md", payload)])
    assert first.added == ["faq.md"]
    assert first.chunks >= 2

    second = kb.ingest_files([("faq-copy.md", payload)])
    assert second.added == []
    assert second.skipped == ["faq-copy.md"]

    library_sources = [s for s in kb.list_sources() if s["scope"] == "library"]
    assert library_sources
    assert kb.remove_source(library_sources[0]["source"]) is True
    assert not any(
        s["source"] == library_sources[0]["source"] for s in kb.list_sources()
    )


def test_remove_demo_source_is_refused(tmp_path, monkeypatch):
    kb = _make_service(tmp_path, monkeypatch)
    demo_source = next(s["source"] for s in kb.list_sources() if s["scope"] == "demo")
    assert kb.remove_source(demo_source) is False


def test_ingest_folder_copies_and_rebuilds(tmp_path, monkeypatch):
    kb = _make_service(tmp_path, monkeypatch)
    import_dir = tmp_path / "data" / "imports" / "policies"
    import_dir.mkdir(parents=True)
    (import_dir / "leave.md").write_text("# 请假\n\n提前一天申请。", encoding="utf-8")
    (import_dir / "skip.exe").write_bytes(b"not supported")

    result = kb.ingest_folder(import_dir)
    assert result.added == ["leave.md"]
    assert result.failed == []
    assert any("leave" in s["name"] for s in kb.list_sources() if s["scope"] == "library")


def test_ingest_folder_outside_data_is_refused(tmp_path, monkeypatch):
    kb = _make_service(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.md").write_text("# A", encoding="utf-8")
    result = kb.ingest_folder(outside)
    assert result.failed
    assert "data/" in result.failed[0][1]


def test_ingest_files_rejects_limits(tmp_path, monkeypatch):
    kb = _make_service(tmp_path, monkeypatch)
    bad = kb.ingest_files([("bad.exe", b"x")])
    assert bad.failed and "不支持" in bad.failed[0][1]
    big = kb.ingest_files([("big.md", b"x" * (settings.max_upload_mb * 1024 * 1024 + 1))])
    assert big.failed and "MB" in big.failed[0][1]

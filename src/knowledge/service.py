"""运行时知识库服务

职责：
- 汇总 demo + library 语料，解析 → 切块 → 重建 Vector/BM25；
- 支持上传文件、导入受限文件夹、列出/删除来源；
- 与图共享同一批 retriever 对象，重建后图无需重建。

MVP 采用全量重建 + 进程内写锁；不承诺生产级并发。
"""

import hashlib
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import PROJECT_ROOT, settings
from src.ingestion import DocumentParser, EmbeddingService, TextChunker
from src.retrieval import BM25Retriever, HybridRetriever, Reranker, VectorStore
from src.utils.logger import logger

from .quality import QualityArchive

_SUPPORTED = DocumentParser.SUPPORTED_EXTENSIONS


@dataclass
class IngestResult:
    """一次入库操作的结果。"""

    added: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks: int = 0
    elapsed_ms: float = 0.0

    @property
    def summary(self) -> str:
        return (
            f"新增 {len(self.added)}，跳过 {len(self.skipped)}，"
            f"失败 {len(self.failed)}，chunks={self.chunks}，"
            f"耗时 {self.elapsed_ms:.0f}ms"
        )


class KnowledgeBaseService:
    """demo + library 知识库与索引。"""

    def __init__(
        self,
        demo_dir: Path | str | None = None,
        library_dir: Path | str | None = None,
        *,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.demo_dir = Path(demo_dir) if demo_dir is not None else PROJECT_ROOT / settings.demo_dir
        self.library_dir = Path(library_dir) if library_dir is not None else PROJECT_ROOT / settings.library_dir
        self.library_dir.mkdir(parents=True, exist_ok=True)

        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.embedder = embedder or EmbeddingService()
        self.vector_store = VectorStore(self.embedder)
        self.bm25 = BM25Retriever()
        self.hybrid = HybridRetriever(self.vector_store, self.bm25)
        self.reranker = Reranker()
        self.quality = QualityArchive(self.library_dir / "quality")
        self._lock = threading.RLock()
        self._chunks: list = []
        self.rebuild()

    # ── 重建 ───────────────────────────────────────────────────────
    def rebuild(self) -> IngestResult:
        """全量解析 demo + library 并重建索引。"""
        with self._lock:
            t0 = time.perf_counter()
            docs = []
            for directory in self._source_dirs():
                docs.extend(self.parser.parse_directory(directory))
            # 优质库路径标记：供 HybridRetriever 的 metadata filter 使用
            quality_dir = (self.library_dir / "quality").resolve()
            for doc in docs:
                source = Path(str(doc.metadata.get("source", ""))).resolve()
                doc.metadata["source_type"] = (
                    "qa_archive" if source.is_relative_to(quality_dir) else "document"
                )
            chunks = self.chunker.split(docs)
            self.vector_store.build_index(chunks)
            self.bm25.build_index(chunks)
            self._chunks = chunks
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[knowledge] 重建完成：docs={len(docs)}, chunks={len(chunks)}, "
                f"elapsed={elapsed:.0f}ms"
            )
            return IngestResult(chunks=len(chunks), elapsed_ms=elapsed)

    def _source_dirs(self) -> list[Path]:
        dirs = []
        if self.demo_dir.exists():
            dirs.append(self.demo_dir)
        elif (PROJECT_ROOT / settings.data_dir).exists():
            # demo 尚未建立时兼容旧语料
            dirs.append(PROJECT_ROOT / settings.data_dir)
        if self.library_dir.exists():
            dirs.append(self.library_dir)
        return dirs

    # ── 上传 / 导入 ────────────────────────────────────────────────
    def ingest_files(self, files: list[tuple[str, bytes]]) -> IngestResult:
        """保存上传文件到 library/uploads，按内容哈希去重，然后重建。"""
        t0 = time.perf_counter()
        result = IngestResult()
        if len(files) > settings.max_import_files:
            result.failed.append(("<batch>", f"单次最多 {settings.max_import_files} 个文件"))
            return result

        upload_dir = self.library_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        for name, data in files:
            safe_name = Path(str(name)).name
            suffix = Path(safe_name).suffix.lower()
            if suffix not in _SUPPORTED:
                result.failed.append((safe_name, f"不支持的格式 {suffix}"))
                continue
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                result.failed.append((safe_name, f"超过 {settings.max_upload_mb}MB 限制"))
                continue
            digest = hashlib.sha256(data).hexdigest()[:16]
            # 内容哈希去重：同内容不同文件名也只保留一份
            if any(upload_dir.glob(f"{digest}-*")):
                result.skipped.append(safe_name)
                continue
            destination = upload_dir / f"{digest}-{safe_name}"
            try:
                destination.write_bytes(data)
                result.added.append(safe_name)
            except OSError as e:
                result.failed.append((safe_name, str(e)))

        rebuild = self.rebuild()
        result.chunks = rebuild.chunks
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    def ingest_folder(self, folder: str | Path) -> IngestResult:
        """导入项目 data/ 下的文件夹；复制到 library/folders/<name>/。"""
        t0 = time.perf_counter()
        result = IngestResult()
        source = Path(folder).expanduser()
        try:
            resolved = source.resolve()
            data_root = (PROJECT_ROOT / "data").resolve()
            resolved.relative_to(data_root)
        except (ValueError, OSError):
            result.failed.append((str(folder), "只允许项目 data/ 下的文件夹"))
            return result
        if not resolved.is_dir():
            result.failed.append((str(folder), "文件夹不存在"))
            return result

        # 已在 library 内：无需复制，直接重建
        try:
            resolved.relative_to(self.library_dir.resolve())
            result.skipped.append(str(resolved))
            rebuild = self.rebuild()
            result.chunks = rebuild.chunks
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            return result
        except ValueError:
            pass

        destination_root = self.library_dir / "folders" / resolved.name
        count = 0
        for file_path in sorted(resolved.rglob("*")):
            if count >= settings.max_import_files:
                result.failed.append((resolved.name, f"超过 {settings.max_import_files} 个文件"))
                break
            if not file_path.is_file() or file_path.is_symlink():
                continue
            if file_path.suffix.lower() not in _SUPPORTED:
                continue
            try:
                rel = file_path.relative_to(resolved)
            except ValueError:
                continue
            digest = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
            if any(destination_root.rglob(f"{digest}-*")):
                result.skipped.append(str(rel))
                continue
            destination = destination_root / rel.parent / f"{digest}-{rel.name}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(file_path, destination)
                result.added.append(str(rel))
                count += 1
            except OSError as e:
                result.failed.append((str(rel), str(e)))

        rebuild = self.rebuild()
        result.chunks = rebuild.chunks
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result

    # ── 来源管理 ───────────────────────────────────────────────────
    def list_sources(self) -> list[dict]:
        """列出 demo + library 中的支持格式文件。"""
        rows = []
        for directory, scope in ((self.demo_dir, "demo"), (self.library_dir, "library")):
            if not directory.exists():
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in _SUPPORTED:
                    rows.append({
                        "source": str(path),
                        "name": path.name,
                        "file_type": path.suffix.lower().lstrip("."),
                        "size": path.stat().st_size,
                        "scope": scope,
                    })
        return rows

    def remove_source(self, source: str | Path) -> bool:
        """删除 library 下的来源文件并重建；demo 文件拒绝删除。"""
        path = Path(source).expanduser()
        try:
            resolved = path.resolve()
            resolved.relative_to(self.library_dir.resolve())
        except (ValueError, OSError):
            return False
        if not resolved.is_file():
            return False
        with self._lock:
            resolved.unlink()
        self.rebuild()
        logger.info(f"[knowledge] 已移除来源：{resolved}")
        return True

    # ── 优质库 ─────────────────────────────────────────────────────
    def archive_quality(
        self,
        *,
        message_id: str,
        question: str,
        original_query: str,
        answer: str,
        sources: list[dict],
        session_id: str,
        model: str | None = None,
    ) -> Path | None:
        """写入一条优质问答并重建索引。"""
        path = self.quality.archive(
            message_id=message_id,
            question=question,
            original_query=original_query,
            answer=answer,
            sources=sources,
            session_id=session_id,
            model=model or settings.deepseek_model,
        )
        if path is not None:
            self.rebuild()
        return path

    def list_quality(self) -> list[dict]:
        return self.quality.list()

    def remove_quality(self, message_id: str) -> bool:
        removed = self.quality.remove(message_id)
        if removed:
            self.rebuild()
        return removed

    @property
    def chunks_count(self) -> int:
        return len(self._chunks)

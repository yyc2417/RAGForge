"""优质问答归档

用户点赞后，把问题 / 回答 / 参考来源写成一条 Markdown 知识条目：
`data/library/quality/<message_id>.md`

- `# Q:` 作为 h1，现有切块器会把问题前缀带到每个 chunk；
- 参考来源只存短摘要，避免与原文档重复竞争；
- `message_id` 命名保证幂等；
- 文件系统即索引，不额外维护 manifest。
"""

import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logger import logger

_MESSAGE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class QualityArchive:
    """优质库 Markdown 归档。"""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def archive(
        self,
        *,
        message_id: str,
        question: str,
        original_query: str,
        answer: str,
        sources: list[dict[str, Any]],
        session_id: str,
        model: str,
        max_snippet: int = 200,
    ) -> Path | None:
        """写入归档；空问题/空回答返回 None，重复 message_id 幂等返回原路径。"""
        if not _MESSAGE_ID_RE.fullmatch(str(message_id or "")):
            logger.warning(f"[quality] 非法 message_id，拒绝归档：{message_id!r}")
            return None
        question = question.strip()
        answer = answer.strip()
        if not question or not answer:
            logger.warning("[quality] 空问题或空回答，跳过归档")
            return None

        path = self.base_dir / f"{message_id}.md"
        if path.exists():
            return path

        lines = [
            f"# Q: {question}",
            "",
            f"- 原始问题：{original_query.strip() or question}",
            f"- 归档时间：{datetime.now(UTC).isoformat(timespec='seconds')}",
            f"- 会话：{session_id}",
            f"- 消息：{message_id}",
            f"- 模型：{model}",
            "",
            "## 回答",
            "",
            answer,
            "",
            "## 参考来源",
            "",
        ]
        for i, source in enumerate(sources, start=1):
            name = self._basename(str(source.get("source", "unknown")))
            page = source.get("page_number")
            chunk = source.get("chunk_index")
            score = float(source.get("score", 0.0))
            meta = [f"`{name}`", f"score {score:.4f}"]
            if page is not None:
                meta.insert(1, f"page {page}")
            if chunk is not None:
                meta.append(f"chunk {chunk}")
            snippet = " ".join(str(source.get("content", "")).split())[:max_snippet]
            lines.append(f"{i}. " + " · ".join(meta))
            if snippet:
                lines.append(f"   > {snippet}")
            lines.append("")

        self._write_atomic(path, "\n".join(lines).rstrip() + "\n")
        logger.info(f"[quality] 已归档：{path.name}（{question[:40]}）")
        return path

    def list(self) -> list[dict[str, Any]]:
        """列出优质库条目（问题取首行 `# Q:`）。"""
        rows = []
        for path in self.base_dir.glob("*.md"):
            try:
                first_line = path.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, IndexError):
                continue
            question = first_line.removeprefix("# Q:").strip() if first_line.startswith("# Q:") else path.stem
            rows.append({
                "message_id": path.stem,
                "question": question,
                "path": str(path),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds"),
            })
        return sorted(rows, key=lambda x: x["updated_at"], reverse=True)

    def remove(self, message_id: str) -> bool:
        """删除归档文件；不存在/非法 ID 返回 False。"""
        if not _MESSAGE_ID_RE.fullmatch(str(message_id or "")):
            return False
        path = self.base_dir / f"{message_id}.md"
        if not path.is_file():
            return False
        path.unlink()
        logger.info(f"[quality] 已移除归档：{path.name}")
        return True

    @staticmethod
    def _basename(path: str) -> str:
        normalized = path.replace("\\", "/")
        return normalized.rsplit("/", 1)[-1] or "unknown"

    def _write_atomic(self, path: Path, content: str) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=self.base_dir, prefix=".quality-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

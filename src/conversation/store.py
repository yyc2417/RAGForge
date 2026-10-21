"""会话 JSON 存储

- 每个会话一个 `data/sessions/<session_id>.json` 文件
- 原子写：先写临时文件再 os.replace，避免半截文件
- 单进程锁：Streamlit 单进程场景足够；多进程后续换 SQLite
"""

import os
import re
import tempfile
import threading
from pathlib import Path

from src.config import PROJECT_ROOT, settings
from src.utils.logger import logger

from .models import FeedbackRating, Session, SessionMessage, SessionMeta, new_id, utc_now

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class ConversationStore:
    """会话持久化。"""

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else PROJECT_ROOT / settings.session_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ── 会话生命周期 ───────────────────────────────────────────────
    def create_session(self, title: str = "新会话") -> Session:
        """创建并落盘一个新会话。"""
        session = Session(session_id=new_id(), title=title or "新会话")
        with self._lock:
            self._write(session)
        return session

    def list_sessions(self) -> list[SessionMeta]:
        """按 updated_at 倒序返回历史会话。损坏文件跳过并告警。"""
        items: list[SessionMeta] = []
        with self._lock:
            for path in self.base_dir.glob("*.json"):
                try:
                    session = Session.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001 - 单个损坏文件不阻塞列表
                    logger.warning(f"[conversation] 跳过损坏会话 {path.name}: {e}")
                    continue
                items.append(SessionMeta(
                    session_id=session.session_id,
                    title=session.title,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    message_count=len(session.messages),
                ))
        return sorted(items, key=lambda x: x.updated_at, reverse=True)

    def load_session(self, session_id: str) -> Session | None:
        """读取会话；不存在或 ID 非法时返回 None。"""
        path = self._path_or_none(session_id)
        if path is None or not path.exists():
            return None
        with self._lock:
            try:
                return Session.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001 - 损坏文件返回 None，由调用方提示
                logger.warning(f"[conversation] 读取会话失败 {session_id}: {e}")
                return None

    def append_message(self, session_id: str, message: SessionMessage) -> Session | None:
        """追加消息并更新标题/时间；会话不存在返回 None。"""
        with self._lock:
            session = self.load_session(session_id)
            if session is None:
                return None
            session.messages.append(message)
            if session.title == "新会话" and message.role == "user" and message.content.strip():
                session.title = self.make_title(message.content)
            session.updated_at = utc_now()
            self._write(session)
            return session

    def update_feedback(
        self,
        session_id: str,
        message_id: str,
        feedback: FeedbackRating,
        *,
        archived: bool | None = None,
    ) -> Session | None:
        """更新 assistant 消息的反馈与归档标记。"""
        with self._lock:
            session = self.load_session(session_id)
            if session is None:
                return None
            for message in session.messages:
                if message.id == message_id and message.role == "assistant":
                    message.feedback = feedback
                    if archived is not None:
                        message.archived = archived
                    session.updated_at = utc_now()
                    self._write(session)
                    return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """删除会话文件；成功返回 True。"""
        path = self._path_or_none(session_id)
        if path is None or not path.exists():
            return False
        with self._lock:
            path.unlink()
        return True

    # ── 内部 ───────────────────────────────────────────────────────
    def _path_or_none(self, session_id: str) -> Path | None:
        if not _SESSION_ID_RE.fullmatch(str(session_id or "")):
            return None
        return self.base_dir / f"{session_id}.json"

    def _write(self, session: Session) -> None:
        """原子写：临时文件 + os.replace。"""
        path = self._path_or_none(session.session_id)
        if path is None:
            raise ValueError(f"非法 session_id: {session.session_id}")
        payload = session.model_dump_json(indent=2)
        fd, tmp_name = tempfile.mkstemp(dir=self.base_dir, prefix=".session-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @staticmethod
    def make_title(text: str, max_len: int = 24) -> str:
        """首条问题压缩为标题：折叠空白并截断。"""
        title = " ".join(str(text).split())
        return title[:max_len] + ("…" if len(title) > max_len else "")

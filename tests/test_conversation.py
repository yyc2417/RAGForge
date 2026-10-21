"""会话层单元测试（离线）

覆盖：JSON 会话落盘、列表/读取/反馈、原子写容错、多轮 condense、
clarify 消息落盘与来源提取。
"""

from langchain_core.documents import Document

from fakes import FakeLLM
from src.conversation import (
    ConversationService,
    ConversationStore,
    SessionMessage,
)
from src.generation.schemas import CondenseResult


def _seed_session(store: ConversationStore):
    session = store.create_session()
    store.append_message(
        session.session_id,
        SessionMessage(role="user", content="Docker 多阶段构建是什么？"),
    )
    store.append_message(
        session.session_id,
        SessionMessage(role="assistant", content="多阶段构建是……"),
    )
    loaded = store.load_session(session.session_id)
    assert loaded is not None
    return loaded


def test_store_roundtrip_and_title(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.create_session()
    assert session.title == "新会话"
    store.append_message(
        session.session_id,
        SessionMessage(role="user", content="  Docker   多阶段构建怎么写？  "),
    )
    loaded = store.load_session(session.session_id)
    assert loaded is not None
    assert loaded.title.startswith("Docker 多阶段构建")
    assert len(loaded.messages) == 1
    assert loaded.messages[0].content.startswith("  Docker")


def test_store_list_and_delete(tmp_path):
    store = ConversationStore(tmp_path)
    s1 = store.create_session("会话一")
    s2 = store.create_session("会话二")
    assert {m.session_id for m in store.list_sessions()} == {s1.session_id, s2.session_id}
    assert store.delete_session(s1.session_id) is True
    assert store.load_session(s1.session_id) is None


def test_store_feedback_update(tmp_path):
    store = ConversationStore(tmp_path)
    session = _seed_session(store)
    assistant = session.messages[-1]
    updated = store.update_feedback(session.session_id, assistant.id, "up")
    assert updated is not None
    assert updated.messages[-1].feedback == "up"
    assert store.update_feedback(session.session_id, "not-found", "down") is None


def test_store_skips_corrupt_file(tmp_path):
    store = ConversationStore(tmp_path)
    good = store.create_session("good")
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")
    metas = store.list_sessions()
    assert [m.session_id for m in metas] == [good.session_id]


def test_prepare_query_uses_history(tmp_path):
    store = ConversationStore(tmp_path)
    session = _seed_session(store)
    llm = FakeLLM({
        "CondenseResult": CondenseResult(
            standalone_query="Docker 多阶段构建怎么写？", reason="补全指代"
        )
    })
    service = ConversationService(store=store, llm=llm)
    standalone, reason = service.prepare_query(session.session_id, "它怎么写？")
    assert standalone == "Docker 多阶段构建怎么写？"
    assert reason == "补全指代"
    assert "Docker 多阶段构建是什么" in llm.structured_kwargs[0]["history"]


def test_prepare_query_without_history_skips_llm(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.create_session()
    llm = FakeLLM({})
    service = ConversationService(store=store, llm=llm)
    standalone, reason = service.prepare_query(session.session_id, "Docker 是什么？")
    assert standalone == "Docker 是什么？"
    assert reason is None
    assert llm.structured_calls == []


def test_prepare_query_falls_back_on_llm_error(tmp_path):
    store = ConversationStore(tmp_path)
    session = _seed_session(store)
    service = ConversationService(store=store, llm=FakeLLM({}))
    standalone, reason = service.prepare_query(session.session_id, "它怎么写？")
    assert standalone == "它怎么写？"
    assert reason is None


def test_record_assistant_state_and_sources(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.create_session()
    service = ConversationService(store=store, llm=FakeLLM({}))
    docs = [
        Document(
            page_content="多阶段构建通过多个 FROM 复用产物……" * 30,
            metadata={"source": "docker.md", "chunk_index": 2, "page_number": 3},
        )
    ]
    message = service.record_assistant_state(
        session.session_id,
        {
            "answer": "多阶段构建……",
            "documents": docs,
            "retrieval_scores": [0.032],
            "messages": ["retrieve: ..."],
            "iteration_count": 1,
            "is_faithful": True,
        },
        resolved_query="Docker 多阶段构建怎么写？",
    )
    assert message.message_type == "answer"
    assert message.resolved_query == "Docker 多阶段构建怎么写？"
    assert message.sources[0]["source"] == "docker.md"
    assert message.sources[0]["page_number"] == 3
    assert len(message.sources[0]["content"]) == 500
    loaded = store.load_session(session.session_id)
    assert loaded is not None and loaded.messages[-1].id == message.id


def test_record_clarify_message(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.create_session()
    service = ConversationService(store=store, llm=FakeLLM({}))
    message = service.record_assistant_state(
        session.session_id,
        {
            "answer": "",
            "clarification_question": "你指的是 Python 的 GIL，还是 Java 的 GC？",
            "ambiguity_type": "multiple_interpretations",
        },
    )
    assert message.message_type == "clarify"
    assert "GIL" in message.content
    assert message.sources == []

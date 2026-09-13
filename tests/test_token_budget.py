"""上下文 token 预算的离线测试（零网络）

覆盖：
1. count_tokens：cl100k_base 口径基本计数
2. fit_to_budget：按分截断（高分优先）、无分数原序裁、单篇超预算保底、
   不超预算原样返回、分数长度不齐兜底
3. 端到端接线：LLMClient.generate 与 verify_node 的真实裁剪路径
   （LLM 替换为离线 fake 模型，不发起网络请求）
"""

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import SecretStr

from fakes import FakeLLM, make_script
from src.agent.nodes.verify import verify_node
from src.config import settings
from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager
from src.utils.token_budget import count_tokens, fit_to_budget


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "t.md", "chunk_index": 0})


def test_count_tokens():
    """cl100k_base 口径：英文常见短句可精确断言，中文非空即为正。"""
    assert count_tokens("hello world") == 2
    assert count_tokens("") == 0
    assert count_tokens("这是一段中文测试文本") > 0


def test_fit_to_budget():
    """超预算时高分优先保留；无分数按原序保第一篇；单篇超预算保底 1 篇。"""
    first = _doc("甲" * 40)  # 分数低但排在前面
    second = _doc("乙" * 40)  # 分数高但排在后面
    docs = [first, second]
    scores = [0.5, 0.9]
    # 预算恰好只装得下 1 篇
    budget = count_tokens(second.page_content)

    kept = fit_to_budget(docs, budget, scores=scores)
    assert kept == [second], "应保留高分的 second，而非靠前的低分 first"

    kept = fit_to_budget(docs, budget)  # 无分数 → 原序装入，保第一篇
    assert kept == [first]

    # 单篇即超预算：保底保留 1 篇，不返回空列表
    assert fit_to_budget(docs, 1, scores=scores) == [second]

    # 不超预算原样返回；分数列表长度不齐时按原序兜底
    assert fit_to_budget(docs, 10**9, scores=scores) == docs
    assert fit_to_budget(docs, budget, scores=[0.9]) == [first]


class _RecordingChatModel(FakeListChatModel):
    """记录最近一次收到的 messages，用于断言实际拼入 prompt 的文档。"""

    last_messages: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.last_messages = messages
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_llm_client_and_verify_trim(monkeypatch):
    """端到端（离线）：预算压到 1 时只保底 1 篇，generate 与 verify 均裁剪。"""
    monkeypatch.setattr(settings, "context_token_budget", 1)
    # 占位 key：LLMClient 构造是纯本地操作，不发起网络请求（同 test_stage1 约定）
    monkeypatch.setattr(settings, "deepseek_api_key", SecretStr("sk-test-placeholder"))
    docs = [_doc(f"文档{i}：" + "内容" * 100) for i in range(4)]
    scores = [0.9, 0.8, 0.7, 0.6]

    # ── LLMClient.generate：替换底层 ChatOpenAI 为离线 fake，走真实裁剪路径 ──
    client = LLMClient()
    fake_chat = _RecordingChatModel(responses=["ok"])
    monkeypatch.setattr(client, "_llm", fake_chat)
    assert client.generate("q", docs, scores=scores) == "ok"
    prompt_text = "\n".join(str(m.content) for m in fake_chat.last_messages)
    assert docs[0].page_content in prompt_text, "最高分文档应保底保留"
    assert docs[1].page_content not in prompt_text, "其余文档应全部被裁"

    # ── verify_node：FakeLLM 记录 kwargs，断言传给 LLM 的 context 已被裁剪 ──
    fake_llm = FakeLLM(make_script())
    verify_node(
        {"query": "q", "answer": "a", "documents": docs, "retrieval_scores": scores},
        llm=fake_llm,
        prompts=PromptManager(),
    )
    context = fake_llm.structured_kwargs[0]["context"]
    assert docs[0].page_content in context
    assert docs[1].page_content not in context

"""生成层 - LLM 客户端、Prompt 模板

- LLMClient：DeepSeek（OpenAI 兼容）封装，重试 + 流式 + 结构化输出
- PromptManager：Prompt 模板集中管理
"""

from src.generation.llm_client import LLMClient
from src.generation.prompts import PromptManager

__all__ = ["LLMClient", "PromptManager"]

"""RAGForge — Agentic RAG 自适应检索增强生成系统"""

import os

# 本地优先：默认使用已缓存的 HF 模型，避免网络抖动阻塞启动。
# 首次下载模型时显式设置 RAGFORGE_HF_OFFLINE=0 或 HF_HUB_OFFLINE=0。
if os.getenv("RAGFORGE_HF_OFFLINE", "1") != "0":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__version__ = "0.1.0"

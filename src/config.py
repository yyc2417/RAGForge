"""
集中式配置管理

使用 pydantic-settings 从环境变量和 .env 文件读取配置，
提供类型安全的全局单例 settings 供所有模块使用。
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（pyproject.toml 所在目录）
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """RAGForge 全局配置，自动从环境变量和 .env 文件读取。"""

    # ── LLM（DeepSeek，OpenAI 兼容格式）────────────────────────────
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # ── Embedding ──────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── 向量存储（ChromaDB）─────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "knowledge_base"

    # ── 数据路径 ────────────────────────────────────────────────────
    data_dir: str = "data/sample"

    # ── 检索参数 ────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 3

    # ── 混合检索 + Reranker（Task 6）────────────────────────────────
    rrf_k: int = 60  # RRF 融合常数，标准值 60
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # CrossEncoder 重排序模型
    reranker_top_n: int = 5  # 重排序后保留的文档数

    # ── API 服务（Task 7）────────────────────────────────────────────
    api_host: str = "0.0.0.0"  # FastAPI 监听地址
    api_port: int = 8000  # FastAPI 监听端口

    # ── LangSmith 追踪（Task 7，可选）──────────────────────────────
    # 零代码改动即可自动记录 trace；填入 key 并置 tracing=true 即启用
    langsmith_api_key: str = ""
    langsmith_project: str = "ragforge"
    langsmith_tracing: bool = False

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
    )


# 全局单例，其他模块直接 from src.config import settings
settings = Settings()

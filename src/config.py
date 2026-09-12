"""
集中式配置管理

使用 pydantic-settings 从环境变量和 .env 文件读取配置，
提供类型安全的全局单例 settings 供所有模块使用。

安全约定：
- API key 以 SecretStr 存储，避免意外 repr/print 泄漏；
  使用时经 deepseek_key_value()/langsmith_key_value() 显式解包
- 数值参数带 Field 约束，配置错误在启动时立即暴露
"""

from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（pyproject.toml 所在目录）
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """RAGForge 全局配置，自动从环境变量和 .env 文件读取。"""

    # ── LLM（DeepSeek，OpenAI 兼容格式）────────────────────────────
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    llm_timeout: int = Field(60, gt=0)  # 单次 LLM 请求超时（秒），防止挂起阻塞状态机

    # ── Embedding ──────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── 向量存储（ChromaDB）─────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "knowledge_base"

    # ── 数据路径 ────────────────────────────────────────────────────
    data_dir: str = "data/sample"

    # ── 检索参数 ────────────────────────────────────────────────────
    chunk_size: int = Field(500, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    retrieval_k: int = Field(3, gt=0)

    # ── 混合检索 + Reranker（Task 6）────────────────────────────────
    rrf_k: int = Field(60, gt=0)  # RRF 融合常数，标准值 60
    reranker_model: str = "BAAI/bge-reranker-v2-m3"  # CrossEncoder 重排序模型
    reranker_top_n: int = Field(5, gt=0)  # 重排序后保留的文档数

    # ── API 服务（Task 7）────────────────────────────────────────────
    api_host: str = "127.0.0.1"  # 默认仅监听本机；对外部署时显式改为 0.0.0.0 并配合鉴权
    api_port: int = Field(8000, ge=1, le=65535)  # FastAPI 监听端口
    api_cors_origins: list[str] = Field(default_factory=lambda: ["*"])  # 浏览器跨域白名单

    # ── LangSmith 追踪（Task 7，可选）──────────────────────────────
    # 零代码改动即可自动记录 trace；填入 key 并置 tracing=true 即启用
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "ragforge"
    langsmith_tracing: bool = False

    @model_validator(mode="after")
    def _validate_chunk_params(self) -> "Settings":
        """交叉校验：重叠必须小于块长，否则切分行为未定义。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 chunk_size({self.chunk_size})"
            )
        return self

    def deepseek_key_value(self) -> str:
        """解包 DeepSeek API key（SecretStr → str），供启动校验与 LLM 客户端。"""
        return self.deepseek_api_key.get_secret_value()

    def langsmith_key_value(self) -> str:
        """解包 LangSmith API key（SecretStr → str）。"""
        return self.langsmith_api_key.get_secret_value()

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
    )


# 全局单例，其他模块直接 from src.config import settings
settings = Settings()

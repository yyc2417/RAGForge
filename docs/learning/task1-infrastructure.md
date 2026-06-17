# Task 1：项目基础设施搭建

## 阶段概述

本阶段是 RAGForge 项目的起点，目标是建立规范的项目骨架、配置管理系统和统一日志。在 8 个 Task 的分阶段开发中，Task 1 为后续所有模块提供基础设施支撑。

**前置依赖**：无（项目初始状态）  
**后续依赖**：Task 2-8 全部依赖本阶段的配置和日志系统

---

## 核心知识点

### 知识点 1：pydantic-settings 配置管理原理

pydantic-settings 是 Pydantic 的配置管理扩展，专为从环境变量和 `.env` 文件读取配置而设计。核心机制是 `BaseSettings` 类的自动字段映射：

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    deepseek_api_key: str = ""  # snake_case 字段名
    deepseek_base_url: str = "https://api.deepseek.com"
    chunk_size: int = 500
    
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
    )
```

**字段映射规则**：pydantic-settings 自动将 `snake_case` 字段名转换为 `SCREAMING_SNAKE_CASE` 环境变量名。例如 `deepseek_api_key` 会依次查找：
1. 环境变量 `DEEPSEEK_API_KEY`
2. `.env` 文件中的 `DEEPSEEK_API_KEY=value`
3. 字段默认值 `""`

**为什么选择 pydantic-settings**：相比 `os.getenv()` 硬编码，它提供类型验证（`int` 字段自动转换）、集中管理（所有配置在一个类中）、环境隔离（不同环境用不同 `.env` 文件）。`SettingsConfigDict` 的 `env_file` 参数指定 `.env` 路径，`env_file_encoding` 确保中文配置正确读取。

### 知识点 2：loguru vs stdlib logging

Python 标准库 `logging` 配置繁琐（需手动配置 Handler、Formatter、Logger 层级），而 loguru 采用"开箱即用 + 按需配置"的简洁模型：

```python
from loguru import logger

# 移除默认 handler，重新配置
logger.remove()

# 终端输出：彩色、简洁
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> - {message}",
)

# 文件输出：自动轮转
logger.add(
    str(LOG_DIR / "ragforge.log"),
    rotation="10 MB",      # 单文件 10MB 后自动轮转
    retention="7 days",    # 保留 7 天历史
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{line} - {message}",
)
```

**handler 模型**：loguru 启动时自带一个默认 handler（输出到 stderr），`logger.remove()` 移除它，然后用 `logger.add()` 添加自定义 handler。这种"先清空再配置"的模式避免重复日志。

**日志轮转**：`rotation="10 MB"` 自动按大小轮转（生成 `ragforge.log.1`、`ragforge.log.2`），`retention="7 days"` 自动清理过期文件。相比 stdlib 的 `RotatingFileHandler`，无需手动计算文件数量和大小。

**format 模板语法**：`<green>{time:HH:mm:ss}</green>` 用 XML 标签包裹实现彩色，`{name}` 自动填充模块名，`{line}` 填充行号。比 stdlib 的 `%` 格式化更直观。

### 知识点 3：项目骨架设计原则

RAGForge 按职责分层设计目录结构，每个包对应一个独立功能域：

```
src/
├── ingestion/     # 文档处理（解析、切分、嵌入）
├── retrieval/     # 检索（向量、BM25、混合）
├── generation/    # LLM 生成（客户端、prompt、schema）
├── agent/         # Agent 状态机（节点、图、状态）
│   └── nodes/     # 8 个独立节点
├── api/           # FastAPI 服务（路由、依赖、schema）
└── utils/         # 工具函数（日志、指标）
```

**设计原则**：
1. **单一职责**：每个包只负责一个功能域，`ingestion` 不碰检索逻辑，`retrieval` 不碰生成逻辑
2. **`__init__.py` 作为公共 API 出口**：包的 `__init__.py` 导出对外接口，隐藏内部实现。例如 `from src.ingestion import DocumentParser` 而非 `from src.ingestion.parser import DocumentParser`
3. **依赖方向单一**：上层（agent）依赖下层（retrieval/generation），下层不依赖上层，避免循环依赖

这种分层使得 Task 2-8 可以独立开发、独立测试，每个 Task 只需关注对应包的实现。

### 知识点 4：ADR（Architecture Decision Record）

ADR 是记录架构决策的标准格式，RAGForge 在 `docs/adr/001-langgraph-framework.md` 记录了选择 LangGraph 的决策。

**ADR 标准格式**：
- **标题**：简短描述决策（如"选择 LangGraph 作为 Agent 框架"）
- **状态**：proposed / accepted / deprecated / superseded
- **上下文（Context）**：面临的问题、可选方案、约束条件
- **决策（Decision）**：选择的方案及理由
- **后果（Consequences）**：决策带来的影响（正面和负面）

**为什么写 ADR**：
1. **知识沉淀**：新成员加入时能快速理解"为什么这样设计"，避免重复讨论
2. **决策追溯**：当决策需要调整时，能回溯当时的约束条件和备选方案
3. **避免遗忘**：口头讨论的决策容易遗忘，ADR 强制书面记录

Task 1 写 ADR-001 记录了选择 LangGraph 而非 LangChain Agent 的理由：LangGraph 的状态机模型更适合多轮检索、条件分支、循环改写的复杂流程。

---

## 设计模式与架构决策

**集中式配置模式**：用 `Settings` 单例替代分散的 `os.getenv()` 调用。全局 `settings = Settings()` 在模块加载时一次性读取，其他模块 `from src.config import settings` 直接使用。优势是配置变更只需改一处（`.env` 或字段默认值）。

**日志单例模式**：loguru 的 `logger` 本身就是全局单例，`logger.py` 配置后，所有模块 `from src.utils.logger import logger` 共享同一份配置。无需像 stdlib 那样每个模块 `getLogger(__name__)`。

**目录骨架先行**：Task 1 创建了所有包的占位文件（`__init__.py` + docstring），即使功能未实现也能 `import`。这让后续 Task 开发时无需修改目录结构，只需填充实现。

---

## 关键代码解读

### config.py：配置管理核心

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（pyproject.toml 所在目录）
PROJECT_ROOT = Path(__file__).parent.parent
```

**设计意图**：`PROJECT_ROOT` 用 `Path(__file__).parent.parent` 动态计算，而非硬编码路径。这样项目移动到任意目录都能正确工作。`Path` 对象比字符串更易用（支持 `/` 拼接路径）。

```python
class Settings(BaseSettings):
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 3
```

**设计意图**：所有配置都有合理默认值，即使 `.env` 缺失也能启动（API key 为空会在运行时报错而非启动时）。DeepSeek 用 OpenAI 兼容格式，所以 `base_url` 和 `api_key` 的命名与 OpenAI SDK 一致。

```python
model_config = SettingsConfigDict(
    env_file=str(PROJECT_ROOT / ".env"),
    env_file_encoding="utf-8",
)

settings = Settings()
```

**设计意图**：`model_config` 是 Pydantic v2 的配置方式（v1 用 `class Config`）。`env_file` 用绝对路径避免工作目录不同导致找不到 `.env`。`settings = Settings()` 在模块级实例化，Python 模块单例特性确保全局只有一个实例。

### logger.py：日志系统

```python
from loguru import logger
from src.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
```

**设计意图**：`LOG_DIR.mkdir(exist_ok=True)` 确保日志目录存在，`exist_ok=True` 避免目录已存在时报错。`logger.remove()` 移除默认 handler，避免与自定义 handler 重复输出。

```python
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> - {message}",
)
```

**设计意图**：终端日志用 INFO 级别（避免 DEBUG 噪声），彩色格式便于快速扫描。`{name}` 自动填充模块名（如 `src.config`），无需手动传 `logger.info("[config] ...")`。

```python
logger.add(
    str(LOG_DIR / "ragforge.log"),
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{line} - {message}",
)
```

**设计意图**：文件日志用 DEBUG 级别（完整记录便于排查），`rotation="10 MB"` 避免单文件过大，`retention="7 days"` 自动清理旧日志。`{line}` 记录行号便于定位代码。`encoding="utf-8"` 确保中文日志不乱码。

---

## 踩坑记录

### 问题 1：`.env` 文件路径找不到

**问题现象**：启动时报错 `KeyError: 'DEEPSEEK_API_KEY'`，但 `.env` 文件明明存在。

**排查过程**：
1. 打印 `os.getcwd()` 发现工作目录是 `d:\ai-test`（父目录），不是 `d:\ai-test\RAGForge`
2. 检查 `SettingsConfigDict(env_file=".env")`，发现用的是相对路径
3. 相对路径相对于 `os.getcwd()`，而非 `config.py` 所在目录

**根因**：`env_file=".env"` 在工作目录不是项目根时找不到文件。

**解决方案**：改用绝对路径 `env_file=str(PROJECT_ROOT / ".env")`，`PROJECT_ROOT = Path(__file__).parent.parent` 确保始终指向项目根目录。

### 问题 2：loguru 日志重复输出

**问题现象**：同一条日志在终端出现两次。

**排查过程**：
1. 检查 `logger.py` 发现调用了两次 `logger.add(sys.stderr, ...)`
2. 原因是 Jupyter notebook 中多次 `import` 同一模块，模块级代码重复执行

**根因**：loguru 默认有一个 stderr handler，`logger.add()` 又添加了一个，导致重复。

**解决方案**：在 `logger.add()` 前先调用 `logger.remove()` 移除默认 handler。即使模块被多次导入，`remove()` + `add()` 的组合也能确保只有一个 handler。

---

## 与其他模块的交互

**输入接口**：无（Task 1 是基础设施，不依赖其他 Task）

**输出接口**：
- `from src.config import settings, PROJECT_ROOT`：所有模块读取配置
- `from src.utils.logger import logger`：所有模块记录日志

**衔接方式**：
- Task 2 的 `TextChunker` 从 `settings.chunk_size` 读取切分参数
- Task 3 的 `VectorStore` 从 `settings.chroma_persist_dir` 读取持久化路径
- Task 4 的 `LLMClient` 从 `settings.deepseek_api_key` 读取 API key
- Task 5-8 的所有模块都通过 `logger.info()` 记录运行状态

---

## 本阶段收获总结

1. **pydantic-settings 是配置管理的最佳实践**：自动环境变量映射 + 类型验证 + `.env` 支持，比 `os.getenv()` 更健壮
2. **loguru 简化日志配置**：开箱即用 + 自动轮转 + 彩色输出，比 stdlib logging 代码量少 80%
3. **目录骨架先行降低后续开发成本**：Task 1 建立的分层结构在 Task 2-8 中无需调整，只需填充实现
4. **ADR 是架构决策的知识沉淀**：书面记录"为什么这样设计"比口头讨论更可靠，新成员能快速理解历史决策
5. **`PROJECT_ROOT` 动态计算避免路径问题**：`Path(__file__).parent.parent` 确保项目移动后仍能正确定位
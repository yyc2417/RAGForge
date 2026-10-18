# 文档处理层（Ingestion）

## 模块职责

文档处理层负责将原始文档转换为可检索的表示，为检索层提供高质量的 Document 列表。核心流程：**多格式解析 → 按标题切节 + token 预算切分 → 向量化**。

支持 MD / TXT / PDF（文本型）/ HTML（本地静态）/ DOCX / CSV / JSONL；网页 URL、OCR、图片和复杂表格不在范围内。

## 类设计

### DocumentParser：策略模式解析器

根据文件后缀自动选择解析策略。

```python
class DocumentParser:
    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".html", ".htm", ".docx", ".csv", ".jsonl"}

    def parse(self, file_path: Path) -> list[Document]:
        """解析单个文件，根据后缀分发到对应解析器。"""

    def parse_directory(self, dir_path: Path, glob_pattern: str = "**/*") -> list[Document]:
        """批量解析目录下所有支持格式的文件。"""
```

**格式矩阵**：

| 格式 | 解析方式 | 关键 metadata |
|---|---|---|
| `.md` | LangChain `TextLoader` + 标题切分 | source / file_type / title / h1-h3 |
| `.txt` | 纯文本，UTF-8/UTF-8-SIG/GBK 回退 | source / file_type / title |
| `.pdf` | PyMuPDF 逐页 | page_number（1-based） |
| `.html/.htm` | 标准库 `HTMLParser`，跳过 script/style，保留 h1-h3 | title / h1-h3 |
| `.docx` | zip + XML 解析段落与标题样式（免额外依赖） | title / h1-h3 |
| `.csv` | `csv.DictReader`，必须有 `text` 或 `content` 列 | row_index / 其他列进 metadata |
| `.jsonl` | 逐行 JSON，`text`/`content` 必填；坏行跳过 | row_index / 标量字段进 metadata |

**CSV/JSONL 约定**：

- CSV：表头必须有 `text` 或 `content`；`id` / `title` 可选；其他列进 metadata；缺列直接报错并列出实际列名。
- JSONL：每行一个对象；`text` 或 `content` 必填；`id` / `title` 可选；坏行记录日志后跳过。
- 这两类结构化文件既可用于知识库导入，也可作为评测数据；评测 adapter 单独处理，不混入 demo 库。

### TextChunker：token 预算切分 + 标题上下文

两段式切分：先经 `MarkdownHeaderTextSplitter` 按标题切节，再用 `RecursiveCharacterTextSplitter.from_tiktoken_encoder`（cl100k_base）按 **token** 预算切块。每个 chunk：

- metadata 携带 `h1/h2/h3` 标题层级 + `chunk_index`（按 source 连续编号，供向量库确定性 ID 使用）
- 正文前缀拼入标题路径（如 `机器学习 FAQ > 过拟合`），使标题词可被向量与 BM25 同时命中
- HTML/DOCX 解析器放在 `doc.metadata` 的 h1-h3 会与 Markdown header splitter 的节标题合并，再生成正文前缀
- 切分**确定性**：同语料两次切分结果逐字节一致（向量库确定性 ID 的前提）

```python
class TextChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """参数单位为 token（cl100k 估算），空时读 settings。"""

    def split(self, documents: list[Document]) -> list[Document]:
        """按标题切节 → 按 token 切块 → 注入标题前缀与 metadata。"""
```

**为什么按 token 而非字符**：embedding 模型的编码窗口有限（all-MiniLM-L6-v2 为 256 token，中文约 1 字 1 token）。按字符切 500 字符的中文长块，后半段会被编码器**静默截断**——约一半内容从未进入向量。220 token 预留了安全余量（标题前缀约占 10~20 token）。

### EmbeddingService：线程安全单例

封装 `HuggingFaceEmbeddings`，模型加载成本高（首次约 1-2 秒），全局共享避免重复加载。

```python
class EmbeddingService:
    _instance: "EmbeddingService | None" = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str | None = None):
        """线程安全单例，__new__ 内完成模型加载；
        model_name 与已加载实例不一致时 raise（配置错误不再静默吞掉）。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成文档向量。"""

    def embed_query(self, text: str) -> list[float]:
        """生成单条查询向量。"""

    @property
    def embedding_function(self) -> HuggingFaceEmbeddings:
        """暴露底层实例，供 ChromaDB 直接使用。"""
```

## 数据流图

```mermaid
graph LR
    A[用户文档] --> B[DocumentParser]
    B --> C[TextChunker]
    C --> D[EmbeddingService]
    D --> E[向量索引/检索层]

    B -.parse.-> B1[Markdown: TextLoader]
    B -.parse.-> B2[PDF: PyMuPDF]

    C -.split.-> C1[MarkdownHeaderTextSplitter 按标题切节]
    C -.split.-> C2[from_tiktoken_encoder 220/30 tokens]
    C -.metadata.-> C3[source + chunk_index + h1/h2/h3]

    D -.embed.-> D1[all-MiniLM-L6-v2]
```

## 设计模式

### 策略模式（Parser）

`DocumentParser.parse()` 根据文件后缀（`.md` / `.pdf`）分发到不同的解析实现：
- `_parse_markdown()`：使用 LangChain TextLoader，metadata 补充 `file_type="markdown"`
- `_parse_pdf()`：使用 PyMuPDF 逐页提取，metadata 含 `page_number`（1-based）

**扩展性**：新增格式只需在 `SUPPORTED_EXTENSIONS` 添加后缀，并实现对应的 `_parse_xxx()` 方法。

### 单例模式（Embedder）

`EmbeddingService` 采用 `__new__` + `threading.Lock` 实现线程安全单例：
- **动机**：模型加载耗时（1-2 秒），内存占用高（约 80MB）
- **实现**：`_lock` 保护下检查 `_instance`，首次创建后全局复用；传入不同 `model_name` 直接报错
- **暴露接口**：`embedding_function` 属性供 ChromaDB 直接注入

## 错误处理

### parse_directory：单文件异常隔离

批量解析时，单个文件的异常不会影响其他文件的处理（loguru 记录错误详情，继续处理后续文件）。

## 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `chunk_size` | 220 | 单个 chunk 的最大 token 数（cl100k 估算，需小于 embedding 窗口） |
| `chunk_overlap` | 30 | 相邻 chunk 的重叠 token 数（必须小于 chunk_size） |
| `embedding_model` | all-MiniLM-L6-v2 | HuggingFace Embedding 模型 |
| `embedding_device` | cpu | 推理设备（cpu / cuda） |
| `embedding_batch_size` | 32 | 批量编码大小 |
| `demo_dir` | data/demo | UI 演示多格式语料 |
| `library_dir` | data/library | 运行时上传 / 文件夹导入 |
| `max_import_files` | 50 | 单次最多导入文件数 |
| `max_upload_mb` | 20 | 单文件大小上限 |

配置通过 `pydantic-settings` 从 `.env` 文件和环境变量读取，全局单例 `settings` 供各模块使用；`chunk_overlap < chunk_size` 由 model_validator 在启动时校验。

## 运行时知识库

`src/knowledge/service.py` 的 `KnowledgeBaseService` 负责：

- 汇总 `demo + library` 语料，解析 → 切块 → 重建 Vector/BM25；
- 上传文件按内容哈希去重，保存到 `library/uploads/`；
- 文件夹导入只允许项目 `data/` 下路径，复制到 `library/folders/<name>/`；
- 列出/删除来源（demo 来源只读）；
- 与图共享同一批 retriever 对象，重建后图无需重建。

MVP 采用全量重建 + 进程内写锁；不承诺生产级并发。

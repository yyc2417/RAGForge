# Task 2：文档处理层（Ingestion）

## 阶段概述

本阶段实现 RAG 系统的数据入口——文档处理管道。目标是将多格式原始文档（Markdown/PDF）转换为统一的向量表示并存入索引。在 8 个 Task 中，Task 2 是数据处理的基础，后续检索、生成、Agent 都依赖本阶段输出的文档向量。

**前置依赖**：Task 1（配置管理、日志系统）  
**后续依赖**：Task 3（检索层依赖文档向量）、Task 4（基础 RAG 管道）

---

## 核心知识点

### 知识点 1：策略模式（Strategy Pattern）

策略模式定义一系列算法（解析策略），把它们封装并使可以相互替换。`DocumentParser.parse()` 根据文件后缀动态选择解析算法：

```python
class DocumentParser:
    """多格式文档解析器，根据文件后缀自动选择解析策略。"""
    
    SUPPORTED_EXTENSIONS = {".md", ".pdf"}
    
    def parse(self, file_path: Path) -> list[Document]:
        """解析单个文件，根据后缀分发到对应解析器。"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}（支持: {self.SUPPORTED_EXTENSIONS}）")
        
        if suffix == ".md":
            return self._parse_markdown(file_path)
        else:  # .pdf
            return self._parse_pdf(file_path)
```

**开闭原则的体现**：对扩展开放（新增格式只需添加 `_parse_xxx` 方法和 `SUPPORTED_EXTENSIONS` 条目），对修改关闭（无需改动 `parse()` 主逻辑）。每种解析策略封装在独立方法中（`_parse_markdown` / `_parse_pdf`），互不干扰。

**为什么用策略模式**：相比 `if-else` 硬编码在 `parse()` 中，策略模式让每种格式的解析逻辑独立可测试。未来新增 `.docx` 格式时，只需添加 `_parse_docx()` 方法，不影响现有 Markdown/PDF 解析。

### 知识点 2：RecursiveCharacterTextSplitter 原理

LangChain 的 `RecursiveCharacterTextSplitter` 按分隔符优先级递归切分文本，优先在语义边界（段落/句子）切分：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
            length_function=len,
        )
```

**分隔符优先级**：`separators` 列表定义了切分优先级：
1. `\n\n`（段落边界）→ 优先在段落间切分
2. `\n`（行边界）→ 段落内按行切分
3. `。` / `！` / `？`（中文句子边界）
4. `.` / `!` / `?`（英文句子边界）
5. ` `（空格，最后手段）

**递归切分逻辑**：先用第一个分隔符切分，如果某 chunk 仍超过 `chunk_size`，则用下一个分隔符递归切分。这样确保 chunk 尽量在语义边界断开，而非任意位置。

**chunk_overlap 的作用**：`chunk_overlap=50` 让相邻 chunk 有 50 字符重叠。这保证了语义连续性——如果关键信息恰好在 chunk 边界被切断，重叠部分能保留上下文。例如一个长句被切分到两个 chunk，重叠部分包含句子的后半段和下一句的前半段。

### 知识点 3：线程安全单例模式

单例模式确保一个类只有一个实例，线程安全版本用 `__new__` + `threading.Lock` + double-check：

```python
import threading
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingService:
    """Embedding 服务，封装 HuggingFaceEmbeddings。
    
    线程安全单例模式：模型加载成本高（首次约 1-2 秒），
    全局共享同一实例，避免重复加载。
    """
    
    _instance: "EmbeddingService | None" = None
    _lock = threading.Lock()
    
    def __new__(cls, model_name: str | None = None):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._model_name = model_name or settings.embedding_model
                logger.info(f"[embedder] 加载模型: {instance._model_name}")
                instance._model = HuggingFaceEmbeddings(
                    model_name=instance._model_name,
                    model_kwargs={"device": "cpu"},
                )
                logger.info("[embedder] 模型加载完成")
                cls._instance = instance
        return cls._instance
    
    def __init__(self, model_name: str | None = None):
        # 初始化已在 __new__ 的锁内完成，此处无需操作
        pass
```

**为什么用 `__new__` 而非 `__init__`**：`__new__` 控制实例创建过程，可以在创建前加锁。`__init__` 在实例已创建后调用，无法阻止多线程同时创建多个实例。

**double-check locking**：虽然代码中只有一层 `if cls._instance is None`，但 `with cls._lock` 已经提供了互斥保护。完整的 double-check 会在锁外先检查一次（快速路径），锁内再检查一次（慢速路径）。这里简化为锁内单次检查，因为模型加载是重型操作，锁竞争不是性能瓶颈。

**为什么 Embedding 需要单例**：HuggingFace 模型加载成本高（首次约 1-2 秒，从磁盘读取模型权重到内存）。如果每次调用都创建新实例，10 次调用就要加载 10 次模型。单例确保全局共享同一模型实例。

### 知识点 4：HuggingFace Embeddings 与 sentence-transformers

`sentence-transformers` 库提供预训练的文本嵌入模型，`all-MiniLM-L6-v2` 是轻量级模型（384 维向量）：

```python
from langchain_huggingface import HuggingFaceEmbeddings

instance._model = HuggingFaceEmbeddings(
    model_name=instance._model_name,
    model_kwargs={"device": "cpu"},
)
```

**模型选择**：`all-MiniLM-L6-v2` 是 sentence-transformers 系列中最快的模型之一：
- 向量维度：384（相比 768/1024 维的大模型，存储和计算成本更低）
- 模型大小：约 80MB（下载快，适合本地部署）
- 性能：在 STS（语义文本相似度）任务上表现良好，适合 RAG 检索场景

**`embed_documents` vs `embed_query`**：
- `embed_documents(texts: list[str])`：批量生成文档向量，用于构建索引。接受列表输入，内部批处理提升效率
- `embed_query(text: str)`：生成单条查询向量，用于检索时。单条输入，返回单个向量

两者使用同一模型，但 API 设计区分了批量和单条场景，便于底层优化（批处理 vs 低延迟）。

---

## 设计模式与架构决策

**策略模式（DocumentParser）**：根据文件后缀分发到不同解析策略，符合开闭原则。新增格式只需添加新方法，无需修改主逻辑。

**单例模式（EmbeddingService）**：模型加载成本高，全局共享同一实例。线程安全设计确保多线程环境下不会重复加载。

**封装模式（TextChunker）**：封装 LangChain 的 `RecursiveCharacterTextSplitter`，为每个 chunk 追加 `chunk_index` 元数据。这种封装隔离了第三方库的变化，如果未来换用其他切分器，只需修改 `TextChunker` 内部实现。

**元数据传递设计**：parser 为每个 Document 的 `metadata` 填充 `source`（文件路径）和 `file_type`（文件类型），chunker 保留这些元数据并追加 `chunk_index`。这种设计让检索结果能追溯到原始文件和位置。

---

## 关键代码解读

### parser.py：文档解析器

```python
def _parse_markdown(self, file_path: Path) -> list[Document]:
    """解析 Markdown 文件（使用 LangChain TextLoader）"""
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    # 补充 file_type 元数据
    for doc in docs:
        doc.metadata["file_type"] = "markdown"
    logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 个文档段")
    return docs
```

**设计意图**：Markdown 文件用 LangChain 的 `TextLoader` 加载，它自动处理编码和换行。`encoding="utf-8"` 确保中文内容正确读取。加载后补充 `file_type` 元数据，便于下游区分文档来源。

```python
def _parse_pdf(self, file_path: Path) -> list[Document]:
    """解析 PDF 文件（使用 PyMuPDF/fitz），逐页提取"""
    import fitz  # PyMuPDF
    
    docs = []
    with fitz.open(str(file_path)) as pdf:
        for page_num, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:  # 跳过空白页
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_type": "pdf",
                        "page_number": page_num + 1,  # 1-based
                    }
                ))
    logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 页")
    return docs
```

**设计意图**：PDF 用 PyMuPDF（`fitz`）逐页提取文本，每页生成一个 Document。`page.get_text("text")` 提取纯文本（不含格式），`.strip()` 去除前后空白。`if text:` 跳过空白页（扫描版 PDF 的空白页）。`page_number` 用 1-based 索引（符合人类阅读习惯）。

```python
def parse_directory(self, dir_path: Path, glob_pattern: str = "**/*") -> list[Document]:
    """批量解析目录下所有支持格式的文件。"""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        logger.warning(f"[parser] 目录不存在: {dir_path}")
        return []
    
    all_docs = []
    for file_path in sorted(dir_path.glob(glob_pattern)):
        if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
            try:
                docs = self.parse(file_path)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"[parser] 解析失败 {file_path}: {e}")
    
    logger.info(f"[parser] 目录 {dir_path} 共解析 {len(all_docs)} 个文档段")
    return all_docs
```

**设计意图**：`parse_directory` 批量处理整个目录，`glob_pattern="**/*"` 递归匹配所有文件。`sorted()` 确保文件处理顺序稳定（便于调试）。`try-except` 捕获单个文件解析失败，不影响其他文件。`extend()` 将每个文件的 Document 列表合并到总列表。

### chunker.py：文本切分器

```python
def split(self, documents: list[Document]) -> list[Document]:
    """切分文档列表，保留原始 metadata 并追加 chunk_index。"""
    chunks = self._splitter.split_documents(documents)
    
    # 为每个 chunk 追加 chunk_index（按 source 分组编号）
    source_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        idx = source_counter.get(source, 0)
        chunk.metadata["chunk_index"] = idx
        source_counter[source] = idx + 1
    
    logger.info(
        f"[chunker] {len(documents)} 个文档 → {len(chunks)} 个 chunks"
        f"（chunk_size={self._chunk_size}, overlap={self._chunk_overlap}）"
    )
    return chunks
```

**设计意图**：`split_documents()` 是 LangChain 的 API，它保留原始 Document 的 `metadata` 并切分 `page_content`。切分后为每个 chunk 追加 `chunk_index`（按 `source` 分组编号），便于追溯"这个 chunk 是原始文档的第几个片段"。`source_counter` 字典按文件分组计数，确保同一文件的 chunk 编号连续。

### embedder.py：Embedding 服务

```python
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    """批量生成文档向量。"""
    return self._model.embed_documents(texts)

def embed_query(self, text: str) -> list[float]:
    """生成单条查询向量。"""
    return self._model.embed_query(text)

@property
def embedding_function(self) -> HuggingFaceEmbeddings:
    """暴露底层 HuggingFaceEmbeddings 实例，供 ChromaDB 直接使用。"""
    return self._model
```

**设计意图**：`embed_documents` 和 `embed_query` 是对底层模型的简单封装，保持 API 一致性。`embedding_function` 属性暴露底层实例，供 Task 3 的 ChromaDB 直接用作 `embedding_function` 参数（ChromaDB 需要直接调用嵌入函数）。

---

## 踩坑记录

### 问题 1：PDF 解析提取空文本

**问题现象**：某些 PDF 文件解析后 `len(docs) == 0`，但文件明明有内容。

**排查过程**：
1. 用 PDF 阅读器打开文件确认有文本
2. 打印 `page.get_text("text")` 发现返回空字符串
3. 检查 PDF 属性发现是"扫描版"（图片 PDF），不是"文本版"

**根因**：PyMuPDF 的 `get_text("text")` 只能提取文本层 PDF 的内容，扫描版 PDF（图片）需要 OCR。

**解决方案**：当前实现用 `if text:` 跳过空白页，扫描版 PDF 会被完全跳过。未来可集成 OCR（如 `pytesseract`）处理扫描版，但 Task 2 阶段只支持文本版 PDF。

### 问题 2：chunk_overlap 导致重复检索

**问题现象**：检索结果中出现多个内容几乎相同的 chunk。

**排查过程**：
1. 检查检索结果发现 top-3 的 chunk 内容高度重叠
2. 查看 `chunk_index` 发现是连续编号（如 5、6、7）
3. 检查 `chunk_overlap=50` 配置

**根因**：`chunk_overlap=50` 让相邻 chunk 有 50 字符重叠，如果关键信息在重叠区域，多个 chunk 都会匹配。

**解决方案**：这是设计权衡而非 bug。`chunk_overlap` 保证语义连续性（避免关键信息被切断），但会导致检索结果重叠。可以在检索后加去重逻辑（如 Task 6 的 RRF 融合用文档内容前 80 字符去重），或调小 `chunk_overlap`。

---

## 与其他模块的交互

**输入接口**：
- `settings.chunk_size` / `settings.chunk_overlap`：从 Task 1 配置读取切分参数
- `settings.embedding_model`：从 Task 1 配置读取模型名
- `data/sample/*.md` / `*.pdf`：原始文档文件

**输出接口**：
- `DocumentParser.parse_directory(dir_path)` → `list[Document]`：供 Task 3 构建索引
- `TextChunker.split(documents)` → `list[Document]`：供 Task 3 构建索引
- `EmbeddingService.embed_documents(texts)` → `list[list[float]]`：供 Task 3 生成向量
- `EmbeddingService.embedding_function` → `HuggingFaceEmbeddings`：供 Task 3 的 ChromaDB 直接使用

**衔接方式**：
- Task 3 的 `VectorStore.build_index(documents)` 接收 chunker 输出的 Document 列表
- Task 3 的 `VectorStore` 注入 `EmbeddingService` 实例，用 `embedding_function` 生成向量
- Task 4 的 `build_rag_pipeline()` 串联 parser → chunker → embedder → vector_store

---

## 本阶段收获总结

1. **策略模式让多格式支持优雅可扩展**：新增格式只需添加方法，无需修改主逻辑，符合开闭原则
2. **RecursiveCharacterTextSplitter 的语义切分优于固定长度**：按段落/句子边界切分，配合 `chunk_overlap` 保证语义连续性
3. **单例模式在重型模型场景必不可少**：Embedding 模型加载成本高，全局共享避免重复加载，线程安全设计防止并发问题
4. **元数据传递设计让检索结果可追溯**：`source` / `file_type` / `chunk_index` 元数据贯穿整个管道，便于调试和分析
5. **封装第三方库隔离变化风险**：`TextChunker` 封装 LangChain API，未来换用其他切分器只需修改内部实现
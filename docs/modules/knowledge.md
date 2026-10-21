# 运行时知识库（Knowledge）

## 模块职责

运行时知识库把 demo 语料与用户导入语料统一成同一套 `Document → chunk` 表示，并维护 live 索引：

- 汇总 `data/demo/` 与 `data/library/`，解析、切块、全量重建 Vector + BM25；
- 上传文件、导入受限文件夹、列出来源、删除来源；
- 承接优质库归档与移除，归档后重建索引；
- 与 Agent 图共享同一批 retriever 对象，重建后无需重建图。

评测索引走独立配置（`chroma_db_eval` / `knowledge_base_eval`），避免上传或优质库重建时按 chunk 数不匹配误清评测索引。

## 目录布局

| 目录 | 用途 | Git |
|---|---|---|
| `data/demo/` | 多格式通用 demo，UI 演示 | 跟踪 |
| `data/library/uploads/` | 上传文件，文件名带内容哈希前缀 | 忽略 |
| `data/library/folders/<name>/` | 受限文件夹导入副本 | 忽略 |
| `data/library/quality/` | 优质问答 Markdown | 忽略 |
| `data/feedback/hard_cases.jsonl` | 点踩难例候选池 | 忽略 |
| `data/sessions/` | 会话 JSON | 忽略 |
| `data/public_eval/` | 公开集下载、转换与本地语料 | 忽略 |

## KnowledgeBaseService

```python
class KnowledgeBaseService:
    def __init__(self, demo_dir=None, library_dir=None, *, embedder=None):
        """解析 demo + library 并完成首次全量重建。"""

    def rebuild(self) -> IngestResult:
        """全量解析 → 标记 source_type → 切块 → Vector/BM25 重建。"""

    def ingest_files(self, files: list[tuple[str, bytes]]) -> IngestResult:
        """保存上传文件到 library/uploads，按内容哈希去重后重建。"""

    def ingest_folder(self, folder: str | Path) -> IngestResult:
        """导入项目 data/ 下的文件夹到 library/folders/<name>/。"""

    def list_sources(self) -> list[dict]:
        """列出 demo + library 中的支持格式文件。"""

    def remove_source(self, source: str | Path) -> bool:
        """删除 library 下的单个来源并重建；demo 文件拒绝删除。"""

    def archive_quality(self, **kwargs) -> Path | None:
        """写入一条优质问答并重建。"""

    def list_quality(self) -> list[dict]: ...
    def remove_quality(self, message_id: str) -> bool:
        """删除归档并重建，恢复基线检索。"""
```

### 重建与写锁

- `rebuild()` 持 `threading.RLock`，按 demo → library 顺序解析；
- 每次重建先删旧 collection 再写入，保证 chunk 集合与当前文件一致；
- 重建是同步全量操作，MVP 不承诺生产级并发；
- `chunks_count` 暴露当前 chunk 数，供启动日志与健康检查使用。

### 上传去重

- 文件名取 `Path(name).name`，丢弃客户端路径；
- 只接受 `DocumentParser.SUPPORTED_EXTENSIONS`；
- 单文件受 `max_upload_mb`（默认 20MB）限制，单批受 `max_import_files`（默认 50）限制；
- SHA-256 前 16 位作为内容指纹，落盘名 `<digest>-<safe_name>`；
- 同内容不同文件名只保留一份，重复项进入 `skipped`。

### 文件夹导入

- 入参必须 `resolve()` 后位于项目 `data/` 下，拒绝路径穿越；
- 跳过符号链接、非普通文件和不支持的格式；
- 目标目录为 `library/folders/<目录名>/`，保持相对路径；
- 已在 library 内的路径不复制，直接触发重建；
- 超过单批文件数上限时停止并记录错误。

### 来源管理

- `list_sources()` 返回 `source / name / file_type / size / scope`，`scope` 为 `demo` 或 `library`；
- `remove_source()` 只允许删除 library 下文件，删除后立即重建；
- demo 文件只读，避免演示语料被误删。

## 优质库与索引

`QualityArchive` 把点赞消息写成 `data/library/quality/<message_id>.md`：

- `# Q:` 作为 h1，切块器会把问题前缀带到每个 chunk；
- 参考来源只保存短摘要，不复制全文；
- `message_id` 命名保证幂等，重复归档返回原路径；
- 重建时 quality 目录下文档标记 `source_type=qa_archive`，供 `HybridRetriever` 的优质通道过滤。

移除归档删除对应 Markdown 并重建；`KnowledgeBaseService` 不修改会话文件，`archived` 标记由 `ConversationStore` 负责。

## 难例候选池

`HardCaseStore` 维护 `data/feedback/hard_cases.jsonl`：

- 按 `message_id` 去重，状态为 `candidate / triaged / promoted / dismissed`；
- 只记录候选，不自动并入正式评测集；
- 晋升由 `scripts/promote_hard_cases.py` 人工补标签后完成。

## 失败与边界

| 场景 | 行为 |
|---|---|
| 不支持的扩展名 | 记入 `failed`，其余文件继续处理 |
| 超过大小 / 数量限制 | 记入 `failed`，不中断批次 |
| 文件夹越界 / 不存在 | 返回失败结果，不复制任何文件 |
| 坏 CSV / JSONL | 解析器报错或跳过坏行，日志可定位 |
| 重复内容 | 记入 `skipped`，不重复入库 |
| demo 文件删除 | 返回 `False`，文件保持不变 |

边界：不做网页抓取、OCR、图片、复杂表格/公式；不做增量索引与多进程并发写入。

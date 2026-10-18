# 格式支持说明

## 支持矩阵

| 格式 | 扩展名 | 解析器 | 标题 / 页码 | 备注 |
|---|---|---|---|---|
| Markdown | `.md` | LangChain TextLoader + 标题切分 | h1-h3 | 首选格式 |
| 纯文本 | `.txt` | 标准库读取 | 文件名作标题 | UTF-8 / UTF-8-SIG / GBK 回退 |
| PDF | `.pdf` | PyMuPDF 逐页 | page_number | 仅文本型，扫描件不支持 |
| HTML | `.html` / `.htm` | 标准库 HTMLParser | h1-h3 + `<title>` | 仅本地静态文件，不抓 URL |
| DOCX | `.docx` | zip + XML | h1-h3 | 段落/标题/文本表格；不解析图片 |
| CSV | `.csv` | csv.DictReader | 行级 metadata | 必须有 `text` 或 `content` 列 |
| JSONL | `.jsonl` | 逐行 JSON | 行级 metadata | 必须有 `text` 或 `content` 字段 |

## CSV 约定

- 表头必须有 `text` 或 `content`，否则直接报错并列出实际列名；
- `id` / `title` 可选；
- 其他列写入 `metadata`；
- 空行/空正文跳过；
- `row_index` 从 0 开始。

示例：

```csv
text,title,category
杭州西湖适合清晨骑行,杭州西湖,旅行
```

## JSONL 约定

- 每行一个 JSON 对象；
- `text` 或 `content` 必填；
- `id` / `title` 可选；
- 其他标量字段写入 metadata；
- 坏行记录日志后跳过，不阻塞整个文件。

示例：

```json
{"text": "退款三天到账", "title": "退款", "category": "售后"}
```

## 统一 metadata

解析后至少包含：

```text
source / file_type / title
page_number（PDF）
h1 / h2 / h3（Markdown / HTML / DOCX）
row_index（CSV / JSONL）
```

`TextChunker` 会把标题层级拼入正文前缀，使标题词进入向量与 BM25。

## 不支持

- 网页 URL 抓取；
- OCR、扫描 PDF、图片文字；
- 复杂表格结构、公式、图表；
- DOCX 图片 / 批注 / 修订记录；
- CSV/JSONL 的自动列映射 UI（首版固定列名）。

## 来源管理

- demo：`data/demo/`，只读；
- library：`data/library/`，可上传、导入文件夹、删除；
- 运行时数据不提交仓库（`.gitignore`）。

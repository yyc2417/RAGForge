"""文档解析器：多格式统一转 Document

支持：
- Markdown / TXT / PDF（文本型）/ HTML（本地静态）/ DOCX / CSV / JSONL

设计约束：
- 所有格式最终返回统一的 `Document` 列表，metadata 至少包含
  source / file_type，可选 title / page_number / h1-h3 / row_index。
- 不抓网页、不做 OCR、不解析图片；HTML 仅本地文件，DOCX 仅段落/标题/文本表格。
- HTML/DOCX 尽量保留标题层级，供现有 TextChunker 的标题前缀复用。
"""

import csv
import io
import json
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from src.utils.logger import logger

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _read_text(path: Path) -> str:
    """按 UTF-8 / UTF-8-SIG / GBK 回退读取文本。"""
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


class _HTMLSectionParser(HTMLParser):
    """极简 HTML 正文抽取：跳过 script/style，保留 h1-h3 与段落。"""

    _SKIP_TAGS = {"script", "style", "noscript", "template"}
    _BLOCK_TAGS = {
        "p", "div", "section", "article", "li", "br", "tr", "table",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.sections: list[tuple[dict[str, str], str]] = []
        self._headings: dict[str, str] = {}
        self._buffer: list[str] = []
        self._heading_tag: str | None = None
        self._heading_text: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3"):
            self._flush_section()
            self._heading_tag = tag
            self._heading_text = []
        elif tag in self._BLOCK_TAGS:
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3") and self._heading_tag == tag:
            heading = " ".join("".join(self._heading_text).split())
            if heading:
                self._headings = {
                    k: v for k, v in self._headings.items()
                    if k not in ("h1", "h2", "h3") or k < tag
                }
                self._headings[tag] = heading
            self._heading_tag = None
            self._heading_text = []
        elif tag in self._BLOCK_TAGS:
            self._buffer.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if self._heading_tag:
            self._heading_text.append(data)
        else:
            self._buffer.append(data)

    def close(self) -> None:
        super().close()
        self._flush_section()

    def _flush_section(self) -> None:
        text = "".join(self._buffer).strip()
        self._buffer = []
        if text:
            self.sections.append((dict(self._headings), text))


class DocumentParser:
    """多格式文档解析器，根据文件后缀自动选择解析策略。"""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".html", ".htm", ".docx", ".csv", ".jsonl"}

    def parse(self, file_path: Path) -> list[Document]:
        """解析单个文件，根据后缀分发到对应解析器。"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}（支持: {self.SUPPORTED_EXTENSIONS}）")

        if suffix == ".md":
            return self._parse_markdown(file_path)
        if suffix == ".txt":
            return self._parse_text(file_path)
        if suffix == ".pdf":
            return self._parse_pdf(file_path)
        if suffix in (".html", ".htm"):
            return self._parse_html(file_path)
        if suffix == ".docx":
            return self._parse_docx(file_path)
        if suffix == ".csv":
            return self._parse_csv(file_path)
        return self._parse_jsonl(file_path)

    # ── 基础格式 ───────────────────────────────────────────────────
    def _parse_markdown(self, file_path: Path) -> list[Document]:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "markdown"
            doc.metadata.setdefault("title", file_path.stem)
        logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 个文档段")
        return docs

    def _parse_text(self, file_path: Path) -> list[Document]:
        text = _read_text(file_path).strip()
        docs = [] if not text else [Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_type": "text",
                "title": file_path.stem,
            },
        )]
        logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 个文档段")
        return docs

    def _parse_pdf(self, file_path: Path) -> list[Document]:
        import fitz  # PyMuPDF

        docs = []
        with fitz.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(file_path),
                            "file_type": "pdf",
                            "title": file_path.stem,
                            "page_number": page_num + 1,
                        },
                    ))
        logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 页")
        return docs

    # ── HTML ───────────────────────────────────────────────────────
    def _parse_html(self, file_path: Path) -> list[Document]:
        parser = _HTMLSectionParser()
        parser.feed(_read_text(file_path))
        parser.close()
        title = " ".join(parser.title.split()) or file_path.stem
        docs = []
        for headings, text in parser.sections:
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_type": "html",
                    "title": title,
                    **headings,
                },
            ))
        logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 个文档段")
        return docs

    # ── DOCX（zip + XML，免额外依赖）───────────────────────────────
    def _parse_docx(self, file_path: Path) -> list[Document]:
        try:
            with zipfile.ZipFile(file_path) as zf:
                document_xml = zf.read("word/document.xml")
                try:
                    styles_xml = zf.read("word/styles.xml")
                except KeyError:
                    styles_xml = b""
        except (zipfile.BadZipFile, KeyError) as e:
            raise ValueError(f"DOCX 解析失败（不是有效 docx）: {file_path.name}") from e

        style_headings = self._docx_style_headings(styles_xml)
        root = ElementTree.fromstring(document_xml)
        body = root.find(f"{_W_NS}body")
        if body is None:
            return []

        headings: dict[str, str] = {}
        sections: list[tuple[dict[str, str], list[str]]] = []
        current: tuple[dict[str, str], list[str]] = ({}, [])
        for para in body.iter(f"{_W_NS}p"):
            text = "".join(node.text or "" for node in para.iter(f"{_W_NS}t")).strip()
            if not text:
                continue
            style_id = ""
            p_style = para.find(f"{_W_NS}pPr/{_W_NS}pStyle")
            if p_style is not None:
                style_id = p_style.get(f"{_W_NS}val", "")
            level = style_headings.get(style_id)
            if level:
                if current[1]:
                    sections.append(current)
                headings = {k: v for k, v in headings.items() if k < level}
                headings[level] = text
                current = (dict(headings), [])
            else:
                current[1].append(text)
        if current[1]:
            sections.append(current)

        title = file_path.stem
        docs = []
        for section_headings, paragraphs in sections:
            content = "\n".join(paragraphs).strip()
            if not content:
                continue
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "file_type": "docx",
                    "title": title,
                    **section_headings,
                },
            ))
        logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 个文档段")
        return docs

    @staticmethod
    def _docx_style_headings(styles_xml: bytes) -> dict[str, str]:
        """styleId → h1/h2/h3（按样式名 heading 1 / 标题 1 识别）。"""
        if not styles_xml:
            return {}
        try:
            root = ElementTree.fromstring(styles_xml)
        except ElementTree.ParseError:
            return {}
        mapping: dict[str, str] = {}
        for style in root.iter(f"{_W_NS}style"):
            style_id = style.get(f"{_W_NS}styleId", "")
            name_node = style.find(f"{_W_NS}name")
            name = (name_node.get(f"{_W_NS}val", "") if name_node is not None else "").lower()
            for level in ("1", "2", "3"):
                if name in (f"heading {level}", f"标题 {level}"):
                    mapping[style_id] = f"h{level}"
                    break
        return mapping

    # ── 结构化格式 ─────────────────────────────────────────────────
    def _parse_csv(self, file_path: Path) -> list[Document]:
        text = _read_text(file_path)
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames or []
        text_col = "text" if "text" in fieldnames else ("content" if "content" in fieldnames else None)
        if text_col is None:
            raise ValueError(
                f"CSV 缺少 text/content 列（实际列: {fieldnames}）: {file_path.name}"
            )
        docs = []
        for index, row in enumerate(reader):
            content = str(row.get(text_col) or "").strip()
            if not content:
                continue
            metadata = {
                "source": str(file_path),
                "file_type": "csv",
                "title": str(row.get("title") or file_path.stem),
                "row_index": index,
            }
            for key, value in row.items():
                if key is None:
                    continue
                if key not in (text_col, "id", "title") and value not in (None, ""):
                    metadata[str(key)] = str(value)
            docs.append(Document(page_content=content, metadata=metadata))
        logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 行")
        return docs

    def _parse_jsonl(self, file_path: Path) -> list[Document]:
        text = _read_text(file_path)
        docs = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"[parser] JSONL 坏行跳过 {file_path.name}:{line_no}: {e}")
                continue
            if not isinstance(obj, dict):
                logger.warning(f"[parser] JSONL 非对象行跳过 {file_path.name}:{line_no}")
                continue
            content = str(obj.get("text") or obj.get("content") or "").strip()
            if not content:
                logger.warning(f"[parser] JSONL 缺少 text/content 跳过 {file_path.name}:{line_no}")
                continue
            metadata = {
                "source": str(file_path),
                "file_type": "jsonl",
                "title": str(obj.get("title") or file_path.stem),
                "row_index": line_no - 1,
            }
            for key, value in obj.items():
                if key in ("text", "content", "id", "title"):
                    continue
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            docs.append(Document(page_content=content, metadata=metadata))
        logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 行")
        return docs

    # ── 目录 ───────────────────────────────────────────────────────
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
                    all_docs.extend(self.parse(file_path))
                except Exception as e:  # noqa: BLE001 - 单文件失败隔离
                    logger.error(f"[parser] 解析失败 {file_path}: {e}")

        logger.info(f"[parser] 目录 {dir_path} 共解析 {len(all_docs)} 个文档段")
        return all_docs

"""文档解析器：策略模式，Markdown + PDF"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from src.utils.logger import logger


class DocumentParser:
    """多格式文档解析器，根据文件后缀自动选择解析策略。"""

    SUPPORTED_EXTENSIONS = {".md", ".pdf"}

    def parse(self, file_path: Path) -> list[Document]:
        """解析单个文件，根据后缀分发到对应解析器。

        Args:
            file_path: 文件路径（支持 .md / .pdf）

        Returns:
            Document 列表，metadata 包含 source, file_type, page_number（PDF 专用）

        Raises:
            ValueError: 不支持的文件格式
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件格式: {suffix}（支持: {self.SUPPORTED_EXTENSIONS}）")

        if suffix == ".md":
            return self._parse_markdown(file_path)
        else:  # .pdf
            return self._parse_pdf(file_path)

    def _parse_markdown(self, file_path: Path) -> list[Document]:
        """解析 Markdown 文件（使用 LangChain TextLoader）"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        # 补充 file_type 元数据
        for doc in docs:
            doc.metadata["file_type"] = "markdown"
        logger.info(f"[parser] 已解析 {file_path.name}：{len(docs)} 个文档段")
        return docs

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

    def parse_directory(self, dir_path: Path, glob_pattern: str = "**/*") -> list[Document]:
        """批量解析目录下所有支持格式的文件。

        Args:
            dir_path: 目录路径
            glob_pattern: 文件匹配模式（默认递归所有文件）

        Returns:
            所有文件的 Document 列表
        """
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

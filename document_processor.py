"""
document_processor.py — DocumentProcessor

Handles the full PDF-to-chunks pipeline:
  1. Load PDF pages via LangChain PyPDFLoader (falls back to PyPDF2)
  2. Clean raw text (fix hyphenation, normalize whitespace, strip artifacts)
  3. Split into overlapping chunks with RecursiveCharacterTextSplitter
  4. Enrich each chunk with detailed metadata for traceability
"""

import logging
import re
from pathlib import Path
from typing import List

import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import ChunkConfig

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Loads, cleans, and chunks academic PDF documents into LangChain
    Document objects ready for embedding.

    Attributes:
        use_langchain_loader (bool): Prefer LangChain's PyPDFLoader when True;
            falls back to PyPDF2 on failure.
    """

    def __init__(self, use_langchain_loader: bool = True):
        self.use_langchain_loader = use_langchain_loader

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, pdf_path: str, config: ChunkConfig) -> List[Document]:
        """
        Full pipeline: load PDF → clean → chunk.

        Args:
            pdf_path: Absolute or relative path to the PDF file.
            config:   ChunkConfig instance defining chunk size and overlap.

        Returns:
            List of Document chunks with enriched metadata.
        """
        pages = self.load_pdf(pdf_path)
        chunks = self.chunk_documents(pages, config)
        return chunks

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF and return one Document per page.
        Tries LangChain PyPDFLoader first; falls back to PyPDF2.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Loading PDF: {path.name}")

        if self.use_langchain_loader:
            try:
                loader = PyPDFLoader(str(path))
                pages = loader.load()
                logger.info(f"  → {len(pages)} pages loaded via PyPDFLoader")
                return pages
            except Exception as exc:
                logger.warning(
                    f"PyPDFLoader failed ({exc}), falling back to PyPDF2"
                )

        return self._load_with_pypdf2(str(path))

    def chunk_documents(
        self, documents: List[Document], config: ChunkConfig
    ) -> List[Document]:
        """
        Split page documents into overlapping text chunks.

        Each chunk is annotated with:
            - chunk_index, total_chunks
            - config_name, chunk_size, chunk_overlap
            - original page and source metadata

        Args:
            documents: List of page-level Documents.
            config:    ChunkConfig specifying chunk_size and overlap.

        Returns:
            List of chunked Document objects with enriched metadata.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            # Ordered separators: prefer paragraph → sentence → word breaks
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
            add_start_index=True,
        )

        # Clean all pages before splitting
        cleaned_docs: List[Document] = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            if cleaned_text:
                cleaned_docs.append(
                    Document(page_content=cleaned_text, metadata=doc.metadata)
                )

        if not cleaned_docs:
            logger.warning("No usable text extracted from PDF.")
            return []

        chunks = splitter.split_documents(cleaned_docs)

        # Enrich metadata on every chunk
        for idx, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "config_name": config.config_name,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                }
            )

        logger.info(
            f"  → [{config.config_name}] {len(chunks)} chunks "
            f"(size={config.chunk_size}, overlap={config.chunk_overlap})"
        )
        return chunks

    def get_stats(self, chunks: List[Document]) -> dict:
        """
        Return a summary statistics dictionary for a list of chunks.

        Returns keys: total_chunks, total_characters, avg_chunk_size,
                       pages_covered, config_name.
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "pages_covered": 0,
                "config_name": "N/A",
            }

        total_chars = sum(len(c.page_content) for c in chunks)
        pages = {c.metadata.get("page", 0) for c in chunks}
        config_name = chunks[0].metadata.get("config_name", "Unknown")

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(chunks),
            "pages_covered": len(pages),
            "config_name": config_name,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_with_pypdf2(self, pdf_path: str) -> List[Document]:
        """Fallback PDF loader using raw PyPDF2."""
        documents: List[Document] = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "page": i + 1,
                                "source": pdf_path,
                                "loader": "PyPDF2",
                                "total_pages": total,
                            },
                        )
                    )
        logger.info(f"  → {len(documents)} pages loaded via PyPDF2")
        return documents

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean raw PDF-extracted text:
          - Fix soft-hyphen line breaks (common in academic PDFs)
          - Collapse excessive blank lines
          - Remove isolated page numbers
          - Normalize unicode punctuation
          - Strip leading/trailing whitespace
        """
        if not text:
            return ""

        # Fix end-of-line hyphenation: "meth-\nod" → "method"
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Collapse 3+ consecutive newlines to a paragraph break
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces/tabs to a single space
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Remove isolated stand-alone page numbers on their own line
        text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)

        # Normalize common Unicode characters to ASCII equivalents
        replacements = {
            "\u2019": "'",   # Right single quote → apostrophe
            "\u2018": "'",   # Left single quote
            "\u201c": '"',   # Left double quote
            "\u201d": '"',   # Right double quote
            "\u2013": "-",   # En dash
            "\u2014": "--",  # Em dash
            "\u00ad": "",    # Soft hyphen (invisible, remove)
            "\u00a0": " ",   # Non-breaking space
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        return text.strip()

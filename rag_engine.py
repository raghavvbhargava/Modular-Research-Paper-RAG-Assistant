"""
rag_engine.py — RAGEngine

Orchestrates the full Retrieval-Augmented Generation pipeline:
  1. Retrieve top-k semantically relevant chunks from a vector store
  2. Assemble a structured context string with page citations
  3. Format a research-focused prompt
  4. Generate a detailed answer using Google Gemini 1.5 Flash
  5. Return a rich RAGResult dataclass with timing and source metadata
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE, TOP_K_RESULTS

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────────────────────────────────────────

RESEARCH_PAPER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert academic research assistant.

Answer the user's question using the context excerpts below. The excerpts come from
different sections of the paper — read ALL of them carefully before answering.

IMPORTANT rules:
- Search every chunk for relevant information before concluding anything is missing.
- If a section heading such as METHODOLOGY, METHODS, RESULTS, or DISCUSSION appears
  in any chunk, treat it as the primary source for questions about that topic.
- Quote specific numbers, terms, and names from the text when available.
- Reference the page or chunk number when citing (e.g., "According to Page 3...").
- Structure your answer with clear sections or bullet points.
- Only say information is missing if it is genuinely absent from ALL chunks below.

─── CONTEXT FROM PAPER ────────────────────────────────────────────────
{context}
────────────────────────────────────────────────────────────────────────

Question: {question}

Detailed answer:"""
)


# ─────────────────────────────────────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """
    Complete result from a single RAG query execution.

    Attributes:
        answer:               Generated answer text from the LLM.
        sources:              Distinct source file paths from retrieved chunks.
        source_pages:         Sorted list of page numbers referenced.
        response_time_ms:     End-to-end wall-clock time in milliseconds.
        config_name:          Human-readable configuration label.
        embed_model:          Embedding model name used for retrieval.
        vector_store_type:    "FAISS" or "Chroma".
        chunk_count_retrieved:Number of chunks retrieved from the store.
        tokens_estimated:     Rough token estimate (context + answer word count).
        error:                Error message if the pipeline failed, else None.
    """
    answer: str
    sources: List[str]
    source_pages: List[int]
    response_time_ms: float
    config_name: str
    embed_model: str
    vector_store_type: str
    chunk_count_retrieved: int
    tokens_estimated: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dictionary."""
        return {
            "config_name": self.config_name,
            "embed_model": self.embed_model,
            "vector_store_type": self.vector_store_type,
            "answer": self.answer,
            "sources": self.sources,
            "source_pages": self.source_pages,
            "response_time_ms": round(self.response_time_ms, 2),
            "chunk_count_retrieved": self.chunk_count_retrieved,
            "tokens_estimated": self.tokens_estimated,
            "error": self.error,
        }

    @property
    def succeeded(self) -> bool:
        """True if the pipeline completed without error."""
        return self.error is None and bool(self.answer)


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Retrieval-Augmented Generation engine powered by Google Gemini.

    Supports any LangChain-compatible vector store (FAISS or ChromaDB).
    The LLM is lazily initialized on first use to minimize startup time.

    Args:
        google_api_key: Your Google AI Studio API key.
        model_name:     Gemini model identifier (default: gemini-1.5-flash).
        temperature:    Sampling temperature — lower = more deterministic.
        max_tokens:     Maximum output tokens per generation call.
    """

    def __init__(
        self,
        google_api_key: str,
        model_name: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ):
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm: Optional[ChatGoogleGenerativeAI] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate_answer(
        self,
        query: str,
        vector_store: Union[FAISS, Chroma],
        config_name: str,
        embed_model_name: str,
        k: int = TOP_K_RESULTS,
    ) -> RAGResult:
        """
        Execute the full RAG pipeline for a single query.

        Steps:
            1. Retrieve top-k chunks from the vector store.
            2. Assemble a structured context string with page numbers.
            3. Format the research-focused prompt.
            4. Call Gemini to generate the answer.
            5. Collect and return metadata in a RAGResult.

        Args:
            query:            The research question to answer.
            vector_store:     Initialized FAISS or Chroma store.
            config_name:      Label for this configuration (for reporting).
            embed_model_name: Embedding model name (for reporting).
            k:                Number of chunks to retrieve.

        Returns:
            RAGResult with the answer, sources, and performance metrics.
        """
        start_time = time.perf_counter()
        store_type = type(vector_store).__name__

        try:
            # Step 1: Retrieve
            logger.info(
                f"[{config_name}] Retrieving top-{k} chunks from {store_type}..."
            )
            retrieved_docs = vector_store.similarity_search(query, k=k)

            if not retrieved_docs:
                logger.warning(f"[{config_name}] No chunks retrieved for query.")
            else:
                # Debug: show exactly which pages/chunks were retrieved
                pages = [d.metadata.get('page', '?') for d in retrieved_docs]
                logger.info(
                    f"[{config_name}] Retrieved {len(retrieved_docs)} chunks "
                    f"from pages: {pages}"
                )
                # Log first 80 chars of each chunk for quick diagnosis
                for i, doc in enumerate(retrieved_docs):
                    snippet = doc.page_content[:80].replace('\n', ' ')
                    logger.debug(
                        f"  Chunk {i+1} (page {doc.metadata.get('page','?')}): {snippet}..."
                    )

            # Step 2: Assemble context
            context = self._build_context(retrieved_docs)

            # Step 3: Format prompt
            messages = RESEARCH_PAPER_PROMPT.format_messages(
                context=context,
                question=query,
            )

            # Step 4: Generate
            logger.info(
                f"[{config_name}] Generating answer via {self.model_name}..."
            )
            llm = self._get_llm()
            response = llm.invoke(messages)
            answer = response.content.strip()

            # Step 5: Gather metadata
            source_pages = sorted(
                {
                    int(d.metadata["page"])
                    for d in retrieved_docs
                    if "page" in d.metadata
                }
            )
            sources = list(
                {d.metadata.get("source", "unknown") for d in retrieved_docs}
            )
            tokens_estimated = len(answer.split()) + len(context.split())

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"[{config_name}] Done — {elapsed_ms:.0f}ms")

            return RAGResult(
                answer=answer,
                sources=sources,
                source_pages=source_pages,
                response_time_ms=elapsed_ms,
                config_name=config_name,
                embed_model=embed_model_name,
                vector_store_type=store_type,
                chunk_count_retrieved=len(retrieved_docs),
                tokens_estimated=tokens_estimated,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"[{config_name}] RAG pipeline failed: {exc}", exc_info=True)
            return RAGResult(
                answer="",
                sources=[],
                source_pages=[],
                response_time_ms=elapsed_ms,
                config_name=config_name,
                embed_model=embed_model_name,
                vector_store_type=store_type,
                chunk_count_retrieved=0,
                error=str(exc),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Lazily initialize Gemini LLM on first call."""
        if self._llm is None:
            logger.info(f"Initializing LLM: {self.model_name}")
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.google_api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        return self._llm

    @staticmethod
    def _build_context(docs: List[Document]) -> str:
        """
        Assemble retrieved chunks into a single labeled context string.

        Each chunk is prefixed with its chunk index and page number so the LLM
        can reference sources accurately.
        """
        if not docs:
            return "No relevant context found in the document."

        parts = []
        for doc in docs:
            page = doc.metadata.get("page", "?")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            config = doc.metadata.get("config_name", "")
            header = f"[Chunk {chunk_idx} | Page {page}]"
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

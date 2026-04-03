"""
vector_store_manager.py — VectorStoreManager

Manages creation, persistence, and similarity search for:
  - FAISS       (local, lightning-fast, in-memory / optional disk save)
  - ChromaDB    (persistent, disk-backed, survives process restarts)

Supports two embedding model providers:
  - HuggingFace (sentence-transformers, fully local)
  - Google      (text-embedding-004, requires GOOGLE_API_KEY)
"""

import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google_embeddings import GoogleV1Embeddings

from config import (
    EmbeddingConfig,
    CHROMA_PERSIST_DIR,
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Factory and manager for FAISS and ChromaDB vector stores.

    Handles embedding model initialization (with caching to avoid
    re-loading heavy HuggingFace models), index building, persistence,
    loading, and unified similarity search.

    Args:
        google_api_key: Required when using Google embedding models.
    """

    def __init__(self, google_api_key: Optional[str] = None):
        self.google_api_key = google_api_key
        self._embedding_cache: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding Model Factory
    # ─────────────────────────────────────────────────────────────────────────

    def get_embeddings(self, embed_config: EmbeddingConfig):
        """
        Return a LangChain-compatible embeddings object for the given config.
        Results are cached by model_name to avoid repeated heavy loads.

        Args:
            embed_config: EmbeddingConfig specifying provider and model name.

        Returns:
            Initialized LangChain embeddings object.

        Raises:
            ValueError: If provider is unknown or API key is missing.
        """
        cache_key = embed_config.model_name
        if cache_key in self._embedding_cache:
            logger.debug(f"Reusing cached embedding model: {embed_config.model_name}")
            return self._embedding_cache[cache_key]

        logger.info(
            f"Loading embedding model: {embed_config.model_name} "
            f"[{embed_config.provider}]"
        )
        start = time.perf_counter()

        if embed_config.provider == "huggingface":
            embeddings = HuggingFaceEmbeddings(
                model_name=embed_config.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        elif embed_config.provider == "google":
            if not self.google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is required for Google embeddings. "
                    "Add it to your .env file."
                )
            embeddings = GoogleV1Embeddings(
                model=embed_config.model_name,
                api_key=self.google_api_key,
            )

        else:
            raise ValueError(
                f"Unknown embedding provider: '{embed_config.provider}'. "
                "Supported: 'huggingface', 'google'."
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"  → Embedding model ready in {elapsed_ms:.0f}ms")

        self._embedding_cache[cache_key] = embeddings
        return embeddings

    # ─────────────────────────────────────────────────────────────────────────
    # FAISS
    # ─────────────────────────────────────────────────────────────────────────

    def build_faiss(
        self,
        documents: List[Document],
        embed_config: EmbeddingConfig,
        save_path: Optional[str] = None,
    ) -> FAISS:
        """
        Build a FAISS index from a list of Document chunks.

        Args:
            documents:    Chunked documents to embed and index.
            embed_config: Which embedding model to use.
            save_path:    If provided, persist the index to this directory.

        Returns:
            An initialized FAISS vector store.
        """
        if not documents:
            raise ValueError("Cannot build FAISS index: no documents provided.")

        logger.info(f"Building FAISS index for {len(documents)} chunks...")
        embeddings = self.get_embeddings(embed_config)

        start = time.perf_counter()
        store = FAISS.from_documents(documents, embeddings)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(f"  → FAISS index built in {elapsed_ms:.0f}ms")

        if save_path:
            store.save_local(save_path)
            logger.info(f"  → FAISS index persisted to: {save_path}")

        return store

    def load_faiss(
        self, load_path: str, embed_config: EmbeddingConfig
    ) -> FAISS:
        """
        Load a previously saved FAISS index from disk.

        Args:
            load_path:    Directory containing the saved FAISS index.
            embed_config: Must match the embeddings used when the index was built.

        Returns:
            Loaded FAISS vector store.
        """
        embeddings = self.get_embeddings(embed_config)
        store = FAISS.load_local(
            load_path, embeddings, allow_dangerous_deserialization=True
        )
        logger.info(f"FAISS index loaded from: {load_path}")
        return store

    # ─────────────────────────────────────────────────────────────────────────
    # ChromaDB
    # ─────────────────────────────────────────────────────────────────────────

    def build_chroma(
        self,
        documents: List[Document],
        embed_config: EmbeddingConfig,
        collection_name: str = "research_papers",
        persist_dir: str = CHROMA_PERSIST_DIR,
        reset: bool = True,
    ) -> Chroma:
        """
        Build a ChromaDB collection with persistent on-disk storage.

        Args:
            documents:       Chunked documents to embed and store.
            embed_config:    Which embedding model to use.
            collection_name: Name for the ChromaDB collection.
            persist_dir:     Directory for persistent storage.
            reset:           If True, delete existing collection before building.

        Returns:
            An initialized Chroma vector store.
        """
        if not documents:
            raise ValueError("Cannot build ChromaDB collection: no documents provided.")

        if reset and Path(persist_dir).exists():
            shutil.rmtree(persist_dir)
            logger.info(f"Cleared existing ChromaDB at: {persist_dir}")

        logger.info(
            f"Building ChromaDB collection '{collection_name}' "
            f"for {len(documents)} chunks..."
        )
        embeddings = self.get_embeddings(embed_config)

        start = time.perf_counter()
        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(f"  → ChromaDB collection built in {elapsed_ms:.0f}ms")
        return store

    def load_chroma(
        self,
        embed_config: EmbeddingConfig,
        collection_name: str = "research_papers",
        persist_dir: str = CHROMA_PERSIST_DIR,
    ) -> Chroma:
        """
        Load an existing ChromaDB collection from persistent storage.

        Args:
            embed_config:    Must match embeddings used when collection was built.
            collection_name: Name of the collection to load.
            persist_dir:     Directory where the collection is stored.

        Returns:
            Loaded Chroma vector store.
        """
        embeddings = self.get_embeddings(embed_config)
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        logger.info(
            f"ChromaDB collection '{collection_name}' loaded from: {persist_dir}"
        )
        return store

    # ─────────────────────────────────────────────────────────────────────────
    # Unified Retrieval Interface
    # ─────────────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        store: Union[FAISS, Chroma],
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        Perform similarity search across either a FAISS or ChromaDB store.

        Args:
            store:           Initialized FAISS or Chroma vector store.
            query:           Natural language query string.
            k:               Number of results to retrieve.
            score_threshold: If set, filter out results below this relevance score.

        Returns:
            List of the most semantically similar Document chunks.
        """
        if score_threshold is not None:
            results_with_scores = store.similarity_search_with_relevance_scores(
                query, k=k
            )
            docs = [
                doc
                for doc, score in results_with_scores
                if score >= score_threshold
            ]
            logger.debug(
                f"Retrieved {len(docs)}/{k} chunks "
                f"(threshold={score_threshold:.2f})"
            )
            return docs

        docs = store.similarity_search(query, k=k)
        logger.debug(f"Retrieved {len(docs)} chunks")
        return docs

    def get_store_stats(self, store: Union[FAISS, Chroma]) -> dict:
        """
        Return basic statistics about an initialized vector store.

        Returns:
            Dict with 'store_type' and 'vector_count'.
        """
        store_type = type(store).__name__
        try:
            if isinstance(store, FAISS):
                count = store.index.ntotal
            elif isinstance(store, Chroma):
                count = store._collection.count()
            else:
                count = -1
        except Exception:
            count = -1

        return {"store_type": store_type, "vector_count": count}

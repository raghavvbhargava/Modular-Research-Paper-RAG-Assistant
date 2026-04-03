"""
config.py — Centralized Configuration for the RAG Research Paper Assistant

All chunking strategies, embedding model definitions, LLM settings,
and vector store paths live here. Modify this file to switch models or
tune parameters without touching the core pipeline code.
"""

from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Chunking Configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkConfig:
    """Defines a document chunking strategy."""
    config_name: str
    chunk_size: int          # Approximate token/character size per chunk
    overlap_ratio: float     # Fractional overlap (e.g., 0.10 = 10%)

    @property
    def chunk_overlap(self) -> int:
        """Compute the absolute overlap size from the ratio."""
        return int(self.chunk_size * self.overlap_ratio)

    def __str__(self) -> str:
        return (
            f"{self.config_name} "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap} [{self.overlap_ratio*100:.0f}%])"
        )


# Config 1: Small chunks — higher precision, more context windows
CONFIG_SMALL = ChunkConfig(
    config_name="Config-1 (Small Chunks)",
    chunk_size=500,
    overlap_ratio=0.10,   # 10% → 50 char overlap
)

# Config 2: Large chunks — broader context, fewer retrieval calls
CONFIG_LARGE = ChunkConfig(
    config_name="Config-2 (Large Chunks)",
    chunk_size=1200,
    overlap_ratio=0.15,   # 15% → 180 char overlap
)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Model Configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    """Defines an embedding model and its provider."""
    model_name: str
    provider: str       # "huggingface" | "google" | "openai"
    dimension: int      # Output embedding dimension
    description: str    # Human-readable label for display


# Model A: HuggingFace MPNet — runs fully locally, no API key required
EMBED_A_MPNET = EmbeddingConfig(
    model_name="sentence-transformers/all-mpnet-base-v2",
    provider="huggingface",
    dimension=768,
    description="HuggingFace MPNet (local, 768-dim)",
)

# Model B: Google gemini-embedding-001 — available on free tier via v1beta
EMBED_B_GEMINI = EmbeddingConfig(
    model_name="models/gemini-embedding-001",
    provider="google",
    dimension=3072,
    description="Google gemini-embedding-001 (API, 3072-dim)",
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Settings
# ─────────────────────────────────────────────────────────────────────────────

LLM_MODEL       = "models/gemini-2.5-flash"   # Gemini 2.5 Flash via v1beta
LLM_TEMPERATURE = 0.3                  # Lower = more factual / deterministic
LLM_MAX_TOKENS  = 1500                 # Max output tokens per response


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Paths
# ─────────────────────────────────────────────────────────────────────────────

FAISS_INDEX_PATH        = "faiss_index"
CHROMA_PERSIST_DIR      = "chroma_db"
CHROMA_COLLECTION_BASE  = "research_papers"


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Settings
# ─────────────────────────────────────────────────────────────────────────────

TOP_K_RESULTS = 8   # Number of chunks to retrieve per query (higher = better recall)


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON_REPORT_PATH = "comparison_report.json"

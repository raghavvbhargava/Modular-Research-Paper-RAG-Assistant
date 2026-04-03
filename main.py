"""
main.py — Research Paper RAG Assistant
System Comparison Mode

Processes the same research paper PDF with two distinct RAG configurations
and prints a side-by-side performance + answer quality report.

Configuration Matrix:
  ┌──────────┬──────────────┬───────────────┬──────────────────┬────────────┐
  │ Config   │ Chunk Size   │ Overlap       │ Embedding Model  │ VectorDB   │
  ├──────────┼──────────────┼───────────────┼──────────────────┼────────────┤
  │ Config 1 │ 500 chars    │ 10% (50 ch)   │ HuggingFace MPNet│ FAISS      │
  │ Config 2 │ 1200 chars   │ 15% (180 ch)  │ Google Emb-004   │ ChromaDB   │
  └──────────┴──────────────┴───────────────┴──────────────────┴────────────┘

Usage:
    # Interactive mode (prompted input)
    python main.py

    # Fully specified via CLI
    python main.py --pdf paper.pdf --query "What methodology was used?"

    # Multiple queries from a file (one per line)
    python main.py --pdf paper.pdf --query-file questions.txt

    # Suppress log noise; skip saving the JSON report
    python main.py --pdf paper.pdf --query "..." --log-level WARNING --no-save

    # Demo mode — picks first PDF from sample_papers/ folder
    python main.py --demo
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
# Plain input used instead of rich.prompt

# Load .env before any module that reads env vars
load_dotenv()

from config import (
    CHROMA_COLLECTION_BASE,
    COMPARISON_REPORT_PATH,
    CONFIG_LARGE,
    CONFIG_SMALL,
    EMBED_A_MPNET,
    EMBED_B_GEMINI,
    TOP_K_RESULTS,
)
from document_processor import DocumentProcessor
from rag_engine import RAGEngine, RAGResult
from utils import (
    print_banner,
    print_chunk_stats,
    print_comparison_table,
    print_config_summary,
    print_full_answers,
    save_comparison_report,
    setup_logging,
)
from vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Argument Parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Research Paper RAG Assistant — System Comparison Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pdf",
        type=str,
        metavar="PATH",
        help="Path to the research paper PDF file.",
    )
    parser.add_argument(
        "--query",
        type=str,
        metavar="TEXT",
        help="Research question to ask about the paper.",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        metavar="FILE",
        help="Path to a text file with one query per line (runs all queries).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_RESULTS,
        metavar="N",
        help=f"Chunks to retrieve per configuration (default: {TOP_K_RESULTS}).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving the JSON comparison report to disk.",
    )
    parser.add_argument(
        "--full-answers",
        action="store_true",
        help="Print full (untruncated) answers after the comparison table.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: automatically use the first PDF in sample_papers/.",
    )
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Environment Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_environment() -> dict:
    """
    Check that required environment variables are set.
    Prints a friendly setup message and exits if GOOGLE_API_KEY is missing.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if not google_api_key:
        print()
        print("ERROR: GOOGLE_API_KEY not found!")
        print("\nGet a free key at: https://aistudio.google.com/app/apikey")
        print("Then add it to your .env file:  GOOGLE_API_KEY=your_key_here")
        sys.exit(1)

    print("  GOOGLE_API_KEY loaded successfully.")
    return {"google_api_key": google_api_key}


# ─────────────────────────────────────────────────────────────────────────────
# PDF Resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_pdf(args: argparse.Namespace) -> Path:
    """
    Determine the PDF path from CLI args, sample_papers/ picker, or manual input.
    Auto-scans sample_papers/ and presents a numbered list to choose from.
    """
    SAMPLE_DIR = Path(__file__).parent / "sample_papers"

    # --pdf flag provided directly — resolve relative to sample_papers/ first
    if args.pdf:
        candidate = Path(args.pdf)
        # Try as-is first
        if candidate.exists() and candidate.suffix.lower() == ".pdf":
            return candidate
        # Try inside sample_papers/
        in_sample = SAMPLE_DIR / args.pdf
        if in_sample.exists() and in_sample.suffix.lower() == ".pdf":
            return in_sample
        print(f"\nERROR: File not found: {args.pdf}")
        sys.exit(1)

    # Demo mode — auto-pick first PDF
    if args.demo:
        pdfs = sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.exists() else []
        if not pdfs:
            print("\nDemo mode: No PDFs found in sample_papers/. Add a PDF there and retry.")
            sys.exit(1)
        print(f"\nDemo mode -> using: {pdfs[0].name}")
        return pdfs[0]

    # Auto-scan sample_papers/ and show a picker
    pdfs = sorted(SAMPLE_DIR.glob("*.pdf")) if SAMPLE_DIR.exists() else []

    if pdfs:
        print("\nPDFs found in sample_papers/:")
        for i, p in enumerate(pdfs, 1):
            size_kb = p.stat().st_size // 1024
            print(f"  {i}. {p.name}  ({size_kb} KB)")

        if len(pdfs) == 1:
            print(f"\nAuto-selected: {pdfs[0].name}")
            return pdfs[0]

        print(f"  {len(pdfs)+1}. Enter a custom path...")
        choice_str = input("\nSelect a paper [1]: ").strip() or "1"
        try:
            choice = int(choice_str)
            if 1 <= choice <= len(pdfs):
                return pdfs[choice - 1]
        except ValueError:
            pass

    # No PDFs in sample_papers/ — ask for manual path
    if not pdfs:
        print(f"\nNo PDFs found in: {SAMPLE_DIR}")
        print("Drop any research paper PDF into that folder, or enter a full path below.")

    raw_path = input("\nFull path to your research paper PDF: ")
    pdf = Path(raw_path.strip().strip('"'))

    if not pdf.exists():
        alt = SAMPLE_DIR / pdf.name
        if alt.exists():
            return alt
        print(f"\nERROR: File not found: {pdf}")
        sys.exit(1)
    if pdf.suffix.lower() != ".pdf":
        print(f"\nERROR: Not a PDF file: {pdf}")
        sys.exit(1)

    return pdf


# ─────────────────────────────────────────────────────────────────────────────
# Query Resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_queries(args: argparse.Namespace) -> list[str]:
    """Return a list of queries from CLI, query file, or interactive prompt."""
    if args.query_file:
        qfile = Path(args.query_file)
        if not qfile.exists():
            print(f"ERROR: Query file not found: {qfile}")
            sys.exit(1)
        queries = [
            line.strip()
            for line in qfile.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not queries:
            print("ERROR: Query file is empty.")
            sys.exit(1)
        print(f"  Loaded {len(queries)} queries from {qfile.name}")
        return queries

    if args.query:
        return [args.query.strip()]

    query = input("\nEnter your research question: ").strip()
    if not query:
        print("ERROR: Query cannot be empty.")
        sys.exit(1)
    return [query]


# ─────────────────────────────────────────────────────────────────────────────
# Core Comparison Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    pdf_path: str,
    query: str,
    google_api_key: str,
    top_k: int = TOP_K_RESULTS,
) -> tuple[RAGResult, RAGResult]:
    """
    Process the same PDF twice using two configurations and return results.

    Config 1:  500-char chunks · 10% overlap · MPNet (HuggingFace) · FAISS
    Config 2: 1200-char chunks · 15% overlap · Gemini text-embedding-004 · ChromaDB

    Args:
        pdf_path:       Path to the research paper PDF.
        query:          The research question to answer.
        google_api_key: Google AI Studio API key.
        top_k:          Number of chunks to retrieve per configuration.

    Returns:
        Tuple of (result_config1, result_config2).
    """
    processor = DocumentProcessor(use_langchain_loader=True)
    vs_manager = VectorStoreManager(google_api_key=google_api_key)
    rag_engine = RAGEngine(google_api_key=google_api_key)

    # ── Config 1: Small Chunks + MPNet + FAISS ────────────────────────────
    print("\nPhase 1 of 2: Config 1 - Small Chunks + FAISS")
    print_config_summary(
        config_name=CONFIG_SMALL.config_name,
        chunk_size=CONFIG_SMALL.chunk_size,
        overlap=CONFIG_SMALL.chunk_overlap,
        embed_model=EMBED_A_MPNET.model_name,
        vector_store="FAISS (local, in-memory)",
    )

    print("  Chunking PDF...")
    chunks_1 = processor.process(pdf_path, CONFIG_SMALL)
    stats_1 = processor.get_stats(chunks_1)
    print("  Chunking complete.")
    print_chunk_stats(CONFIG_SMALL.config_name, stats_1)

    print("  Building FAISS index (MPNet embeddings)...")
    faiss_store = vs_manager.build_faiss(chunks_1, EMBED_A_MPNET)
    print("  FAISS index ready.")

    print("\n  Generating answer (Config 1)...")
    result_1 = rag_engine.generate_answer(
        query=query,
        vector_store=faiss_store,
        config_name=CONFIG_SMALL.config_name,
        embed_model_name=EMBED_A_MPNET.model_name,
        k=top_k,
    )
    _print_result_status(result_1)

    # ── Config 2: Large Chunks + Gemini Embeddings + ChromaDB ────────────
    print("\nPhase 2 of 2: Config 2 - Large Chunks + ChromaDB")
    print_config_summary(
        config_name=CONFIG_LARGE.config_name,
        chunk_size=CONFIG_LARGE.chunk_size,
        overlap=CONFIG_LARGE.chunk_overlap,
        embed_model=EMBED_B_GEMINI.model_name,
        vector_store="ChromaDB (persistent, on-disk)",
    )

    print("  Chunking PDF...")
    chunks_2 = processor.process(pdf_path, CONFIG_LARGE)
    stats_2 = processor.get_stats(chunks_2)
    print("  Chunking complete.")
    print_chunk_stats(CONFIG_LARGE.config_name, stats_2)

    print("  Building ChromaDB (Gemini embeddings)...")
    chroma_store = vs_manager.build_chroma(
        chunks_2,
        EMBED_B_GEMINI,
        collection_name=f"{CHROMA_COLLECTION_BASE}_config2",
        reset=True,
    )
    print("  ChromaDB collection ready.")

    print("\n  Generating answer (Config 2)...")
    result_2 = rag_engine.generate_answer(
        query=query,
        vector_store=chroma_store,
        config_name=CONFIG_LARGE.config_name,
        embed_model_name=EMBED_B_GEMINI.model_name,
        k=top_k,
    )
    _print_result_status(result_2)

    return result_1, result_2


def _print_result_status(result: RAGResult) -> None:
    """Print a one-line status line after answer generation."""
    if result.succeeded:
        print(
            f"  Answer generated in {result.response_time_ms:.0f}ms  |  "
            f"{result.chunk_count_retrieved} chunks used  |  "
            f"~{result.tokens_estimated} tokens"
        )
    else:
        print(f"  Generation failed: {result.error}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    print_banner()

    # Validate environment
    print("Checking environment...")
    env = validate_environment()

    # Resolve PDF
    pdf_file = resolve_pdf(args)
    print(f"\nPaper: {pdf_file.name}  ({pdf_file.stat().st_size // 1024} KB)")

    # Resolve queries (one or many)
    queries = resolve_queries(args)

    # Iterate over all queries
    all_reports = []
    for query_idx, query in enumerate(queries, 1):
        if len(queries) > 1:
            print(f"\n=== Query {query_idx}/{len(queries)} ===")

        print(f"\nStarting System Comparison")
        print(f"Question: {query}")

        try:
            result_1, result_2 = run_comparison(
                pdf_path=str(pdf_file),
                query=query,
                google_api_key=env["google_api_key"],
                top_k=args.top_k,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(0)
        except Exception as exc:
            print(f"\nPipeline error: {exc}")
            logger.exception("Unhandled pipeline error")
            sys.exit(1)

        # Display results
        print_comparison_table(result_1, result_2, query)

        if args.full_answers:
            print_full_answers(result_1, result_2)

        # Save report
        if not args.no_save:
            report_path = (
                COMPARISON_REPORT_PATH
                if len(queries) == 1
                else f"comparison_report_q{query_idx}.json"
            )
            report = save_comparison_report(
                result1=result_1,
                result2=result_2,
                query=query,
                pdf_path=str(pdf_file.absolute()),
                output_path=report_path,
            )
            all_reports.append(report)

        # Prompt to continue if multiple queries
        if len(queries) > 1 and query_idx < len(queries):
            cont = input("\nContinue to next query? [Y/n]: ").strip().lower()
            if cont in ("n", "no"):
                print("Stopping early.")
                break

    print("\nAll done!\n")


if __name__ == "__main__":
    main()

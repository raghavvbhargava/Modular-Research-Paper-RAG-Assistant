"""
utils.py — Shared Utilities

Provides:
  - Logging configuration with clean formatting
  - A context-manager timer for block-level profiling
  - Rich-powered terminal display functions (banner, config cards, comparison table)
  - JSON report serializer
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=True)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure the root logger.

    Args:
        level:    One of DEBUG, INFO, WARNING, ERROR.
        log_file: If provided, also write logs to this file path.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
    datefmt = "%H:%M:%S"

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy third-party loggers
    for noisy_lib in [
        "httpx", "httpcore", "urllib3", "chromadb",
        "faiss", "sentence_transformers", "transformers",
        "google.auth", "google.api_core",
    ]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Timer
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str):
    """
    Context manager that prints elapsed time after the block completes.

    Usage:
        with timer("Building FAISS index"):
            store = build_faiss(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        console.print(
            f"  ⏱  [dim]{label}:[/dim] "
            f"[bold cyan]{elapsed_ms:.0f}ms[/bold cyan]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Rich Terminal Display
# ─────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    """Print the application header banner."""
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]🔬  Research Paper RAG Assistant[/bold blue]\n"
            "[dim]Modular Retrieval-Augmented Generation · System Comparison Mode[/dim]",
            border_style="blue",
            padding=(1, 4),
        )
    )
    console.print()


def print_config_summary(
    config_name: str,
    chunk_size: int,
    overlap: int,
    embed_model: str,
    vector_store: str,
    color: str = "blue",
) -> None:
    """
    Print a styled configuration summary panel.

    Args:
        config_name:  Display name for this configuration.
        chunk_size:   Characters per chunk.
        overlap:      Overlap characters.
        embed_model:  Embedding model name.
        vector_store: Vector store type (FAISS / ChromaDB).
        color:        Rich color for the panel border.
    """
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Setting", style="bold cyan", width=18)
    table.add_column("Value", style="white")

    table.add_row("Chunk Size", f"{chunk_size} chars")
    table.add_row("Chunk Overlap", f"{overlap} chars")
    table.add_row("Embedding Model", embed_model.split("/")[-1])
    table.add_row("Vector Store", vector_store)

    console.print(
        Panel(
            table,
            title=f"[bold {color}]{config_name}[/bold {color}]",
            border_style=color,
            expand=False,
        )
    )


def print_chunk_stats(config_name: str, stats: dict) -> None:
    """Print a one-line chunking summary after processing."""
    console.print(
        f"  [dim]│[/dim] [bold]{stats['total_chunks']}[/bold] chunks  "
        f"· avg [bold]{stats['avg_chunk_size']}[/bold] chars  "
        f"· [bold]{stats['pages_covered']}[/bold] pages covered"
    )


def print_comparison_table(result1, result2, query: str) -> None:
    """
    Display a rich side-by-side comparison of two RAGResult objects.

    Includes:
      - Performance metrics table (timing, chunks, pages, tokens)
      - Two answer panels displayed in columns
    """
    from rag_engine import RAGResult  # local import to avoid circular

    console.print()
    console.print(Rule("[bold yellow]📊 SYSTEM COMPARISON RESULTS[/bold yellow]"))
    console.print()
    console.print(f"[bold]Query:[/bold] [italic]{query}[/italic]")
    console.print()

    # ── Metrics Table ──────────────────────────────────────────────────────
    metrics = Table(
        title="⚡ Performance Metrics",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        header_style="bold white on dark_blue",
        expand=False,
    )
    metrics.add_column("Metric", style="bold cyan", width=26)
    metrics.add_column(
        Text(result1.config_name, style="bold green"), width=32
    )
    metrics.add_column(
        Text(result2.config_name, style="bold yellow"), width=32
    )

    # Response time — highlight the winner
    t1_raw = result1.response_time_ms
    t2_raw = result2.response_time_ms
    t1_str = f"{t1_raw:.0f} ms"
    t2_str = f"{t2_raw:.0f} ms"
    if t1_raw < t2_raw:
        t1_str = f"✅ {t1_str}  (faster)"
    elif t2_raw < t1_raw:
        t2_str = f"✅ {t2_str}  (faster)"
    else:
        t1_str = f"🟰 {t1_str}  (tied)"
        t2_str = f"🟰 {t2_str}  (tied)"

    metrics.add_row("⏱  Response Time", t1_str, t2_str)
    metrics.add_row(
        "🗄  Vector Store",
        result1.vector_store_type,
        result2.vector_store_type,
    )
    metrics.add_row(
        "🧠  Embedding Model",
        result1.embed_model.split("/")[-1],
        result2.embed_model.split("/")[-1],
    )
    metrics.add_row(
        "📄  Chunks Retrieved",
        str(result1.chunk_count_retrieved),
        str(result2.chunk_count_retrieved),
    )
    metrics.add_row(
        "📑  Source Pages",
        _format_pages(result1.source_pages),
        _format_pages(result2.source_pages),
    )
    metrics.add_row(
        "🔢  Est. Tokens Used",
        str(result1.tokens_estimated),
        str(result2.tokens_estimated),
    )
    metrics.add_row(
        "✅  Status",
        "[green]Success[/green]" if result1.succeeded else "[red]Error[/red]",
        "[green]Success[/green]" if result2.succeeded else "[red]Error[/red]",
    )

    console.print(metrics)
    console.print()

    # ── Answer Panels ──────────────────────────────────────────────────────
    console.print("[bold]Generated Answers (preview):[/bold]")
    console.print()

    a1_panel = Panel(
        _truncate(result1.answer, 800)
        if result1.succeeded
        else f"[bold red]❌ Error:[/bold red] {result1.error}",
        title="[bold green]Config 1: Answer[/bold green]",
        border_style="green",
        width=72,
        padding=(1, 2),
    )
    a2_panel = Panel(
        _truncate(result2.answer, 800)
        if result2.succeeded
        else f"[bold red]❌ Error:[/bold red] {result2.error}",
        title="[bold yellow]Config 2: Answer[/bold yellow]",
        border_style="yellow",
        width=72,
        padding=(1, 2),
    )

    console.print(Columns([a1_panel, a2_panel], equal=True, expand=True))


def print_full_answers(result1, result2) -> None:
    """Print the full (untruncated) answers for both configurations."""
    console.print()
    console.print(Rule("[bold green]Config 1 — Full Answer[/bold green]"))
    console.print(result1.answer if result1.succeeded else f"Error: {result1.error}")

    console.print()
    console.print(Rule("[bold yellow]Config 2 — Full Answer[/bold yellow]"))
    console.print(result2.answer if result2.succeeded else f"Error: {result2.error}")


# ─────────────────────────────────────────────────────────────────────────────
# Report Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_report(
    result1,
    result2,
    query: str,
    pdf_path: str,
    output_path: str = "comparison_report.json",
) -> dict:
    """
    Serialize both RAGResult objects to a structured JSON report.

    Args:
        result1:     RAGResult from Config 1.
        result2:     RAGResult from Config 2.
        query:       The research question that was asked.
        pdf_path:    Path to the source PDF file.
        output_path: Where to write the JSON report.

    Returns:
        The report dictionary (also written to disk).
    """
    speed_winner = (
        result1.config_name
        if result1.response_time_ms <= result2.response_time_ms
        else result2.config_name
    )
    speed_diff_ms = abs(result1.response_time_ms - result2.response_time_ms)

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pdf_file": pdf_path,
            "query": query,
            "llm_model": "gemini-1.5-flash",
        },
        "config_1": result1.to_dict(),
        "config_2": result2.to_dict(),
        "summary": {
            "winner_speed": speed_winner,
            "speed_diff_ms": round(speed_diff_ms, 2),
            "config_1_succeeded": result1.succeeded,
            "config_2_succeeded": result2.succeeded,
        },
    }

    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(
        f"\n✅ [bold green]Full report saved →[/bold green] "
        f"[underline]{out.absolute()}[/underline]"
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: str, max_chars: int = 600) -> str:
    """Truncate a string and append an ellipsis if it exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[dim]… (truncated — see full report)[/dim]"


def _format_pages(pages: List[int]) -> str:
    """Format a list of page numbers for display."""
    if not pages:
        return "N/A"
    if len(pages) <= 6:
        return ", ".join(str(p) for p in pages)
    return ", ".join(str(p) for p in pages[:6]) + f" … (+{len(pages) - 6} more)"

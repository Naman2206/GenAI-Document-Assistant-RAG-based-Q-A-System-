"""
main.py
-------
Command-line entry point for the GenAI Document Assistant.

Usage:
    python main.py ingest  --docs data/sample_docs [--mock]
    python main.py ask     --question "What does the warranty cover?" [--mock]
    python main.py evaluate --question "..." [--mock]
    python main.py demo    [--mock]      # ingest + ask + evaluate in one go

--mock runs the entire pipeline offline (see src/mock_clients.py) so you
can demo it live in an interview without needing provisioned Azure
resources or paying for API calls.
"""
from __future__ import annotations

import argparse
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _enable_mock():
    os.environ["RAG_MOCK_MODE"] = "true"
    # config.settings was already loaded at import time elsewhere; reload it
    import importlib
    import src.config as config_module
    importlib.reload(config_module)


def cmd_ingest(args):
    from src.ingestion import ingest_directory

    chunks = ingest_directory(args.docs)
    console.print(f"[green]Ingested {len(chunks)} chunks from '{args.docs}'[/green]")

    if args.mock:
        from src.embeddings import embed_texts
        from src.mock_clients import register_mock_documents
        register_mock_documents(chunks)
        console.print("[yellow]Mock mode:[/yellow] chunks embedded and stored in-memory (no Azure calls).")
        return

    from src.embeddings import embed_texts
    from src.search_index import create_or_update_index, upload_chunks

    create_or_update_index()
    vectors = embed_texts([c.content for c in chunks])
    upload_chunks(chunks, vectors)
    console.print("[green]Index build complete.[/green]")


def cmd_ask(args):
    from src.generation import answer_question

    result = answer_question(args.question, top_k=args.top_k)
    console.print(Panel(result.answer, title="Answer", border_style="cyan"))

    table = Table(title="Retrieved Context")
    table.add_column("Source")
    table.add_column("Chunk")
    table.add_column("Score", justify="right")
    for p in result.passages:
        table.add_row(p.source, str(p.chunk_index), f"{p.score:.3f}")
    console.print(table)
    return result


def cmd_evaluate(args):
    from src.generation import answer_question
    from src.evaluation import llm_judge

    result = answer_question(args.question, top_k=args.top_k)
    eval_result = llm_judge(args.question, result)

    console.print(Panel(result.answer, title="Answer", border_style="cyan"))
    console.print(
        f"[bold]Groundedness:[/bold] {eval_result.groundedness}/5   "
        f"[bold]Relevance:[/bold] {eval_result.relevance}/5"
    )
    console.print(f"[italic]{eval_result.rationale}[/italic]")


def cmd_demo(args):
    console.rule("[bold]1. Ingest + Index[/bold]")
    ingest_args = argparse.Namespace(docs="data/sample_docs", mock=args.mock)
    cmd_ingest(ingest_args)

    demo_question = args.question or "What is covered under the standard product warranty?"
    console.rule("[bold]2. Ask[/bold]")
    ask_args = argparse.Namespace(question=demo_question, top_k=None, mock=args.mock)
    cmd_ask(ask_args)

    console.rule("[bold]3. Evaluate[/bold]")
    eval_args = argparse.Namespace(question=demo_question, top_k=None, mock=args.mock)
    cmd_evaluate(eval_args)


def main():
    parser = argparse.ArgumentParser(description="GenAI Document Assistant — RAG CLI")
    parser.add_argument("--mock", action="store_true", help="Run fully offline, no Azure credentials needed")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Chunk, embed, and index documents")
    p_ingest.add_argument("--docs", default="data/sample_docs")

    p_ask = sub.add_parser("ask", help="Ask a grounded question")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--top-k", type=int, default=None)

    p_eval = sub.add_parser("evaluate", help="Ask a question and score groundedness/relevance")
    p_eval.add_argument("--question", required=True)
    p_eval.add_argument("--top-k", type=int, default=None)

    p_demo = sub.add_parser("demo", help="Run ingest -> ask -> evaluate in one shot")
    p_demo.add_argument("--question", default=None)

    args = parser.parse_args()
    if args.mock:
        _enable_mock()

    dispatch = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)

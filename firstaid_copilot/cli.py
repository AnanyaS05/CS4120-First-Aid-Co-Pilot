from __future__ import annotations

# Command-line entry points for index building, querying, serving, and evaluation.

import argparse
import asyncio
import json
import sys
from typing import Sequence

import uvicorn
from rich.console import Console
from rich.table import Table

from .config import AppConfig
from .evaluation import run_final_evaluation
from .schemas import QueryRequest, QueryResponse, RetrievalHit, StreamEvent
from .service import FirstAidCopilotService

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser and subcommands."""
    parser = argparse.ArgumentParser(prog="firstaid-copilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor")
    subparsers.add_parser("evaluate-tfidf")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--profile", choices=("experiment", "demo"), default="demo")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--models", nargs="*")
    evaluate.add_argument("--skip-generated", action="store_true")
    evaluate.add_argument("--skip-unavailable-models", action="store_true")
    evaluate.add_argument("--no-force-index", action="store_true")
    evaluate.add_argument("--request-timeout-seconds", type=float)

    build_index = subparsers.add_parser("build-index")
    build_index.add_argument("--profile", choices=("experiment", "demo"), required=True)
    build_index.add_argument("--force", action="store_true")

    query = subparsers.add_parser("query")
    query.add_argument("--model", default="qwen3:0.6b")
    query.add_argument("--profile", choices=("experiment", "demo"), default="demo")
    query.add_argument("--text", required=True)
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument("--session-id")
    query.add_argument("--stream", action="store_true")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    return parser


def _render_doctor(service: FirstAidCopilotService) -> None:
    """Print environment, model, and index status."""
    report = service.doctor()
    console.print(f"Python executable: {report.python_executable}")
    console.print(f".venv exists: {report.venv_exists}")
    console.print(f"Ollama available: {report.ollama_available}")

    table = Table(title="Configured Models")
    table.add_column("Model")
    table.add_column("Available")
    for model in report.models:
        table.add_row(model.name, str(model.available))
    console.print(table)
    console.print("Indexes built:")
    console.print(json.dumps(report.indexes_built, indent=2))


def _render_tfidf_evaluation(service: FirstAidCopilotService) -> None:
    """Run TF-IDF selection and print the top configurations."""
    result, path = service.evaluate_tfidf()
    console.print(f"Saved TF-IDF evaluation at {path}")
    console.print(f"Selected weighted score: {result.best_weighted_score:.6f}")

    table = Table(title="Top Dev Configurations Re-ranked on Test")
    table.add_column("Rank")
    table.add_column("N-gram")
    table.add_column("min_df")
    table.add_column("max_df")
    table.add_column("sublinear_tf")
    table.add_column("max_features")
    table.add_column("Dev")
    table.add_column("Test")
    table.add_column("Weighted")
    for index, candidate in enumerate(result.final_candidates, start=1):
        params = candidate.params
        table.add_row(
            str(index),
            str(params.ngram_range),
            str(params.min_df),
            str(params.max_df),
            str(params.sublinear_tf),
            str(params.max_features),
            f"{candidate.dev_score:.6f}",
            f"{candidate.test_score:.6f}",
            f"{candidate.weighted_score:.6f}",
        )
    console.print(table)


def _render_final_evaluation(service: FirstAidCopilotService, args: argparse.Namespace) -> None:
    """Run the final retrieval and generation evaluation."""
    if args.request_timeout_seconds is not None:
        service.config.request_timeout_seconds = args.request_timeout_seconds
    result = run_final_evaluation(
        service,
        models=tuple(args.models) if args.models else None,
        profile=args.profile,
        top_k=args.top_k,
        limit=args.limit,
        skip_generated=args.skip_generated,
        skip_unavailable=args.skip_unavailable_models,
        force_index=not args.no_force_index,
    )
    console.print(f"Saved final evaluation summary at {result.summary_path}")
    console.print(f"Saved TF-IDF test rows at {result.tfidf_rows_path}")
    if result.generated_rows_path:
        console.print(f"Saved generated-answer rows at {result.generated_rows_path}")

    tfidf_metrics = result.summary["tfidf_test"]["metrics"]
    tfidf_table = Table(title="TF-IDF on test.csv")
    tfidf_table.add_column("Metric")
    tfidf_table.add_column("Value")
    for metric_name, value in tfidf_metrics.items():
        tfidf_table.add_row(metric_name, f"{float(value):.6f}")
    console.print(tfidf_table)

    generated_summary = result.summary.get("generated_answers")
    if not generated_summary:
        return

    model_table = Table(title="Generated Answers on generated_answer_eval.csv")
    model_table.add_column("Model")
    model_table.add_column("Answer F1")
    model_table.add_column("ROUGE-L")
    model_table.add_column("Emerg Recall")
    model_table.add_column("Emerg Lang")
    model_table.add_column("Errors")
    for model_name, metrics in generated_summary["metrics_by_model"].items():
        emergency = metrics["emergency_detection"]
        model_table.add_row(
            model_name,
            f"{float(metrics['answer_unigram_f1']):.6f}",
            f"{float(metrics['answer_rouge_l_f1']):.6f}",
            f"{float(emergency['recall']):.6f}",
            f"{float(metrics['emergency_language_inclusion_rate']):.6f}",
            str(metrics["error_count"]),
        )
    console.print(model_table)


def _print_sources(hits: list[RetrievalHit]) -> None:
    """Print retrieved source metadata."""
    if not hits:
        return
    console.print("\n[bold]Sources[/bold]")
    for hit in hits:
        console.print(f"- {hit.doc_id} | {hit.category} | {hit.source} | score={hit.score}")


def _print_warnings(warnings: list[str]) -> None:
    """Print response warnings."""
    if not warnings:
        return
    console.print("\n[bold]Warnings[/bold]")
    for warning in warnings:
        console.print(f"- {warning}")


def _write_stream_text(text: str) -> None:
    """Write streamed token text to the active console or stdout."""
    if console.record:
        console.print(text, end="")
        return
    sys.stdout.write(text)
    sys.stdout.flush()


def _render_query(service: FirstAidCopilotService, args: argparse.Namespace) -> None:
    """Run a single non-streaming query from CLI arguments."""
    request = QueryRequest(
        query=args.text,
        model=args.model,
        profile=args.profile,
        top_k=args.top_k,
        session_id=args.session_id,
    )
    response = service.answer_query(request)
    console.print(f"[bold]Session:[/bold] {response.session_id}")
    console.print(f"[bold]Category:[/bold] {response.risk_category}")
    console.print(f"[bold]Emergency:[/bold] {response.call_emergency_now}")
    console.print("\n[bold]Answer[/bold]")
    console.print(response.answer_text)
    _print_sources(response.sources)
    _print_warnings(response.warnings)


async def _render_query_stream(
    service: FirstAidCopilotService,
    args: argparse.Namespace,
) -> None:
    """Render a streaming query in terminal-friendly sections."""
    request = QueryRequest(
        query=args.text,
        model=args.model,
        profile=args.profile,
        top_k=args.top_k,
        session_id=args.session_id,
    )

    final_response: QueryResponse | None = None
    answer_started = False
    streamed_answer_parts: list[str] = []

    async for event in service.astream_query(request):
        if event.type == "session":
            console.print(f"[bold]Session:[/bold] {event.data['session_id']}")
            console.print(f"[bold]Category:[/bold] {event.data['risk_category']}")
            console.print(f"[bold]Emergency:[/bold] {event.data['call_emergency_now']}")
            continue

        if event.type == "status":
            status_value = str(event.data.get("value", "")).strip()
            if status_value in {"retrieving", "retrying", "fallback"}:
                console.print(f"\n[bold]Status:[/bold] {status_value}")
            continue

        if event.type == "retrieval":
            hits = [
                RetrievalHit.model_validate(payload)
                for payload in event.data.get("hits", [])
            ]
            console.print("\n[bold]Retrieval[/bold]")
            for hit in hits:
                console.print(
                    f"- {hit.doc_id} | {hit.category} | {hit.source} | score={hit.score}"
                )
            continue

        if event.type == "token":
            text = str(event.data.get("text", ""))
            if not text:
                continue
            if not answer_started:
                console.print("\n[bold]Answer[/bold]")
                answer_started = True
            streamed_answer_parts.append(text)
            _write_stream_text(text)
            continue

        if event.type == "final":
            final_response = QueryResponse.model_validate(event.data)
            continue

        if event.type == "error":
            console.print(f"\n[bold red]Error[/bold red] {event.data.get('message', '')}")
            return

    if final_response is None:
        return

    streamed_answer_text = "".join(streamed_answer_parts).strip()
    if answer_started:
        console.print()
    if not answer_started:
        console.print("\n[bold]Answer[/bold]")
        console.print(final_response.answer_text)
    elif streamed_answer_text != final_response.answer_text.strip():
        console.print("\n[bold]Final Answer[/bold]")
        console.print(final_response.answer_text)

    _print_sources(final_response.sources)
    _print_warnings(final_response.warnings)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the requested CLI command."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    service = FirstAidCopilotService(AppConfig())

    if args.command == "doctor":
        _render_doctor(service)
        return 0
    if args.command == "evaluate-tfidf":
        _render_tfidf_evaluation(service)
        return 0
    if args.command == "evaluate":
        _render_final_evaluation(service, args)
        return 0
    if args.command == "build-index":
        index_dir = service.build_index(args.profile, force=args.force)
        console.print(f"Built index at {index_dir}")
        return 0
    if args.command == "query":
        if args.stream:
            asyncio.run(_render_query_stream(service, args))
        else:
            _render_query(service, args)
        return 0
    if args.command == "serve":
        uvicorn.run("firstaid_copilot.api:app", host=args.host, port=args.port, reload=False)
        return 0
    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from typing import Sequence

import uvicorn
from rich.console import Console
from rich.table import Table

from .config import AppConfig
from .schemas import QueryRequest
from .service import FirstAidCopilotService

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="firstaid-copilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor")

    build_index = subparsers.add_parser("build-index")
    build_index.add_argument("--profile", choices=("experiment", "demo"), required=True)
    build_index.add_argument("--force", action="store_true")

    query = subparsers.add_parser("query")
    query.add_argument("--model", default="qwen3:0.6b")
    query.add_argument("--profile", choices=("experiment", "demo"), default="demo")
    query.add_argument("--text", required=True)
    query.add_argument("--top-k", type=int, default=3)
    query.add_argument("--session-id")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    return parser


def _render_doctor(service: FirstAidCopilotService) -> None:
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


def _render_query(service: FirstAidCopilotService, args: argparse.Namespace) -> None:
    response = service.answer_query(
        QueryRequest(
            query=args.text,
            model=args.model,
            profile=args.profile,
            top_k=args.top_k,
            session_id=args.session_id,
        )
    )
    console.print(f"[bold]Session:[/bold] {response.session_id}")
    console.print(f"[bold]Category:[/bold] {response.risk_category}")
    console.print(f"[bold]Emergency:[/bold] {response.call_emergency_now}")
    console.print("\n[bold]Answer[/bold]")
    console.print(response.answer_text)
    console.print("\n[bold]Sources[/bold]")
    for hit in response.sources:
        console.print(f"- {hit.doc_id} | {hit.category} | {hit.source} | score={hit.score}")
    if response.warnings:
        console.print("\n[bold]Warnings[/bold]")
        for warning in response.warnings:
            console.print(f"- {warning}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    service = FirstAidCopilotService(AppConfig())

    if args.command == "doctor":
        _render_doctor(service)
        return 0
    if args.command == "build-index":
        index_dir = service.build_index(args.profile, force=args.force)
        console.print(f"Built index at {index_dir}")
        return 0
    if args.command == "query":
        _render_query(service, args)
        return 0
    if args.command == "serve":
        uvicorn.run("firstaid_copilot.api:app", host=args.host, port=args.port, reload=False)
        return 0
    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (
    ConversationSummary,
    ConversationThread,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    StreamEvent,
)
from .service import FirstAidCopilotService

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _serialize_sse_event(event: StreamEvent) -> str:
    return (
        f"event: {event.type}\n"
        f"data: {json.dumps(event.data, ensure_ascii=False)}\n\n"
    )


def create_app(service: FirstAidCopilotService | None = None) -> FastAPI:
    app = FastAPI(title="First-Aid Co-Pilot", version="0.1.0")
    app.state.service = service or FirstAidCopilotService()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return app.state.service.health_status()

    @app.get("/models")
    def models():
        return app.state.service.model_statuses()

    @app.get("/conversations", response_model=list[ConversationSummary])
    def conversations() -> list[ConversationSummary]:
        return app.state.service.list_conversations()

    @app.get("/conversations/{session_id}", response_model=ConversationThread)
    def conversation_thread(session_id: str) -> ConversationThread:
        try:
            return app.state.service.get_conversation_thread(session_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        try:
            hits = app.state.service.retrieve(request.query, request.profile, k=request.k)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RetrieveResponse(query=request.query, profile=request.profile, hits=hits)

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        try:
            return app.state.service.answer_query(request)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/query/stream")
    async def query_stream(request: QueryRequest) -> StreamingResponse:
        async def event_source():
            try:
                async for event in app.state.service.astream_query(request):
                    yield _serialize_sse_event(event)
            except Exception as exc:
                yield _serialize_sse_event(
                    StreamEvent(type="error", data={"message": str(exc)})
                )

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    app.mount(
        "/",
        StaticFiles(directory=str(STATIC_DIR), html=True),
        name="web-ui",
    )

    return app


app = create_app()

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .schemas import HealthResponse, QueryRequest, QueryResponse, RetrieveRequest, RetrieveResponse
from .service import FirstAidCopilotService


def create_app(service: FirstAidCopilotService | None = None) -> FastAPI:
    app = FastAPI(title="First-Aid Co-Pilot", version="0.1.0")
    app.state.service = service or FirstAidCopilotService()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return app.state.service.health_status()

    @app.get("/models")
    def models():
        return app.state.service.model_statuses()

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

    return app


app = create_app()


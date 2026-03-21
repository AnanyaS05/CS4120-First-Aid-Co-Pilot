from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ProfileName = Literal["experiment", "demo"]


class RetrievalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    question: str
    answer: str
    category: str
    source: str
    split: str
    score: float


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    profile: ProfileName = "demo"
    k: int = Field(default=5, ge=1, le=5)


class RetrieveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    profile: ProfileName
    hits: list[RetrievalHit]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    model: str = "qwen3:0.6b"
    profile: ProfileName = "demo"
    top_k: int = Field(default=3, ge=1, le=5)
    session_id: str | None = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    turn_id: str
    query: str
    model: str
    profile: ProfileName
    risk_category: str
    call_emergency_now: bool
    steps: list[str]
    answer_text: str
    sources: list[RetrievalHit]
    retrieval_hits: list[RetrievalHit]
    warnings: list[str]
    used_retrieval_tool: bool


class ModelStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    available: bool


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    ollama_available: bool
    available_profiles: list[str]
    configured_models: list[ModelStatus]


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_executable: str
    venv_exists: bool
    ollama_available: bool
    models: list[ModelStatus]
    indexes_built: dict[str, bool]


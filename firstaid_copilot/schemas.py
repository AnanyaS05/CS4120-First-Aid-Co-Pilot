from __future__ import annotations

from typing import Any, Literal

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


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    title: str
    preview: str
    updated_at: str
    turn_count: int
    model: str
    profile: ProfileName


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: Literal["user", "assistant"]
    text: str
    timestamp: str
    sources: list[RetrievalHit] | None = None
    warnings: list[str] | None = None


class ConversationThread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    title: str
    messages: list[ConversationMessage]


class ConversationTraceMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["human", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ConversationTurnRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    turn_id: str
    timestamp: str
    user_query: str
    model: str
    profile: ProfileName
    risk_category: str
    retrieval_hits: list[RetrievalHit]
    final_answer: str
    warnings: list[str]
    trace_messages: list[ConversationTraceMessage]


StreamEventType = Literal[
    "session",
    "status",
    "retrieval",
    "token",
    "warning",
    "final",
    "error",
]


class StreamEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: StreamEventType
    data: dict[str, Any]


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


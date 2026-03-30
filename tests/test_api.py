from __future__ import annotations

from fastapi.testclient import TestClient

from firstaid_copilot.api import create_app
from firstaid_copilot.schemas import (
    ConversationMessage,
    ConversationSummary,
    ConversationThread,
    QueryResponse,
    RetrievalHit,
    StreamEvent,
)


class FakeService:
    def health_status(self):
        return {
            "status": "ok",
            "ollama_available": False,
            "available_profiles": ["experiment"],
            "configured_models": [],
        }

    def model_statuses(self):
        return []

    def list_conversations(self):
        return [
            ConversationSummary(
                session_id="session-1",
                title="How do I cool a burn?",
                preview="1. Cool the burn.",
                updated_at="2026-03-30T10:00:00+00:00",
                turn_count=1,
                model="qwen3:0.6b",
                profile="experiment",
            )
        ]

    def get_conversation_thread(self, session_id):
        return ConversationThread(
            session_id=session_id,
            title="How do I cool a burn?",
            messages=[
                ConversationMessage(
                    id="turn-1-user",
                    role="user",
                    text="How do I cool a burn?",
                    timestamp="2026-03-30T10:00:00+00:00",
                ),
                ConversationMessage(
                    id="turn-1-assistant",
                    role="assistant",
                    text="1. Cool the burn.",
                    timestamp="2026-03-30T10:00:00+00:00",
                ),
            ],
        )

    def retrieve(self, query, profile, k):
        return [
            RetrievalHit(
                doc_id="train-00000",
                question="Q",
                answer="A",
                category="burns",
                source="FirstAidQA",
                split="train",
                score=0.9,
            )
        ]

    def answer_query(self, request):
        return QueryResponse(
            session_id="session-1",
            turn_id="turn-1",
            query=request.query,
            model=request.model,
            profile=request.profile,
            risk_category="burns",
            call_emergency_now=False,
            steps=["Cool the burn."],
            answer_text="1. Cool the burn.",
            sources=self.retrieve(request.query, request.profile, request.top_k),
            retrieval_hits=self.retrieve(request.query, request.profile, request.top_k),
            warnings=[],
            used_retrieval_tool=True,
        )

    async def astream_query(self, request):
        response = self.answer_query(request)
        yield StreamEvent(
            type="session",
            data={
                "session_id": response.session_id,
                "turn_id": response.turn_id,
                "model": response.model,
                "profile": response.profile,
                "risk_category": response.risk_category,
                "call_emergency_now": response.call_emergency_now,
            },
        )
        yield StreamEvent(type="token", data={"text": response.answer_text})
        yield StreamEvent(type="final", data=response.model_dump(mode="json"))


def test_api_endpoints_return_expected_shapes():
    client = TestClient(create_app(service=FakeService()))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    retrieve = client.post("/retrieve", json={"query": "burn", "profile": "experiment", "k": 1})
    assert retrieve.status_code == 200
    assert retrieve.json()["hits"][0]["category"] == "burns"

    query = client.post(
        "/query",
        json={"query": "How do I cool a burn?", "model": "qwen3:0.6b", "profile": "experiment"},
    )
    assert query.status_code == 200
    assert query.json()["used_retrieval_tool"] is True

    conversations = client.get("/conversations")
    assert conversations.status_code == 200
    assert conversations.json()[0]["session_id"] == "session-1"

    thread = client.get("/conversations/session-1")
    assert thread.status_code == 200
    assert thread.json()["messages"][0]["role"] == "user"


def test_query_stream_endpoint_returns_sse_events():
    client = TestClient(create_app(service=FakeService()))

    with client.stream(
        "POST",
        "/query/stream",
        json={"query": "How do I cool a burn?", "model": "qwen3:0.6b", "profile": "experiment"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: session" in body
    assert "event: token" in body
    assert "event: final" in body


def test_root_route_returns_web_ui_shell():
    client = TestClient(create_app(service=FakeService()))

    response = client.get("/")

    assert response.status_code == 200
    assert "First-Aid Co-Pilot" in response.text
    assert "conversation-list" in response.text


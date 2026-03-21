from __future__ import annotations

from fastapi.testclient import TestClient

from firstaid_copilot.api import create_app
from firstaid_copilot.schemas import QueryResponse, RetrievalHit


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


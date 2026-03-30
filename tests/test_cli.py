from __future__ import annotations

import argparse
import asyncio

from rich.console import Console

from firstaid_copilot.cli import _render_query_stream
from firstaid_copilot.schemas import QueryResponse, RetrievalHit, StreamEvent


class FakeStreamingService:
    async def astream_query(self, request):
        hit = RetrievalHit(
            doc_id="train-00000",
            question="Q",
            answer="A",
            category="burns",
            source="FirstAidQA",
            split="train",
            score=0.9,
        )
        response = QueryResponse(
            session_id="session-1",
            turn_id="turn-1",
            query=request.query,
            model=request.model,
            profile=request.profile,
            risk_category="burns",
            call_emergency_now=False,
            steps=["Cool the burn."],
            answer_text="1. Cool the burn.",
            sources=[hit],
            retrieval_hits=[hit],
            warnings=[],
            used_retrieval_tool=True,
        )
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
        yield StreamEvent(type="status", data={"value": "retrieving"})
        yield StreamEvent(type="retrieval", data={"hits": [hit.model_dump(mode="json")]})
        yield StreamEvent(type="token", data={"text": response.answer_text})
        yield StreamEvent(type="final", data=response.model_dump(mode="json"))


def test_render_query_stream_outputs_single_answer_and_sources(monkeypatch):
    recording_console = Console(record=True, force_terminal=False, width=120)
    monkeypatch.setattr("firstaid_copilot.cli.console", recording_console)

    args = argparse.Namespace(
        text="How do I cool a burn?",
        model="qwen3:0.6b",
        profile="experiment",
        top_k=3,
        session_id=None,
        stream=True,
    )

    asyncio.run(_render_query_stream(FakeStreamingService(), args))
    output = recording_console.export_text()

    assert "Session:" in output
    assert "Retrieval" in output
    assert "Answer" in output
    assert "Sources" in output
    assert output.count("1. Cool the burn.") == 1

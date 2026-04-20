from __future__ import annotations

# Service tests isolate agent behavior with fake LangChain agents and streams.

import asyncio
import json

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from firstaid_copilot.schemas import QueryRequest, QueryResponse
from firstaid_copilot.service import FirstAidCopilotService


def test_build_index_creates_expected_files(temp_config):
    service = FirstAidCopilotService(temp_config)
    index_dir = service.build_index("experiment")

    assert (index_dir / "vectorizer.joblib").exists()
    assert (index_dir / "doc_matrix.npz").exists()
    assert (index_dir / "documents.jsonl").exists()
    assert (index_dir / "config.json").exists()
    assert (temp_config.evaluations_dir / "tfidf_selection.json").exists()

    metadata = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    assert metadata["tuning"]["selection_strategy"] == "top_dev_then_weighted_dev_test"
    assert metadata["tuning"]["top_n"] == 5
    assert metadata["tuning"]["dev_weight"] == 0.30
    assert metadata["tuning"]["test_weight"] == 0.70
    assert len(metadata["tuning"]["final_candidates"]) == 5


def test_answer_query_logs_turn_trace_with_tool_messages(monkeypatch, temp_config):
    captured_prompts = []
    fake_tools = []

    class FakeAgent:
        # Simulate a full tool-call trace so the logger can persist it.
        def invoke(self, payload):
            tool_output = fake_tools[0].invoke(
                {"query": "How should I treat severe bleeding?", "k": 3}
            )
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_first_aid_knowledge",
                                "args": {"query": "How should I treat severe bleeding?"},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        tool_call_id="call-1",
                        name="search_first_aid_knowledge",
                        content=tool_output,
                    ),
                    AIMessage(content="1. Call emergency services.\n2. Apply direct pressure."),
                ]
            }

    def fake_create_agent(*, model, tools, system_prompt, **kwargs):
        captured_prompts.append(system_prompt)
        fake_tools[:] = list(tools)
        return FakeAgent()

    monkeypatch.setattr("firstaid_copilot.service.create_agent", fake_create_agent)
    service = FirstAidCopilotService(temp_config)
    service.build_index("experiment")

    response = service.answer_query(
        QueryRequest(
            query="How should I treat severe bleeding?",
            model="qwen3:0.6b",
            profile="experiment",
        )
    )

    assert captured_prompts
    prompt = captured_prompts[0].lower()
    assert "conversational follow-up" in prompt
    assert "do not fabricate" in prompt
    assert response.used_retrieval_tool is True
    assert response.sources

    log_path = next(temp_config.conversations_dir.glob("session-*.jsonl"))
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert [message["role"] for message in record["trace_messages"]] == [
        "human",
        "assistant",
        "tool",
        "assistant",
    ]


def test_model_statuses_accept_latest_tag_alias(monkeypatch, temp_config):
    service = FirstAidCopilotService(temp_config)
    monkeypatch.setattr(
        service,
        "_ollama_models",
        lambda: (
            True,
            {
                "functiongemma:latest",
                "qwen3:0.6b",
                "qwen3.5:0.8b",
                "granite4:350m",
            },
        ),
    )

    statuses = {status.name: status.available for status in service.model_statuses()}

    assert statuses["functiongemma"] is True
    assert statuses["qwen3:0.6b"] is True
    assert statuses["qwen3.5:0.8b"] is True
    assert statuses["granite4:350m"] is True


def test_model_statuses_accept_case_insensitive_ollama_names(monkeypatch, temp_config):
    service = FirstAidCopilotService(temp_config)
    monkeypatch.setattr(
        service,
        "_ollama_models",
        lambda: (
            True,
            {
                "functiongemma:latest",
                "qwen3:0.6b",
                "qwen3.5:0.8b",
                "granite4:350M",
            },
        ),
    )

    statuses = {status.name: status.available for status in service.model_statuses()}

    assert statuses["functiongemma"] is True
    assert statuses["qwen3:0.6b"] is True
    assert statuses["qwen3.5:0.8b"] is True
    assert statuses["granite4:350m"] is True


def test_answer_query_uses_previous_conversation_context(monkeypatch, temp_config):
    service = FirstAidCopilotService(temp_config)
    service.build_index("experiment")
    service.logger.log_turn(
        "session-1",
        {
            "session_id": "session-1",
            "turn_id": "turn-1",
            "timestamp": "2026-03-30T10:00:00+00:00",
            "user_query": "I got stung by a bee",
            "model": "qwen3:0.6b",
            "profile": "experiment",
            "risk_category": "allergies",
            "retrieval_hits": [],
            "final_answer": "Use a cold compress and monitor for worsening symptoms.",
            "warnings": [],
            "trace_messages": [
                {"role": "human", "content": "I got stung by a bee"},
                {
                    "role": "assistant",
                    "content": "Use a cold compress and monitor for worsening symptoms.",
                },
            ],
        },
    )

    captured_messages = []

    class FakeAgent:
        def invoke(self, payload):
            captured_messages[:] = payload["messages"]
            return {"messages": [AIMessage(content="You're welcome. Keep monitoring the sting site.")]}

    monkeypatch.setattr(
        "firstaid_copilot.service.create_agent",
        lambda **kwargs: FakeAgent(),
    )

    response = service.answer_query(
        QueryRequest(
            query="Thanks, I am feeling better now",
            model="qwen3:0.6b",
            profile="experiment",
            session_id="session-1",
        )
    )

    assert [type(message).__name__ for message in captured_messages] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert captured_messages[0].content == "I got stung by a bee"
    assert captured_messages[1].content == "Use a cold compress and monitor for worsening symptoms."
    assert captured_messages[2].content == "Thanks, I am feeling better now"
    assert response.used_retrieval_tool is False
    assert response.sources == []


def test_answer_query_does_not_retry_only_because_tool_was_not_used(monkeypatch, temp_config):
    attempt_count = {"value": 0}

    class FakeAgent:
        def invoke(self, payload):
            attempt_count["value"] += 1
            return {
                "messages": [
                    AIMessage(content="You're welcome. Keep the area clean and keep monitoring it.")
                ]
            }

    monkeypatch.setattr(
        "firstaid_copilot.service.create_agent",
        lambda **kwargs: FakeAgent(),
    )
    service = FirstAidCopilotService(temp_config)
    service.build_index("experiment")

    response = service.answer_query(
        QueryRequest(
            query="Thanks, that helps",
            model="qwen3:0.6b",
            profile="experiment",
        )
    )

    assert attempt_count["value"] == 1
    assert response.used_retrieval_tool is False
    assert response.sources == []
    assert all("retrieval tool was not used" not in warning.lower() for warning in response.warnings)


def test_astream_query_streams_tokens_without_tool_when_history_is_enough(monkeypatch, temp_config):
    service = FirstAidCopilotService(temp_config)
    service.build_index("experiment")
    service.logger.log_turn(
        "session-1",
        {
            "session_id": "session-1",
            "turn_id": "turn-1",
            "timestamp": "2026-03-30T10:00:00+00:00",
            "user_query": "I got stung by a bee",
            "model": "qwen3:0.6b",
            "profile": "experiment",
            "risk_category": "allergies",
            "retrieval_hits": [],
            "final_answer": "Use a cold compress and monitor for worsening symptoms.",
            "warnings": [],
            "trace_messages": [
                {"role": "human", "content": "I got stung by a bee"},
                {
                    "role": "assistant",
                    "content": "Use a cold compress and monitor for worsening symptoms.",
                },
            ],
        },
    )

    class FakeAgent:
        async def astream(self, payload, stream_mode):
            assert [type(message).__name__ for message in payload["messages"]] == [
                "HumanMessage",
                "AIMessage",
                "HumanMessage",
            ]
            yield (
                "messages",
                (
                    AIMessageChunk(content="You're welcome. "),
                    {"langgraph_node": "agent"},
                ),
            )
            yield (
                "messages",
                (
                    AIMessageChunk(content="Keep monitoring the sting site."),
                    {"langgraph_node": "agent"},
                ),
            )

    monkeypatch.setattr(
        "firstaid_copilot.service.create_agent",
        lambda **kwargs: FakeAgent(),
    )

    async def collect_events():
        events = []
        async for event in service.astream_query(
            QueryRequest(
                query="Thanks, I am feeling better now",
                model="qwen3:0.6b",
                profile="experiment",
                session_id="session-1",
            )
        ):
            events.append(event)
        return events

    events = asyncio.run(collect_events())
    token_text = "".join(event.data["text"] for event in events if event.type == "token")
    response = QueryResponse.model_validate(
        next(event for event in events if event.type == "final").data
    )

    assert token_text == "You're welcome. Keep monitoring the sting site."
    assert not any(event.type == "retrieval" for event in events)
    assert response.used_retrieval_tool is False
    assert response.sources == []


def test_astream_query_retries_when_first_attempt_is_empty(monkeypatch, temp_config):
    attempt_number = {"value": 0}

    class FakeAgent:
        # First stream is empty; the second stream proves retry behavior.
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt

        async def astream(self, payload, stream_mode):
            if self.attempt == 1:
                return
            yield {
                "type": "messages",
                "ns": (),
                "data": (
                    AIMessageChunk(content="Use an epinephrine auto-injector if available."),
                    {"langgraph_node": "agent"},
                ),
            }
            yield {
                "type": "messages",
                "ns": (),
                "data": (
                    AIMessageChunk(content=" Call emergency services immediately."),
                    {"langgraph_node": "agent"},
                ),
            }

    def fake_create_agent(**kwargs):
        attempt_number["value"] += 1
        return FakeAgent(attempt_number["value"])

    monkeypatch.setattr("firstaid_copilot.service.create_agent", fake_create_agent)
    service = FirstAidCopilotService(temp_config)
    service.build_index("experiment")

    async def collect_events():
        events = []
        async for event in service.astream_query(
            QueryRequest(
                query="I got stung by a bee, and now I have a rash and difficulty breathing",
                model="qwen3:0.6b",
                profile="experiment",
            )
        ):
            events.append(event)
        return events

    events = asyncio.run(collect_events())
    token_text = "".join(event.data["text"] for event in events if event.type == "token")
    response = QueryResponse.model_validate(
        next(event for event in events if event.type == "final").data
    )

    assert attempt_number["value"] == 2
    assert any(
        event.type == "status" and event.data.get("value") == "retrying"
        for event in events
    )
    assert "epinephrine auto-injector" in token_text
    assert response.used_retrieval_tool is False

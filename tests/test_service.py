from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from firstaid_copilot.schemas import QueryRequest
from firstaid_copilot.service import FirstAidCopilotService


def test_build_index_creates_expected_files(temp_config):
    service = FirstAidCopilotService(temp_config)
    index_dir = service.build_index("experiment")

    assert (index_dir / "vectorizer.joblib").exists()
    assert (index_dir / "doc_matrix.npz").exists()
    assert (index_dir / "documents.jsonl").exists()
    assert (index_dir / "config.json").exists()


def test_answer_query_uses_retrieval_prompt_and_logs(monkeypatch, temp_config):
    captured_prompts = []

    class FakeAgent:
        def invoke(self, payload):
            tool_output = fake_tools[0].invoke({"query": payload["messages"][0]["content"], "k": 3})
            assert tool_output
            return {
                "messages": [
                    ToolMessage(tool_call_id="call-1", name="search_first_aid_knowledge", content=tool_output),
                    AIMessage(content="1. Call emergency services.\n2. Apply direct pressure."),
                ]
            }

    fake_tools = []

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
    assert "must call the tool search_first_aid_knowledge" in captured_prompts[0].lower()
    assert response.used_retrieval_tool is True
    assert response.sources
    logs = list(temp_config.conversations_dir.glob("session-*.jsonl"))
    assert logs


def test_model_statuses_accept_latest_tag_alias(monkeypatch, temp_config):
    service = FirstAidCopilotService(temp_config)
    monkeypatch.setattr(
        service,
        "_ollama_models",
        lambda: (True, {"functiongemma:latest", "qwen3:0.6b", "granite4:350m"}),
    )

    statuses = {status.name: status.available for status in service.model_statuses()}

    assert statuses["functiongemma"] is True
    assert statuses["qwen3:0.6b"] is True
    assert statuses["granite4:350m"] is True

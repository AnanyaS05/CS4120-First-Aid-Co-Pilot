from __future__ import annotations

# Conversation tests cover JSONL persistence and prompt-history reconstruction.

from firstaid_copilot.conversation import ConversationLogger


def test_conversation_logger_lists_summaries_newest_first(temp_config):
    logger = ConversationLogger(temp_config.conversations_dir)
    logger.log_turn(
        "session-a",
        {
            "session_id": "session-a",
            "turn_id": "turn-a",
            "timestamp": "2026-03-30T10:00:00+00:00",
            "user_query": "How do I treat a burn?",
            "model": "qwen3:0.6b",
            "profile": "demo",
            "risk_category": "burns",
            "retrieval_hits": [],
            "final_answer": "Cool the burn under running water.",
            "warnings": [],
            "trace_messages": [
                {"role": "human", "content": "How do I treat a burn?"},
                {"role": "assistant", "content": "Cool the burn under running water."},
            ],
        },
    )
    logger.log_turn(
        "session-b",
        {
            "session_id": "session-b",
            "turn_id": "turn-b",
            "timestamp": "2026-03-30T11:00:00+00:00",
            "user_query": "How do I help someone choking?",
            "model": "functiongemma",
            "profile": "experiment",
            "risk_category": "choking",
            "retrieval_hits": [],
            "final_answer": "Encourage coughing and call emergency services.",
            "warnings": ["Emergency escalation language should appear early in the answer."],
            "trace_messages": [
                {"role": "human", "content": "How do I help someone choking?"},
                {
                    "role": "assistant",
                    "content": "Encourage coughing and call emergency services.",
                },
            ],
        },
    )

    summaries = logger.list_conversations()

    assert [summary.session_id for summary in summaries] == ["session-b", "session-a"]
    assert summaries[0].title == "How do I help someone choking?"
    assert summaries[0].preview == "Encourage coughing and call emergency services."


def test_conversation_logger_reconstructs_thread_messages(temp_config):
    logger = ConversationLogger(temp_config.conversations_dir)
    logger.log_turn(
        "session-a",
        {
            "session_id": "session-a",
            "turn_id": "turn-a",
            "timestamp": "2026-03-30T10:00:00+00:00",
            "user_query": "How do I treat a burn?",
            "model": "qwen3:0.6b",
            "profile": "demo",
            "risk_category": "burns",
            "retrieval_hits": [
                {
                    "doc_id": "train-0001",
                    "question": "Q",
                    "answer": "A",
                    "category": "burns",
                    "source": "FirstAidQA",
                    "split": "train",
                    "score": 0.9,
                }
            ],
            "final_answer": "Cool the burn under running water.",
            "warnings": ["Watch for worsening pain."],
            "trace_messages": [
                {"role": "human", "content": "How do I treat a burn?"},
                {"role": "assistant", "content": "Cool the burn under running water."},
            ],
        },
    )

    thread = logger.load_conversation("session-a")

    assert thread.session_id == "session-a"
    assert [message.role for message in thread.messages] == ["user", "assistant"]
    assert thread.messages[0].text == "How do I treat a burn?"
    assert thread.messages[1].text == "Cool the burn under running water."
    assert thread.messages[1].sources[0].category == "burns"
    assert thread.messages[1].warnings == ["Watch for worsening pain."]


def test_conversation_logger_loads_last_five_human_rounds(temp_config):
    logger = ConversationLogger(temp_config.conversations_dir)
    for index in range(1, 8):
        logger.log_turn(
            "session-a",
            {
                "session_id": "session-a",
                "turn_id": f"turn-{index}",
                "timestamp": f"2026-03-30T1{index}:00:00+00:00",
                "user_query": f"user {index}",
                "model": "qwen3:0.6b",
                "profile": "demo",
                "risk_category": "other",
                "retrieval_hits": [],
                "final_answer": f"assistant {index}",
                "warnings": [],
                "trace_messages": [
                    {"role": "human", "content": f"user {index}"},
                    {"role": "assistant", "content": f"assistant {index}"},
                ],
            },
        )

    history = logger.load_context_messages("session-a", max_human_messages=5)

    assert [message.content for message in history] == [
        "user 3",
        "assistant 3",
        "user 4",
        "assistant 4",
        "user 5",
        "assistant 5",
        "user 6",
        "assistant 6",
        "user 7",
        "assistant 7",
    ]

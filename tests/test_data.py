from __future__ import annotations

# Data tests verify the retrieval document template and metadata shape.

from firstaid_copilot.data import build_documents, load_split_dataframe


def test_build_documents_uses_weighted_question_template(temp_config):
    frame = load_split_dataframe(temp_config, "train")
    documents = build_documents(frame, "train")

    assert documents[0].metadata["doc_id"] == "train-00000"
    assert documents[0].page_content.count("Question:") == 2
    assert "Category: severe_bleeding" in documents[0].page_content


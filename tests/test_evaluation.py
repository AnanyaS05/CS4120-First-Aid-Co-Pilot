from __future__ import annotations

import pytest

from firstaid_copilot.data import load_split_dataframe
from firstaid_copilot.evaluation import (
    evaluate_generated_answers,
    evaluate_tfidf_on_test,
    rouge_l_f1,
)
from firstaid_copilot.safety import EMERGENCY_CATEGORIES
from firstaid_copilot.schemas import ModelStatus, QueryResponse, RetrievalHit
from firstaid_copilot.service import FirstAidCopilotService


def test_rouge_l_f1_rewards_longest_common_subsequence():
    score = rouge_l_f1("apply direct pressure now", "apply firm direct pressure")

    assert score == pytest.approx(0.75)


def test_evaluate_tfidf_on_test_returns_expected_metric_keys(temp_config):
    service = FirstAidCopilotService(temp_config)
    selection_result, _path = service.evaluate_tfidf()

    summary, rows = evaluate_tfidf_on_test(temp_config, selection_result, top_k=3)

    assert len(rows) == 2
    assert summary["split"] == "test"
    assert summary["metrics"]["category_hit_at_1"] >= 0
    assert "top5_best_answer_unigram_f1" in summary["metrics"]
    assert "mrr_at_5" in summary["metrics"]


def test_evaluate_generated_answers_summarizes_model_outputs(temp_config):
    eval_frame = load_split_dataframe(temp_config, "generated_answer_eval")
    answers_by_query = {
        row["question"]: row
        for _index, row in eval_frame.iterrows()
    }

    class FakeService:
        config = temp_config

        def model_statuses(self):
            return [ModelStatus(name="qwen3:0.6b", available=True)]

        def answer_query(self, request):
            row = answers_by_query[request.query]
            category = str(row["category"])
            answer = str(row["answer"])
            return QueryResponse(
                session_id=request.session_id or "session-1",
                turn_id="turn-1",
                query=request.query,
                model=request.model,
                profile=request.profile,
                risk_category=category,
                call_emergency_now=category in EMERGENCY_CATEGORIES,
                steps=[answer],
                answer_text=answer,
                sources=[
                    RetrievalHit(
                        doc_id="train-00000",
                        question=request.query,
                        answer=answer,
                        category=category,
                        source="FirstAidQA",
                        split="train",
                        score=1.0,
                    )
                ],
                retrieval_hits=[],
                warnings=[],
                used_retrieval_tool=True,
            )

    summary, rows = evaluate_generated_answers(
        FakeService(),
        models=("qwen3:0.6b",),
        profile="experiment",
        top_k=3,
    )

    metrics = summary["metrics_by_model"]["qwen3:0.6b"]
    assert len(rows) == len(eval_frame)
    assert metrics["answer_unigram_f1"] == pytest.approx(1.0)
    assert metrics["source_category_hit_at_1"] == pytest.approx(1.0)
    assert metrics["used_retrieval_tool_rate"] == pytest.approx(1.0)

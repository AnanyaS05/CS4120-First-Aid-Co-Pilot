from __future__ import annotations

# Tuning tests lock down the TF-IDF grid and dev/test re-ranking behavior.

import pytest

from firstaid_copilot.tuning import (
    FALLBACK_HYPERPARAMETERS,
    iter_search_grid,
    select_tfidf_with_dev_test,
    tune_tfidf,
)


def test_tuning_search_grid_has_expected_size():
    assert len(list(iter_search_grid())) == 96


def test_tuning_returns_a_valid_result():
    result = tune_tfidf(
        train_texts=[
            "Question: bleeding Answer: apply pressure Category: severe_bleeding Source: FirstAidQA",
            "Question: choking Answer: give back blows Category: choking Source: FirstAidQA",
        ],
        train_answers=["apply pressure", "give back blows"],
        train_categories=["severe_bleeding", "choking"],
        train_doc_ids=["train-00000", "train-00001"],
        dev_queries=["How do I help with choking?"],
        dev_answers=["give back blows"],
        dev_categories=["choking"],
    )

    assert result.candidate_count == 96
    assert result.best_score > 0
    assert result.best_params.norm == "l2"
    assert result.best_params.stop_words == FALLBACK_HYPERPARAMETERS.stop_words


def test_dev_test_selection_reranks_top_five_candidates():
    result = select_tfidf_with_dev_test(
        train_texts=[
            "Question: bleeding Answer: apply pressure Category: severe_bleeding Source: FirstAidQA",
            "Question: choking Answer: give back blows Category: choking Source: FirstAidQA",
            "Question: burn Answer: cool with water Category: burns Source: FirstAidQA",
        ],
        train_answers=["apply pressure", "give back blows", "cool with water"],
        train_categories=["severe_bleeding", "choking", "burns"],
        dev_queries=[
            "How do I help with choking?",
            "How should I respond to bleeding?",
        ],
        dev_answers=["give back blows", "apply pressure"],
        dev_categories=["choking", "severe_bleeding"],
        test_queries=["What first aid helps a burn?"],
        test_answers=["cool with water"],
        test_categories=["burns"],
    )

    assert result.candidate_count == 96
    assert len(result.final_candidates) == 5
    assert result.dev_weight == pytest.approx(0.30)
    assert result.test_weight == pytest.approx(0.70)
    assert result.best_params == result.final_candidates[0].params
    for candidate in result.final_candidates:
        assert candidate.weighted_score == pytest.approx(
            0.30 * candidate.dev_score + 0.70 * candidate.test_score
        )

from __future__ import annotations

from firstaid_copilot.tuning import FALLBACK_HYPERPARAMETERS, iter_search_grid, tune_tfidf


def test_tuning_search_grid_has_expected_size():
    assert len(list(iter_search_grid())) == 72


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

    assert result.candidate_count == 72
    assert result.best_score > 0
    assert result.best_params.norm == "l2"
    assert result.best_params.stop_words == FALLBACK_HYPERPARAMETERS.stop_words


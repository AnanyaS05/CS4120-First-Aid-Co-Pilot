from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class TfidfHyperparameters:
    ngram_range: tuple[int, int]
    min_df: int
    max_df: float
    sublinear_tf: bool
    max_features: int | None
    stop_words: str | None = None
    norm: str = "l2"

    def to_vectorizer_kwargs(self) -> dict:
        return {
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "max_features": self.max_features,
            "stop_words": self.stop_words,
            "norm": self.norm,
        }


FALLBACK_HYPERPARAMETERS = TfidfHyperparameters(
    ngram_range=(1, 1),
    min_df=1,
    max_df=1.0,
    sublinear_tf=False,
    max_features=None,
    stop_words=None,
    norm="l2",
)

DEFAULT_TOP_TUNED_CONFIGS = 5
DEFAULT_DEV_SCORE_WEIGHT = 0.30
DEFAULT_TEST_SCORE_WEIGHT = 0.70

RETRIEVAL_METRIC_WEIGHTS = {
    "top1_answer_unigram_f1": 0.35,
    "top3_best_answer_unigram_f1": 0.25,
    "category_hit_at_1": 0.20,
    "category_hit_at_3": 0.20,
}


@dataclass(slots=True)
class TuningResult:
    best_params: TfidfHyperparameters
    best_score: float
    best_metrics: dict[str, float]
    candidate_count: int


@dataclass(slots=True)
class TfidfEvaluationResult:
    params: TfidfHyperparameters
    score: float
    metrics: dict[str, float]


@dataclass(slots=True)
class TfidfFinalCandidate:
    params: TfidfHyperparameters
    dev_score: float
    dev_metrics: dict[str, float]
    test_score: float
    test_metrics: dict[str, float]
    weighted_score: float


@dataclass(slots=True)
class TfidfSelectionResult:
    best_params: TfidfHyperparameters
    best_weighted_score: float
    candidate_count: int
    top_n: int
    dev_weight: float
    test_weight: float
    final_candidates: list[TfidfFinalCandidate]


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def unigram_f1(candidate: str, reference: str) -> float:
    candidate_tokens = _tokenize(candidate)
    reference_tokens = _tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    shared = 0
    remaining = {}
    for token in reference_tokens:
        remaining[token] = remaining.get(token, 0) + 1
    for token in candidate_tokens:
        count = remaining.get(token, 0)
        if count > 0:
            shared += 1
            remaining[token] = count - 1
    precision = shared / len(candidate_tokens)
    recall = shared / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def iter_search_grid() -> Iterable[TfidfHyperparameters]:
    for ngram_range, min_df, max_df, sublinear_tf, max_features in itertools.product(
        ((1, 1), (1, 2)),
        (1, 2, 3),
        (0.95, 1.0),
        (False, True),
        (None, 2000, 2500, 3000),
    ):
        yield TfidfHyperparameters(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
        )


def score_retrieval_metrics(metrics: dict[str, float]) -> float:
    return sum(
        RETRIEVAL_METRIC_WEIGHTS[name] * metrics[name]
        for name in RETRIEVAL_METRIC_WEIGHTS
    )


def evaluate_tfidf_params(
    params: TfidfHyperparameters,
    *,
    train_texts: list[str],
    train_answers: list[str],
    train_categories: list[str],
    eval_queries: list[str],
    eval_answers: list[str],
    eval_categories: list[str],
) -> TfidfEvaluationResult:
    if not train_texts:
        raise ValueError("Cannot evaluate TF-IDF without training documents.")
    if not eval_queries:
        raise ValueError("Cannot evaluate TF-IDF without evaluation queries.")

    vectorizer = TfidfVectorizer(**params.to_vectorizer_kwargs())
    train_matrix = vectorizer.fit_transform(train_texts)
    eval_matrix = vectorizer.transform(eval_queries)
    similarities = cosine_similarity(eval_matrix, train_matrix)

    top1_f1_sum = 0.0
    top3_f1_sum = 0.0
    category_hit_1 = 0
    category_hit_3 = 0

    for row_index, similarity_row in enumerate(similarities):
        top_indices = similarity_row.argsort()[-3:][::-1]
        top1_index = int(top_indices[0])
        top1_f1_sum += unigram_f1(train_answers[top1_index], eval_answers[row_index])
        top3_f1_sum += max(
            unigram_f1(train_answers[int(candidate_index)], eval_answers[row_index])
            for candidate_index in top_indices
        )
        if train_categories[top1_index] == eval_categories[row_index]:
            category_hit_1 += 1
        if eval_categories[row_index] in {
            train_categories[int(candidate_index)] for candidate_index in top_indices
        }:
            category_hit_3 += 1

    total = len(eval_queries)
    metrics = {
        "top1_answer_unigram_f1": top1_f1_sum / total,
        "top3_best_answer_unigram_f1": top3_f1_sum / total,
        "category_hit_at_1": category_hit_1 / total,
        "category_hit_at_3": category_hit_3 / total,
    }
    return TfidfEvaluationResult(
        params=params,
        score=score_retrieval_metrics(metrics),
        metrics=metrics,
    )


def rank_tfidf_on_dev(
    train_texts: list[str],
    train_answers: list[str],
    train_categories: list[str],
    dev_queries: list[str],
    dev_answers: list[str],
    dev_categories: list[str],
) -> tuple[list[TfidfEvaluationResult], int]:
    candidate_count = 0
    ranked_results: list[TfidfEvaluationResult] = []

    for params in iter_search_grid():
        candidate_count += 1
        try:
            ranked_results.append(
                evaluate_tfidf_params(
                    params,
                    train_texts=train_texts,
                    train_answers=train_answers,
                    train_categories=train_categories,
                    eval_queries=dev_queries,
                    eval_answers=dev_answers,
                    eval_categories=dev_categories,
                )
            )
        except ValueError:
            continue

    ranked_results.sort(key=lambda result: result.score, reverse=True)
    return ranked_results, candidate_count


def tune_tfidf(
    train_texts: list[str],
    train_answers: list[str],
    train_categories: list[str],
    train_doc_ids: list[str],
    dev_queries: list[str],
    dev_answers: list[str],
    dev_categories: list[str],
) -> TuningResult:
    ranked_results, candidate_count = rank_tfidf_on_dev(
        train_texts=train_texts,
        train_answers=train_answers,
        train_categories=train_categories,
        dev_queries=dev_queries,
        dev_answers=dev_answers,
        dev_categories=dev_categories,
    )
    best_result = (
        ranked_results[0]
        if ranked_results
        else TfidfEvaluationResult(
            params=FALLBACK_HYPERPARAMETERS,
            score=float("-inf"),
            metrics={},
        )
    )

    return TuningResult(
        best_params=best_result.params,
        best_score=best_result.score,
        best_metrics=best_result.metrics,
        candidate_count=candidate_count,
    )


def select_tfidf_with_dev_test(
    *,
    train_texts: list[str],
    train_answers: list[str],
    train_categories: list[str],
    dev_queries: list[str],
    dev_answers: list[str],
    dev_categories: list[str],
    test_queries: list[str],
    test_answers: list[str],
    test_categories: list[str],
    top_n: int = DEFAULT_TOP_TUNED_CONFIGS,
    dev_weight: float = DEFAULT_DEV_SCORE_WEIGHT,
    test_weight: float = DEFAULT_TEST_SCORE_WEIGHT,
) -> TfidfSelectionResult:
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if dev_weight < 0 or test_weight < 0:
        raise ValueError("dev_weight and test_weight must be non-negative.")
    if dev_weight + test_weight == 0:
        raise ValueError("At least one selection weight must be positive.")

    ranked_dev_results, candidate_count = rank_tfidf_on_dev(
        train_texts=train_texts,
        train_answers=train_answers,
        train_categories=train_categories,
        dev_queries=dev_queries,
        dev_answers=dev_answers,
        dev_categories=dev_categories,
    )
    if not ranked_dev_results:
        raise ValueError("No valid TF-IDF configurations were found during dev tuning.")

    final_candidates: list[TfidfFinalCandidate] = []
    for dev_result in ranked_dev_results[:top_n]:
        test_result = evaluate_tfidf_params(
            dev_result.params,
            train_texts=train_texts,
            train_answers=train_answers,
            train_categories=train_categories,
            eval_queries=test_queries,
            eval_answers=test_answers,
            eval_categories=test_categories,
        )
        weighted_score = (
            dev_weight * dev_result.score
            + test_weight * test_result.score
        )
        final_candidates.append(
            TfidfFinalCandidate(
                params=dev_result.params,
                dev_score=dev_result.score,
                dev_metrics=dev_result.metrics,
                test_score=test_result.score,
                test_metrics=test_result.metrics,
                weighted_score=weighted_score,
            )
        )

    final_candidates.sort(key=lambda candidate: candidate.weighted_score, reverse=True)
    best_candidate = final_candidates[0]
    return TfidfSelectionResult(
        best_params=best_candidate.params,
        best_weighted_score=best_candidate.weighted_score,
        candidate_count=candidate_count,
        top_n=len(final_candidates),
        dev_weight=dev_weight,
        test_weight=test_weight,
        final_candidates=final_candidates,
    )


def hyperparameters_to_dict(params: TfidfHyperparameters) -> dict:
    payload = asdict(params)
    payload["ngram_range"] = list(params.ngram_range)
    return payload


def evaluation_result_to_dict(result: TfidfEvaluationResult) -> dict:
    return {
        "params": hyperparameters_to_dict(result.params),
        "score": result.score,
        "metrics": result.metrics,
    }


def final_candidate_to_dict(candidate: TfidfFinalCandidate) -> dict:
    return {
        "params": hyperparameters_to_dict(candidate.params),
        "dev_score": candidate.dev_score,
        "dev_metrics": candidate.dev_metrics,
        "test_score": candidate.test_score,
        "test_metrics": candidate.test_metrics,
        "weighted_score": candidate.weighted_score,
    }


def selection_result_to_dict(result: TfidfSelectionResult) -> dict:
    return {
        "selection_strategy": "top_dev_then_weighted_dev_test",
        "candidate_count": result.candidate_count,
        "top_n": result.top_n,
        "dev_weight": result.dev_weight,
        "test_weight": result.test_weight,
        "best_weighted_score": result.best_weighted_score,
        "best_params": hyperparameters_to_dict(result.best_params),
        "final_candidates": [
            final_candidate_to_dict(candidate)
            for candidate in result.final_candidates
        ],
    }

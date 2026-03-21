from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import RetrievalHit


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


@dataclass(slots=True)
class TuningResult:
    best_params: TfidfHyperparameters
    best_score: float
    best_metrics: dict[str, float]
    candidate_count: int


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
        (None, 10000, 20000),
    ):
        yield TfidfHyperparameters(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            max_features=max_features,
        )


def tune_tfidf(
    train_texts: list[str],
    train_answers: list[str],
    train_categories: list[str],
    train_doc_ids: list[str],
    dev_queries: list[str],
    dev_answers: list[str],
    dev_categories: list[str],
) -> TuningResult:
    best_params = FALLBACK_HYPERPARAMETERS
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    candidate_count = 0

    for params in iter_search_grid():
        candidate_count += 1
        vectorizer = TfidfVectorizer(**params.to_vectorizer_kwargs())
        try:
            train_matrix = vectorizer.fit_transform(train_texts)
            dev_matrix = vectorizer.transform(dev_queries)
        except ValueError:
            continue
        similarities = cosine_similarity(dev_matrix, train_matrix)

        top1_f1_sum = 0.0
        top3_f1_sum = 0.0
        category_hit_1 = 0
        category_hit_3 = 0

        for row_index, similarity_row in enumerate(similarities):
            top_indices = similarity_row.argsort()[-3:][::-1]
            top1_index = int(top_indices[0])
            top1_f1_sum += unigram_f1(train_answers[top1_index], dev_answers[row_index])
            top3_f1_sum += max(
                unigram_f1(train_answers[int(candidate_index)], dev_answers[row_index])
                for candidate_index in top_indices
            )
            if train_categories[top1_index] == dev_categories[row_index]:
                category_hit_1 += 1
            if dev_categories[row_index] in {
                train_categories[int(candidate_index)] for candidate_index in top_indices
            }:
                category_hit_3 += 1

        total = len(dev_queries)
        metrics = {
            "top1_answer_unigram_f1": top1_f1_sum / total,
            "top3_best_answer_unigram_f1": top3_f1_sum / total,
            "category_hit_at_1": category_hit_1 / total,
            "category_hit_at_3": category_hit_3 / total,
        }
        score = (
            0.35 * metrics["top1_answer_unigram_f1"]
            + 0.25 * metrics["top3_best_answer_unigram_f1"]
            + 0.20 * metrics["category_hit_at_1"]
            + 0.20 * metrics["category_hit_at_3"]
        )
        if score > best_score:
            best_params = params
            best_score = score
            best_metrics = metrics

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        best_metrics=best_metrics,
        candidate_count=candidate_count,
    )


def hyperparameters_to_dict(params: TfidfHyperparameters) -> dict:
    payload = asdict(params)
    payload["ngram_range"] = list(params.ngram_range)
    return payload

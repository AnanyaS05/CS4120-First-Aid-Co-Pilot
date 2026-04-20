from __future__ import annotations

# Final evaluation utilities for retrieval, generation, latency, and safety metrics.

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import AppConfig
from .conversation import sanitize_identifier
from .data import build_documents, load_split_dataframe
from .safety import EMERGENCY_CATEGORIES, has_required_emergency_language
from .schemas import QueryRequest, QueryResponse
from .service import FirstAidCopilotService
from .tuning import (
    TfidfSelectionResult,
    hyperparameters_to_dict,
    selection_result_to_dict,
    unigram_f1,
)

FINAL_EVALUATION_DIRNAME = "final"
TFIDF_TEST_ROWS_FILENAME = "tfidf_test_rows.csv"
GENERATED_ANSWER_ROWS_FILENAME = "generated_answer_rows.csv"
FINAL_EVALUATION_SUMMARY_FILENAME = "final_evaluation_summary.json"


@dataclass(slots=True)
class FinalEvaluationResult:
    output_dir: Path
    summary_path: Path
    tfidf_rows_path: Path
    generated_rows_path: Path | None
    summary: dict[str, Any]


def _mean(values: Iterable[float]) -> float:
    """Return the mean of values, or zero for an empty iterable."""
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide while returning zero for a zero denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _as_bool(value: Any) -> bool:
    """Coerce common truthy values into a boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().casefold()
    return normalized in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float:
    """Coerce a value to float, defaulting to zero."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_error(row: dict[str, Any]) -> bool:
    """Return whether an evaluation row contains an error."""
    return bool(str(row.get("error", "")).strip())


def _tokenize(text: str) -> list[str]:
    """Tokenize evaluation text with the simple whitespace tokenizer."""
    return str(text).lower().split()


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Compute ROUGE-L F1 using token-level longest common subsequence."""
    candidate_tokens = _tokenize(candidate)
    reference_tokens = _tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    # Dynamic programming over token sequences gives the longest common subsequence.
    previous = [0] * (len(reference_tokens) + 1)
    for candidate_token in candidate_tokens:
        current = [0]
        for column, reference_token in enumerate(reference_tokens, start=1):
            if candidate_token == reference_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[column - 1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _emergency_expected(category: str) -> bool:
    """Return whether a category should be treated as an emergency."""
    return str(category) in EMERGENCY_CATEGORIES


def _binary_metrics(rows: list[dict[str, Any]], expected_key: str, predicted_key: str) -> dict[str, float | int]:
    """Compute accuracy, precision, recall, F1, and confusion counts."""
    true_positive = sum(1 for row in rows if _as_bool(row[expected_key]) and _as_bool(row[predicted_key]))
    false_positive = sum(1 for row in rows if not _as_bool(row[expected_key]) and _as_bool(row[predicted_key]))
    false_negative = sum(1 for row in rows if _as_bool(row[expected_key]) and not _as_bool(row[predicted_key]))
    true_negative = sum(1 for row in rows if not _as_bool(row[expected_key]) and not _as_bool(row[predicted_key]))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    accuracy = _safe_divide(true_positive + true_negative, len(rows))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def evaluate_tfidf_on_test(
    config: AppConfig,
    selection_result: TfidfSelectionResult,
    *,
    top_k: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate the selected TF-IDF retriever on held-out test rows."""
    train_frame = load_split_dataframe(config, "train")
    test_frame = load_split_dataframe(config, "test")
    train_documents = build_documents(train_frame, "train")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    vectorizer = TfidfVectorizer(**selection_result.best_params.to_vectorizer_kwargs())
    train_texts = [document.page_content for document in train_documents]
    train_matrix = vectorizer.fit_transform(train_texts)
    test_queries = [str(query) for query in test_frame["question"].tolist()]
    test_matrix = vectorizer.transform(test_queries)
    similarities = cosine_similarity(test_matrix, train_matrix)

    rows: list[dict[str, Any]] = []
    for row_index, similarity_row in enumerate(similarities):
        reference_answer = str(test_frame.iloc[row_index]["answer"])
        reference_category = str(test_frame.iloc[row_index]["category"])
        top_indices = similarity_row.argsort()[-top_k:][::-1]
        top_documents = [train_documents[int(index)] for index in top_indices]
        top_scores = [float(similarity_row[int(index)]) for index in top_indices]
        top_answers = [str(document.metadata["answer"]) for document in top_documents]
        top_categories = [str(document.metadata["category"]) for document in top_documents]
        category_rank = next(
            (
                rank
                for rank, category in enumerate(top_categories, start=1)
                if category == reference_category
            ),
            0,
        )
        answer_f1_values = [
            unigram_f1(answer, reference_answer)
            for answer in top_answers
        ]
        top1_score = top_scores[0] if top_scores else 0.0
        top2_score = top_scores[1] if len(top_scores) > 1 else 0.0
        rows.append(
            {
                "row_index": row_index,
                "question": str(test_frame.iloc[row_index]["question"]),
                "reference_answer": reference_answer,
                "category": reference_category,
                "top1_doc_id": str(top_documents[0].metadata["doc_id"]) if top_documents else "",
                "top1_category": top_categories[0] if top_categories else "",
                "top1_score": top1_score,
                "top1_top2_margin": top1_score - top2_score,
                "category_rank": category_rank,
                "category_hit_at_1": category_rank == 1,
                "category_hit_at_3": 1 <= category_rank <= 3,
                "category_hit_at_5": 1 <= category_rank <= 5,
                "mrr_at_5": (1 / category_rank) if 1 <= category_rank <= 5 else 0.0,
                "top1_answer_unigram_f1": answer_f1_values[0] if answer_f1_values else 0.0,
                "top3_best_answer_unigram_f1": max(answer_f1_values[:3]) if answer_f1_values else 0.0,
                "top5_best_answer_unigram_f1": max(answer_f1_values[:5]) if answer_f1_values else 0.0,
                "top_doc_ids": json.dumps(
                    [str(document.metadata["doc_id"]) for document in top_documents]
                ),
                "top_categories": json.dumps(top_categories),
                "top_scores": json.dumps(top_scores),
            }
        )

    summary = {
        "split": "test",
        "row_count": len(rows),
        "top_k": top_k,
        "selected_params": hyperparameters_to_dict(selection_result.best_params),
        "selection": selection_result_to_dict(selection_result),
        "metrics": {
            "category_hit_at_1": _mean(float(row["category_hit_at_1"]) for row in rows),
            "category_hit_at_3": _mean(float(row["category_hit_at_3"]) for row in rows),
            "category_hit_at_5": _mean(float(row["category_hit_at_5"]) for row in rows),
            "mrr_at_5": _mean(float(row["mrr_at_5"]) for row in rows),
            "top1_answer_unigram_f1": _mean(float(row["top1_answer_unigram_f1"]) for row in rows),
            "top3_best_answer_unigram_f1": _mean(float(row["top3_best_answer_unigram_f1"]) for row in rows),
            "top5_best_answer_unigram_f1": _mean(float(row["top5_best_answer_unigram_f1"]) for row in rows),
            "average_top1_score": _mean(float(row["top1_score"]) for row in rows),
            "average_top1_top2_margin": _mean(float(row["top1_top2_margin"]) for row in rows),
        },
    }
    return summary, pd.DataFrame(rows)


def _response_sources_json(response: QueryResponse | None) -> str:
    """Serialize response sources for generated-answer row output."""
    if response is None:
        return "[]"
    return json.dumps(
        [source.model_dump(mode="json") for source in response.sources],
        ensure_ascii=False,
    )


def _summarize_generated_model_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate generated-answer rows for one model."""
    successful_rows = [row for row in rows if not _has_error(row)]
    emergency_rows = [row for row in successful_rows if _as_bool(row["expected_emergency"])]
    emergency_language_rate = _mean(
        float(_as_bool(row["emergency_language_present"]))
        for row in emergency_rows
    )
    return {
        "row_count": len(rows),
        "success_count": len(successful_rows),
        "error_count": len(rows) - len(successful_rows),
        "error_rate": _safe_divide(len(rows) - len(successful_rows), len(rows)),
        "answer_unigram_f1": _mean(_as_float(row["answer_unigram_f1"]) for row in successful_rows),
        "answer_rouge_l_f1": _mean(_as_float(row["answer_rouge_l_f1"]) for row in successful_rows),
        "empty_answer_rate": _mean(float(_as_bool(row["empty_answer"])) for row in successful_rows),
        "used_retrieval_tool_rate": _mean(float(_as_bool(row["used_retrieval_tool"])) for row in successful_rows),
        "warning_rate": _mean(float(_as_float(row["warning_count"]) > 0) for row in successful_rows),
        "source_category_hit_at_1": _mean(float(_as_bool(row["source_category_hit_at_1"])) for row in successful_rows),
        "source_category_hit_at_5": _mean(float(_as_bool(row["source_category_hit_at_5"])) for row in successful_rows),
        "average_source_count": _mean(_as_float(row["source_count"]) for row in successful_rows),
        "average_latency_seconds": _mean(_as_float(row["latency_seconds"]) for row in successful_rows),
        "emergency_detection": _binary_metrics(
            successful_rows,
            "expected_emergency",
            "predicted_emergency",
        ),
        "emergency_language_inclusion_rate": emergency_language_rate,
    }


def _summarize_generated_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate generated-answer rows by model name."""
    return {
        model_name: _summarize_generated_model_rows(model_rows)
        for model_name, model_rows in _group_rows_by_model(rows).items()
    }


def _group_rows_by_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group evaluation rows under their model names."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    return grouped


def evaluate_generated_answers(
    service: FirstAidCopilotService,
    *,
    models: tuple[str, ...],
    profile: str = "demo",
    top_k: int = 5,
    limit: int | None = None,
    skip_unavailable: bool = False,
    rows_output_path: Path | None = None,
    resume: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run local models on the generated-answer evaluation split."""
    frame = load_split_dataframe(service.config, "generated_answer_eval")
    if limit is not None:
        frame = frame.head(limit)

    available_by_model = {
        status.name: status.available
        for status in service.model_statuses()
    }
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, int]] = set()
    if resume and rows_output_path and rows_output_path.exists():
        # Long model evaluations can resume from the checkpoint CSV.
        existing_frame = pd.read_csv(rows_output_path, keep_default_na=False)
        rows = existing_frame.to_dict(orient="records")
        completed = {
            (str(row["model"]), int(row["row_index"]))
            for row in rows
            if "model" in row and "row_index" in row
        }

    def save_checkpoint() -> None:
        """Persist generated-answer rows during long evaluations."""
        if rows_output_path is None:
            return
        rows_output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(rows_output_path, index=False)

    run_id = str(int(time.time()))
    for model_name in models:
        if skip_unavailable and not available_by_model.get(model_name, False):
            if (model_name, -1) in completed:
                continue
            rows.append(
                {
                    "model": model_name,
                    "row_index": -1,
                    "category": "",
                    "question": "",
                    "reference_answer": "",
                    "generated_answer": "",
                    "error": "model unavailable",
                    "latency_seconds": 0.0,
                    "answer_unigram_f1": 0.0,
                    "answer_rouge_l_f1": 0.0,
                    "empty_answer": True,
                    "expected_emergency": False,
                    "predicted_emergency": False,
                    "emergency_language_present": False,
                    "used_retrieval_tool": False,
                    "warning_count": 0,
                    "source_count": 0,
                    "source_category_hit_at_1": False,
                    "source_category_hit_at_5": False,
                    "sources_json": "[]",
                }
            )
            save_checkpoint()
            continue

        for row_index, row in frame.reset_index(drop=True).iterrows():
            if (model_name, row_index) in completed:
                continue
            question = str(row["question"])
            reference_answer = str(row["answer"])
            category = str(row["category"])
            session_id = sanitize_identifier(
                f"eval-{run_id}-{model_name}-{row_index}"
            )
            started_at = time.perf_counter()
            response: QueryResponse | None = None
            error = ""
            try:
                response = service.answer_query(
                    QueryRequest(
                        query=question,
                        model=model_name,
                        profile=profile,
                        top_k=top_k,
                        session_id=session_id,
                    )
                )
                generated_answer = response.answer_text
            except Exception as exc:
                generated_answer = ""
                error = str(exc)
            latency_seconds = time.perf_counter() - started_at
            source_categories = [
                source.category for source in response.sources
            ] if response else []
            expected_emergency = _emergency_expected(category)
            predicted_emergency = bool(response.call_emergency_now) if response else False
            rows.append(
                {
                    "model": model_name,
                    "row_index": row_index,
                    "category": category,
                    "question": question,
                    "reference_answer": reference_answer,
                    "generated_answer": generated_answer,
                    "error": error,
                    "latency_seconds": latency_seconds,
                    "answer_unigram_f1": unigram_f1(generated_answer, reference_answer),
                    "answer_rouge_l_f1": rouge_l_f1(generated_answer, reference_answer),
                    "empty_answer": not bool(generated_answer.strip()),
                    "expected_emergency": expected_emergency,
                    "predicted_emergency": predicted_emergency,
                    "emergency_language_present": has_required_emergency_language(generated_answer),
                    "used_retrieval_tool": bool(response.used_retrieval_tool) if response else False,
                    "warning_count": len(response.warnings) if response else 0,
                    "source_count": len(response.sources) if response else 0,
                    "source_category_hit_at_1": bool(source_categories and source_categories[0] == category),
                    "source_category_hit_at_5": category in source_categories[:5],
                    "sources_json": _response_sources_json(response),
                }
            )
            save_checkpoint()

    summary = {
        "split": "generated_answer_eval",
        "profile": profile,
        "top_k": top_k,
        "row_count_per_model": len(frame),
        "models": list(models),
        "model_availability": {
            model_name: bool(available_by_model.get(model_name, False))
            for model_name in models
        },
        "metrics_by_model": _summarize_generated_rows(rows),
    }
    return summary, pd.DataFrame(rows)


def run_final_evaluation(
    service: FirstAidCopilotService,
    *,
    models: tuple[str, ...] | None = None,
    profile: str = "demo",
    top_k: int = 5,
    limit: int | None = None,
    output_dir: Path | None = None,
    skip_generated: bool = False,
    skip_unavailable: bool = False,
    force_index: bool = True,
) -> FinalEvaluationResult:
    """Run TF-IDF and optional generated-answer final evaluation."""
    output_dir = output_dir or service.config.evaluations_dir / FINAL_EVALUATION_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_result, selection_path = service.evaluate_tfidf()
    tfidf_summary, tfidf_rows = evaluate_tfidf_on_test(
        service.config,
        selection_result,
        top_k=top_k,
    )
    tfidf_rows_path = output_dir / TFIDF_TEST_ROWS_FILENAME
    tfidf_rows.to_csv(tfidf_rows_path, index=False)

    generated_rows_path: Path | None = None
    generated_summary: dict[str, Any] | None = None
    if not skip_generated:
        model_names = models or tuple(service.config.model_names)
        validated_profile = service.config.validate_profile(profile)
        if force_index or not service._index_built(validated_profile):
            service.build_index(validated_profile, force=force_index)
        generated_rows_path = output_dir / GENERATED_ANSWER_ROWS_FILENAME
        generated_summary, generated_rows = evaluate_generated_answers(
            service,
            models=tuple(model_names),
            profile=profile,
            top_k=top_k,
            limit=limit,
            skip_unavailable=skip_unavailable,
            rows_output_path=generated_rows_path,
            resume=True,
        )
        generated_rows.to_csv(generated_rows_path, index=False)

    summary = {
        "tfidf_selection_path": str(selection_path),
        "tfidf_test_rows_path": str(tfidf_rows_path),
        "generated_answer_rows_path": str(generated_rows_path) if generated_rows_path else None,
        "tfidf_test": tfidf_summary,
        "generated_answers": generated_summary,
    }
    summary_path = output_dir / FINAL_EVALUATION_SUMMARY_FILENAME
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return FinalEvaluationResult(
        output_dir=output_dir,
        summary_path=summary_path,
        tfidf_rows_path=tfidf_rows_path,
        generated_rows_path=generated_rows_path,
        summary=summary,
    )

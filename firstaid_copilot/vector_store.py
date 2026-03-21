from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from langchain_core.documents import Document
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_serialized_documents, serialize_documents
from .tuning import TfidfHyperparameters, hyperparameters_to_dict


@dataclass(slots=True)
class TfidfIndexMetadata:
    profile: str
    source_split: str
    hyperparameters: TfidfHyperparameters
    tuning: dict | None = None

    def to_json(self) -> dict:
        return {
            "profile": self.profile,
            "source_split": self.source_split,
            "hyperparameters": hyperparameters_to_dict(self.hyperparameters),
            "tuning": self.tuning or {},
        }

    @classmethod
    def from_json(cls, payload: dict) -> "TfidfIndexMetadata":
        params = payload["hyperparameters"]
        return cls(
            profile=payload["profile"],
            source_split=payload["source_split"],
            hyperparameters=TfidfHyperparameters(
                ngram_range=tuple(params["ngram_range"]),
                min_df=params["min_df"],
                max_df=params["max_df"],
                sublinear_tf=params["sublinear_tf"],
                max_features=params["max_features"],
                stop_words=params.get("stop_words"),
                norm=params.get("norm", "l2"),
            ),
            tuning=payload.get("tuning") or {},
        )


class TfidfVectorStore:
    def __init__(
        self,
        documents: list[Document],
        vectorizer: TfidfVectorizer,
        doc_matrix,
        metadata: TfidfIndexMetadata,
    ) -> None:
        self.documents = documents
        self.vectorizer = vectorizer
        self.doc_matrix = doc_matrix
        self.metadata = metadata

    @classmethod
    def build(
        cls,
        documents: list[Document],
        metadata: TfidfIndexMetadata,
    ) -> "TfidfVectorStore":
        vectorizer = TfidfVectorizer(**metadata.hyperparameters.to_vectorizer_kwargs())
        doc_matrix = vectorizer.fit_transform([document.page_content for document in documents])
        return cls(documents=documents, vectorizer=vectorizer, doc_matrix=doc_matrix, metadata=metadata)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "vectorizer.joblib")
        sparse.save_npz(directory / "doc_matrix.npz", self.doc_matrix)
        serialize_documents(self.documents, directory / "documents.jsonl")
        with (directory / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata.to_json(), handle, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "TfidfVectorStore":
        vectorizer = joblib.load(directory / "vectorizer.joblib")
        doc_matrix = sparse.load_npz(directory / "doc_matrix.npz")
        documents = load_serialized_documents(directory / "documents.jsonl")
        with (directory / "config.json").open("r", encoding="utf-8") as handle:
            metadata = TfidfIndexMetadata.from_json(json.load(handle))
        return cls(documents=documents, vectorizer=vectorizer, doc_matrix=doc_matrix, metadata=metadata)

    def similarity_search_with_scores(
        self,
        query: str,
        *,
        k: int = 3,
    ) -> list[tuple[Document, float]]:
        if not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.doc_matrix).ravel()
        top_indices = scores.argsort()[-k:][::-1]
        return [(self.documents[int(index)], float(scores[int(index)])) for index in top_indices]

    def similarity_search(self, query: str, *, k: int = 3) -> list[Document]:
        return [document for document, _score in self.similarity_search_with_scores(query, k=k)]


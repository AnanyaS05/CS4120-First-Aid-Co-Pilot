from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from langchain_core.documents import Document

from .config import AppConfig, ProfileName

SPLIT_FILES = {
    "train": "train.csv",
    "dev": "dev.csv",
    "test": "test.csv",
    "full_clean": "full_clean.csv",
    "eval_subset": "eval_subset.csv",
    "robustness_test": "robustness_test.csv",
}

REQUIRED_QA_COLUMNS = {
    "question",
    "answer",
    "source",
    "question_norm",
    "category",
}


def load_split_dataframe(config: AppConfig, split_name: str) -> pd.DataFrame:
    try:
        filename = SPLIT_FILES[split_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported split '{split_name}'.") from exc

    path = config.preprocessing_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessing split at {path}.")

    frame = pd.read_csv(path)
    if split_name not in {"robustness_test"}:
        missing = REQUIRED_QA_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"Split '{split_name}' is missing columns: {sorted(missing)}.")
    return frame


def format_document_text(question: str, answer: str, category: str, source: str) -> str:
    return (
        f"Question: {question}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Category: {category}\n"
        f"Source: {source}"
    )


def build_documents(frame: pd.DataFrame, split: str) -> list[Document]:
    documents: list[Document] = []
    for row_index, row in frame.reset_index(drop=True).iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        category = str(row["category"]).strip()
        source = str(row["source"]).strip()
        doc_id = f"{split}-{row_index:05d}"
        documents.append(
            Document(
                page_content=format_document_text(question, answer, category, source),
                metadata={
                    "doc_id": doc_id,
                    "question": question,
                    "answer": answer,
                    "question_norm": str(row.get("question_norm", question)).strip(),
                    "category": category,
                    "source": source,
                    "split": split,
                },
            )
        )
    return documents


def get_profile_source_split(profile: ProfileName) -> str:
    return "train" if profile == "experiment" else "full_clean"


def serialize_documents(documents: Iterable[Document], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    {
                        "page_content": document.page_content,
                        "metadata": document.metadata,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")


def load_serialized_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            documents.append(
                Document(
                    page_content=payload["page_content"],
                    metadata=payload["metadata"],
                )
            )
    return documents


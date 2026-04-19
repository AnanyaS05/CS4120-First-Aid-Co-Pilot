from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from firstaid_copilot.config import AppConfig


def _qa_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["q_char_len"] = frame["question"].str.len()
    frame["a_char_len"] = frame["answer"].str.len()
    frame["q_word_len"] = frame["question"].str.split().str.len()
    frame["a_word_len"] = frame["answer"].str.split().str.len()
    return frame


@pytest.fixture()
def temp_config(tmp_path: Path) -> AppConfig:
    preprocessing_dir = tmp_path / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        {
            "question": "How should I respond to severe bleeding from an arm?",
            "answer": "Apply direct pressure with a clean dressing, elevate if appropriate, and call emergency services if bleeding is severe or does not stop.",
            "source": "FirstAidQA",
            "question_norm": "how should i respond to severe bleeding from an arm?",
            "category": "severe_bleeding",
        },
        {
            "question": "What should I do if someone is choking and cannot speak?",
            "answer": "Call emergency services, encourage coughing if possible, and give back blows and abdominal thrusts if the obstruction persists.",
            "source": "FirstAidQA",
            "question_norm": "what should i do if someone is choking and cannot speak?",
            "category": "choking",
        },
        {
            "question": "What is the best first aid for a minor burn?",
            "answer": "Cool the burn under cool running water for at least 20 minutes and cover it loosely with a sterile dressing.",
            "source": "FirstAidQA",
            "question_norm": "what is the best first aid for a minor burn?",
            "category": "burns",
        },
    ]
    dev_rows = [
        {
            "question": "How do I treat heavy bleeding from a leg wound?",
            "answer": "Apply firm direct pressure with a clean dressing and call emergency services if the bleeding is severe or persistent.",
            "source": "FirstAidQA",
            "question_norm": "how do i treat heavy bleeding from a leg wound?",
            "category": "severe_bleeding",
        },
        {
            "question": "What first aid steps help a person who is choking?",
            "answer": "Call emergency services, encourage coughing if possible, and give back blows if the airway remains blocked.",
            "source": "FirstAidQA",
            "question_norm": "what first aid steps help a person who is choking?",
            "category": "choking",
        },
    ]
    test_rows = dev_rows
    full_clean_rows = train_rows + dev_rows

    train = _qa_frame(train_rows)
    dev = _qa_frame(dev_rows)
    test = _qa_frame(test_rows)
    full_clean = _qa_frame(full_clean_rows)
    eval_subset = dev.copy()
    generated_answer_eval = test.copy()
    robustness = pd.DataFrame(
        [
            {
                "question": "What should I do if someone collapses?",
                "answer": "Check breathing and call emergency services.",
                "source": "FirstAidInstructions",
            }
        ]
    )

    train.to_csv(preprocessing_dir / "train.csv", index=False)
    dev.to_csv(preprocessing_dir / "dev.csv", index=False)
    test.to_csv(preprocessing_dir / "test.csv", index=False)
    full_clean.to_csv(preprocessing_dir / "full_clean.csv", index=False)
    eval_subset.to_csv(preprocessing_dir / "eval_subset.csv", index=False)
    generated_answer_eval.to_csv(preprocessing_dir / "generated_answer_eval.csv", index=False)
    robustness.to_csv(preprocessing_dir / "robustness_test.csv", index=False)

    return AppConfig(root_dir=tmp_path)


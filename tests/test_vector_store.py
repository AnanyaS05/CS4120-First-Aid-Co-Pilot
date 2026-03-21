from __future__ import annotations

from firstaid_copilot.data import build_documents, load_split_dataframe
from firstaid_copilot.tuning import FALLBACK_HYPERPARAMETERS
from firstaid_copilot.vector_store import TfidfIndexMetadata, TfidfVectorStore


def test_vector_store_save_load_and_search(temp_config):
    frame = load_split_dataframe(temp_config, "train")
    documents = build_documents(frame, "train")
    metadata = TfidfIndexMetadata(
        profile="experiment",
        source_split="train",
        hyperparameters=FALLBACK_HYPERPARAMETERS,
        tuning={"best_score": 1.0},
    )
    store = TfidfVectorStore.build(documents, metadata)
    store.save(temp_config.index_dir("experiment"))

    loaded = TfidfVectorStore.load(temp_config.index_dir("experiment"))
    hits = loaded.similarity_search_with_scores("How do I help with choking?", k=1)

    assert hits
    assert hits[0][0].metadata["category"] == "choking"
    assert (temp_config.index_dir("experiment") / "config.json").exists()


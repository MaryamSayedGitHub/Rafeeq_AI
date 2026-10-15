"""Integration sanity test: REAL BGE-M3 embeddings -> FAISS store.

Loads the real model (slow on first run). Run it separately:
    uv run pytest tests/knowledge/integration/test_faiss_bge_m3.py -v -s
and skip it in quick runs:
    uv run pytest tests --ignore=tests/knowledge/integration

Only the ranking is checked. No score thresholds.
"""

import sys

import pytest

from rafeeq.knowledge.embeddings.embedder import Embedder
from rafeeq.knowledge.ingestion.chunker import chunk_pages
from rafeeq.knowledge.vector_store import FAISSVectorStore

TEXTS = [
    "Self-attention allows each token to attend to other tokens.",
    "Convolutional neural networks are useful for image processing.",
    "Gradient descent is used to optimize neural networks.",
]
AR_QUERY = "يعني إيه Self-Attention؟"


def show(line: str = "") -> None:
    enc = sys.stdout.encoding or "utf-8"
    print(line.encode(enc, errors="replace").decode(enc))


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    e = Embedder()
    e.load()
    return e


@pytest.fixture(scope="module")
def store(embedder) -> FAISSVectorStore:
    pages = [{"page": i + 1, "text": text} for i, text in enumerate(TEXTS)]
    chunks = chunk_pages(pages, room_id="llm_101", file_name="sample.pdf")
    s = FAISSVectorStore(embedder.dimension)
    s.add(embedder.embed_documents([c["text"] for c in chunks]), chunks)
    return s


def test_arabic_query_retrieves_self_attention_chunk_first(embedder, store):
    results = store.search(embedder.embed_query(AR_QUERY), top_k=3, room_id="llm_101")

    show(f"\nQuery: {AR_QUERY}")
    for rank, r in enumerate(results, start=1):
        show(f"  {rank}. score={r.score:.4f}  page={r.metadata['page']}  {r.text}")

    assert len(results) == 3
    assert results[0].text == TEXTS[0]
    assert results[0].metadata["page"] == 1
    assert results[0].metadata["file_name"] == "sample.pdf"


def test_other_room_returns_nothing(embedder, store):
    assert store.search(embedder.embed_query(AR_QUERY), top_k=3, room_id="another_room") == []


def test_ranking_survives_save_and_load_without_reembedding(embedder, store, tmp_path):
    store.save(tmp_path)
    loaded = FAISSVectorStore.load(tmp_path)  # no embedder involved
    results = loaded.search(embedder.embed_query(AR_QUERY), top_k=1, room_id="llm_101")
    assert results[0].text == TEXTS[0]
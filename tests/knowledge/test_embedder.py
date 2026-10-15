"""Tests for the BGE-M3 embedder.

Tests that validate inputs never load the model. Tests that need real
embeddings share ONE module-scoped model (loaded on first use only), because
loading BGE-M3 takes a while and the first run downloads ~2 GB.
"""

import sys

import numpy as np
import pytest

from rafeeq.knowledge.embeddings.embedder import DEFAULT_MODEL_NAME, Embedder

EN_SENTENCE = "Self-attention allows each token to attend to other tokens."

DOCS = [
    "Self-attention allows each token to attend to other tokens.",
    "Convolutional neural networks are useful for image processing.",
    "Gradient descent is used to optimize neural networks.",
]
AR_QUERY = "يعني إيه Self-Attention؟"
EN_QUERY = "What is self-attention?"


def show(line: str = "") -> None:
    """print() that can't crash on consoles that can't encode Arabic."""
    enc = sys.stdout.encoding or "utf-8"
    print(line.encode(enc, errors="replace").decode(enc))


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """One shared, loaded model. Only created by tests that request it."""
    e = Embedder()
    e.load()
    return e


# ---------- no model needed ----------

def test_default_model_is_bge_m3():
    assert DEFAULT_MODEL_NAME == "BAAI/bge-m3"
    assert Embedder().model_name == "BAAI/bge-m3"


def test_constructing_does_not_load_the_model():
    e = Embedder()
    assert e.is_loaded is False
    assert e._model is None


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError, match="batch_size"):
        Embedder(batch_size=0)


def test_empty_list_raises_clear_error_without_loading_model():
    e = Embedder()
    with pytest.raises(ValueError, match="at least one text"):
        e.embed_documents([])
    assert e.is_loaded is False


def test_single_string_instead_of_list_is_rejected():
    e = Embedder()
    with pytest.raises(TypeError, match="not a single string"):
        e.embed_documents("just one string")
    assert e.is_loaded is False


def test_non_string_item_rejected():
    e = Embedder()
    with pytest.raises(TypeError, match=r"texts\[1\] must be a string"):
        e.embed_documents(["ok", 123])
    assert e.is_loaded is False


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_document_rejected(blank):
    e = Embedder()
    with pytest.raises(ValueError, match=r"texts\[1\] is empty"):
        e.embed_documents(["ok", blank])
    assert e.is_loaded is False


@pytest.mark.parametrize("blank", ["", "  ", "\n"])
def test_blank_query_rejected(blank):
    e = Embedder()
    with pytest.raises(ValueError, match="query must not be empty"):
        e.embed_query(blank)
    assert e.is_loaded is False


def test_non_string_query_rejected():
    e = Embedder()
    with pytest.raises(TypeError, match="query must be a string"):
        e.embed_query(["not", "a", "string"])
    assert e.is_loaded is False


# ---------- real model ----------

def test_model_can_be_loaded(embedder):
    assert embedder.is_loaded is True


def test_one_english_sentence_produces_an_embedding(embedder):
    vectors = embedder.embed_documents([EN_SENTENCE])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (1, embedder.dimension)
    assert np.isfinite(vectors).all()


def test_embedding_dimension_is_consistent(embedder):
    assert embedder.dimension == 1024  # BAAI/bge-m3
    short = embedder.embed_documents(["Hi."])
    long = embedder.embed_documents([" ".join(["attention"] * 300)])
    arabic = embedder.embed_documents([AR_QUERY])
    query = embedder.embed_query(EN_SENTENCE)
    assert short.shape[1] == long.shape[1] == arabic.shape[1] == query.shape[0] == embedder.dimension


def test_multiple_texts_return_expected_2d_shape(embedder):
    vectors = embedder.embed_documents(DOCS)
    assert vectors.ndim == 2
    assert vectors.shape == (len(DOCS), embedder.dimension)


def test_batching_does_not_change_the_number_of_rows(embedder):
    texts = [f"Sentence number {i} about transformers." for i in range(10)]  # > default batch size
    assert embedder.embed_documents(texts).shape == (10, embedder.dimension)


def test_query_returns_expected_1d_shape(embedder):
    vector = embedder.embed_query(AR_QUERY)
    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1
    assert vector.shape == (embedder.dimension,)


def test_embeddings_are_normalized(embedder):
    docs = embedder.embed_documents(DOCS)
    query = embedder.embed_query(AR_QUERY)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-4)
    assert np.isclose(np.linalg.norm(query), 1.0, atol=1e-4)


def test_embedding_the_same_text_twice_is_consistent(embedder):
    first = embedder.embed_query(EN_SENTENCE)
    second = embedder.embed_query(EN_SENTENCE)
    assert np.allclose(first, second, atol=1e-5)


def test_query_and_document_embedding_of_same_text_match(embedder):
    # BGE-M3 uses no query prefix, so both paths must give the same vector.
    as_doc = embedder.embed_documents([EN_SENTENCE])[0]
    as_query = embedder.embed_query(EN_SENTENCE)
    assert np.allclose(as_doc, as_query, atol=1e-4)


# ---------- cross-lingual sanity check (ranking only, no score thresholds) ----------

def _rank(embedder: Embedder, query: str):
    doc_vecs = embedder.embed_documents(DOCS)
    query_vec = embedder.embed_query(query)
    scores = doc_vecs @ query_vec  # normalized vectors -> dot product == cosine similarity
    order = np.argsort(-scores)
    return scores, order


def _print_ranking(title: str, query: str, scores, order) -> None:
    show(f"\n=== {title} ===")
    show(f"Query: {query}")
    for rank, idx in enumerate(order, start=1):
        show(f"  {rank}. score={scores[idx]:.4f}  {DOCS[idx]}")


def test_arabic_query_ranks_self_attention_document_first(embedder):
    scores, order = _rank(embedder, AR_QUERY)
    _print_ranking("Arabic query -> English documents", AR_QUERY, scores, order)
    assert order[0] == 0, "Self-attention document should rank first for the Arabic query"


def test_english_query_ranks_self_attention_document_first(embedder):
    # Control: if this passes but the Arabic test fails, the problem is cross-lingual transfer.
    scores, order = _rank(embedder, EN_QUERY)
    _print_ranking("English query -> English documents (control)", EN_QUERY, scores, order)
    assert order[0] == 0
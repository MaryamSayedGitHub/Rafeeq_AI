"""Retriever tests: real FAISSVectorStore + a deterministic fake embedder (no BGE-M3)."""

import pytest

from rafeeq.knowledge.retrieval import Retriever, retrieve, set_default_retriever

SELF_ATTENTION = "Self-attention allows each token to attend to other tokens."
CNN = "Convolutional neural networks are useful for image processing."
GRADIENT = "Gradient descent is used to optimize neural networks."

CONTRACT_KEYS = {"chunk_id", "text", "file_name", "page", "topic_id", "score"}
EN_QUERY = "What is self-attention?"
AR_QUERY = "يعني إيه Self-Attention؟"
AR_QUERY_2 = "ما هو الانتباه الذاتي؟"

ROOMS = {
    "room_A": {"hands_on_llm.pdf": [SELF_ATTENTION, CNN, GRADIENT]},
    # Room B holds the SAME self-attention text on a different page of a different file.
    "room_B": {"other_course.pdf": [CNN, GRADIENT, SELF_ATTENTION]},
}


@pytest.fixture()
def retriever(build_store, hashing_embedder):
    return Retriever(build_store(ROOMS), hashing_embedder)


# ---------- contract ----------

def test_result_has_exactly_the_contract_fields_and_types(retriever):
    results = retriever.retrieve("room_A", EN_QUERY, k=3)
    assert results
    for item in results:
        assert set(item) == CONTRACT_KEYS
        assert isinstance(item["chunk_id"], str) and isinstance(item["text"], str)
        assert isinstance(item["file_name"], str) and isinstance(item["page"], int)
        assert isinstance(item["topic_id"], str) and isinstance(item["score"], float)


def test_top_result_is_the_right_chunk_with_page_file_and_topic(retriever):
    top = retriever.retrieve("room_A", EN_QUERY)[0]
    assert top["text"] == SELF_ATTENTION
    assert top["file_name"] == "hands_on_llm.pdf"
    assert top["page"] == 1
    assert top["topic_id"] == "topic_attention"
    assert top["score"] > 0


def test_results_are_sorted_best_first(retriever):
    scores = [r["score"] for r in retriever.retrieve("room_A", EN_QUERY, k=3)]
    assert scores == sorted(scores, reverse=True)


def test_default_k_is_six_and_k_limits_results(retriever):
    assert len(retriever.retrieve("room_A", EN_QUERY)) == 3  # only 3 chunks exist in room_A
    assert len(retriever.retrieve("room_A", EN_QUERY, k=2)) == 2
    assert len(retriever.retrieve("room_A", EN_QUERY, k=1)) == 1


# ---------- Arabic + English ----------

@pytest.mark.parametrize("query", [EN_QUERY, AR_QUERY, AR_QUERY_2])
def test_arabic_and_english_queries_find_the_english_self_attention_chunk(retriever, query):
    top = retriever.retrieve("room_A", query)[0]
    assert top["text"] == SELF_ATTENTION and top["page"] == 1


def test_arabic_and_english_give_the_same_ranking(retriever):
    en = [r["chunk_id"] for r in retriever.retrieve("room_A", EN_QUERY, k=3)]
    ar = [r["chunk_id"] for r in retriever.retrieve("room_A", AR_QUERY, k=3)]
    assert en == ar


def test_arabic_query_for_a_different_topic_finds_a_different_chunk(retriever):
    top = retriever.retrieve("room_A", "ما هي الشبكات العصبية الالتفافية؟")[0]
    assert top["text"] == CNN and top["page"] == 2


# ---------- room isolation ----------

def test_each_room_only_returns_its_own_chunks(retriever):
    a = retriever.retrieve("room_A", EN_QUERY, k=10)
    b = retriever.retrieve("room_B", EN_QUERY, k=10)
    assert {r["file_name"] for r in a} == {"hands_on_llm.pdf"}
    assert {r["file_name"] for r in b} == {"other_course.pdf"}
    assert not {r["chunk_id"] for r in a} & {r["chunk_id"] for r in b}


def test_identical_text_in_two_rooms_is_returned_from_the_requested_room_only(retriever):
    assert retriever.retrieve("room_A", EN_QUERY)[0]["page"] == 1
    assert retriever.retrieve("room_B", EN_QUERY)[0]["page"] == 3
    assert retriever.retrieve("room_B", EN_QUERY)[0]["file_name"] == "other_course.pdf"


def test_unknown_room_returns_empty_list_not_another_rooms_chunks(retriever):
    assert retriever.retrieve("room_C", EN_QUERY) == []


def test_room_with_fewer_chunks_than_k_returns_only_its_own(build_store, hashing_embedder):
    store = build_store({"small": {"s.pdf": [SELF_ATTENTION]}, "big": {"b.pdf": [CNN, GRADIENT, SELF_ATTENTION]}})
    results = Retriever(store, hashing_embedder).retrieve("small", EN_QUERY, k=6)
    assert len(results) == 1 and results[0]["file_name"] == "s.pdf"


# ---------- no fabrication ----------

def test_every_result_exists_in_the_store_unchanged(build_store, hashing_embedder):
    store = build_store(ROOMS)
    indexed = {(r["chunk_id"], r["text"], r["page"], r["file_name"]) for r in store._records}
    for item in Retriever(store, hashing_embedder).retrieve("room_A", AR_QUERY, k=10):
        assert (item["chunk_id"], item["text"], item["page"], item["file_name"]) in indexed


def test_empty_store_returns_empty_list(empty_store, hashing_embedder):
    assert Retriever(empty_store, hashing_embedder).retrieve("room_A", EN_QUERY) == []


def test_query_is_embedded_with_the_query_path_exactly_once(retriever, hashing_embedder):
    retriever.retrieve("room_A", AR_QUERY)
    assert hashing_embedder.query_calls == [AR_QUERY]


# ---------- guard against a misbehaving store ----------

class LeakyStore:
    """A broken store that ignores the room filter."""

    class Hit:
        text = "leaked"
        score = 0.9
        metadata = {"room_id": "room_B", "chunk_id": "x", "file_name": "f.pdf", "page": 1}

    def search(self, query_embedding, top_k=5, room_id=None, **filters):
        return [self.Hit()]


def test_retriever_fails_loudly_if_the_store_ever_leaks_another_room(hashing_embedder):
    with pytest.raises(RuntimeError, match="room isolation"):
        Retriever(LeakyStore(), hashing_embedder).retrieve("room_A", EN_QUERY)


# ---------- validation ----------

@pytest.mark.parametrize("bad", ["", "   ", None])
def test_invalid_query_rejected(retriever, bad):
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("room_A", bad)


@pytest.mark.parametrize("bad", ["", None])
def test_invalid_room_rejected(retriever, bad):
    with pytest.raises(ValueError, match="room_id"):
        retriever.retrieve(bad, EN_QUERY)


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3", True])
def test_invalid_k_rejected(retriever, bad):
    with pytest.raises(ValueError, match="k must"):
        retriever.retrieve("room_A", EN_QUERY, k=bad)


# ---------- module-level contract ----------

def test_module_level_retrieve_needs_configuration_then_matches_the_class(retriever):
    set_default_retriever(None)
    try:
        with pytest.raises(RuntimeError, match="no retriever configured"):
            retrieve("room_A", EN_QUERY)
        set_default_retriever(retriever)
        assert retrieve("room_A", EN_QUERY, 2) == retriever.retrieve("room_A", EN_QUERY, 2)
        assert retrieve("room_A", EN_QUERY) == retriever.retrieve("room_A", EN_QUERY, 6)
    finally:
        set_default_retriever(None)


# ---------- end to end with the answerer (fake LLM) ----------

class EchoLLM:
    def generate(self, system, prompt):
        return "Self-attention lets tokens attend to each other [1]."


def test_answer_cites_the_page_of_the_retrieved_chunk(retriever):
    from rafeeq.knowledge.answering import Answerer

    result = Answerer(retriever, EchoLLM(), min_score=0.1).answer("room_B", AR_QUERY)
    assert result["grounded"] is True
    assert result["citations"] == [{"file_name": "other_course.pdf", "page": 3}]


def test_answer_for_an_unknown_room_is_not_grounded(retriever):
    from rafeeq.knowledge.answering import Answerer

    result = Answerer(retriever, EchoLLM(), min_score=0.1).answer("room_C", EN_QUERY)
    assert result["grounded"] is False and result["citations"] == []
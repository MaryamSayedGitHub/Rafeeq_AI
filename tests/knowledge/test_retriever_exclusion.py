"""Retriever(exclude_non_content=True): index-like hits are skipped, everything else is untouched."""

from types import SimpleNamespace

import pytest

from rafeeq.knowledge.retrieval import Retriever
from test_content_filter import PROSE, index_page

QUERY = "What is self-attention?"
ATTENTION = "Self-attention allows each token to attend to other tokens in the sequence."
CNN = "Convolutional neural networks are useful for image processing."
GRADIENT = "Gradient descent is used to optimize neural networks."
# An index page that repeats the query's words, so it ranks first with the hashing embedder.
SELF_ATTENTION_INDEX = index_page([("self-attention", "Self Attention Layers, Self Attention Heads")] * 12)

ROOMS = {
    "room_A": {"book.pdf": [CNN, SELF_ATTENTION_INDEX, ATTENTION, GRADIENT]},
    "room_B": {"other.pdf": [SELF_ATTENTION_INDEX, ATTENTION]},
}


@pytest.fixture()
def stores(build_store, hashing_embedder):
    store = build_store(ROOMS)
    return Retriever(store, hashing_embedder), Retriever(store, hashing_embedder, exclude_non_content=True)


def pages(results):
    return [r["page"] for r in results]


def test_default_behaviour_is_unchanged_and_the_index_page_ranks_first(stores):
    baseline, _ = stores
    assert baseline.exclude_non_content is False
    results = baseline.retrieve("room_A", QUERY, k=4)
    assert pages(results)[0] == 2 and results[0]["text"] == SELF_ATTENTION_INDEX


def test_excluding_removes_only_the_index_page_and_keeps_order_and_scores(stores):
    baseline, excluding = stores
    before = baseline.retrieve("room_A", QUERY, k=4)
    after = excluding.retrieve("room_A", QUERY, k=4)
    assert pages(after) == [p for p in pages(before) if p != 2]
    by_page = {r["page"]: r for r in before}
    for item in after:
        assert item == by_page[item["page"]]          # same chunk, same score, same metadata


def test_freed_slots_are_filled_with_the_next_best_chunks(stores):
    baseline, excluding = stores
    top2_before = pages(baseline.retrieve("room_A", QUERY, k=2))
    top2_after = pages(excluding.retrieve("room_A", QUERY, k=2))
    assert 2 in top2_before and 2 not in top2_after
    assert len(top2_after) == 2
    assert top2_after[0] == 3                           # the self-attention page, previously second


def test_fewer_than_k_are_returned_when_the_room_has_nothing_more(stores):
    _, excluding = stores
    assert len(excluding.retrieve("room_A", QUERY, k=6)) == 3          # 4 pages minus the index page


def test_result_contract_is_unchanged(stores):
    _, excluding = stores
    for item in excluding.retrieve("room_A", QUERY, k=3):
        assert set(item) == {"chunk_id", "text", "file_name", "page", "topic_id", "score"}


def test_room_isolation_still_holds(stores):
    _, excluding = stores
    assert {r["file_name"] for r in excluding.retrieve("room_A", QUERY, k=6)} == {"book.pdf"}
    assert {r["file_name"] for r in excluding.retrieve("room_B", QUERY, k=6)} == {"other.pdf"}
    assert excluding.retrieve("no_such_room", QUERY) == []


def test_argument_validation_is_unchanged(stores):
    _, excluding = stores
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            excluding.retrieve("room_A", QUERY, k=bad)
    with pytest.raises(ValueError):
        excluding.retrieve("room_A", "   ")
    with pytest.raises(ValueError):
        excluding.retrieve("", QUERY)


# ------------------------------------------------ candidate pool growth, with a scripted store

class ScriptedStore:
    """Returns the first top_k of a fixed ranked list and records every request."""

    def __init__(self, ranked):
        self.ranked = ranked
        self.requests = []

    def search(self, query_embedding, top_k=5, room_id=None, **filters):
        self.requests.append(top_k)
        return self.ranked[:top_k]


class StubEmbedder:
    def embed_query(self, text):
        return [1.0]


def hit(n, text, room="r"):
    meta = {"room_id": room, "chunk_id": f"c{n}", "file_name": "b.pdf", "page": n, "topic_id": None}
    return SimpleNamespace(text=text, score=1.0 - n / 1000, metadata=meta)


def test_pool_grows_until_enough_content_chunks_are_found():
    ranked = [hit(n, index_page()) for n in range(1, 41)] + [hit(n, PROSE) for n in range(41, 51)]
    store = ScriptedStore(ranked)
    results = Retriever(store, StubEmbedder(), exclude_non_content=True).retrieve("r", "q", k=3)
    assert [r["page"] for r in results] == [41, 42, 43]            # the first three non-index chunks, in order
    assert store.requests == sorted(store.requests) and len(set(store.requests)) > 1   # it asked for more than once
    assert store.requests[-1] >= 43


def test_everything_index_like_gives_an_empty_result_instead_of_looping():
    store = ScriptedStore([hit(n, index_page()) for n in range(1, 31)])
    results = Retriever(store, StubEmbedder(), exclude_non_content=True).retrieve("r", "q", k=3)
    assert results == []
    assert len(store.requests) <= 6


def test_a_chunk_from_another_room_is_still_a_hard_error():
    store = ScriptedStore([hit(1, PROSE, room="someone_else")])
    with pytest.raises(RuntimeError, match="room isolation"):
        Retriever(store, StubEmbedder(), exclude_non_content=True).retrieve("r", "q", k=1)


def test_without_the_flag_the_store_is_asked_for_exactly_k():
    store = ScriptedStore([hit(n, PROSE) for n in range(1, 20)])
    Retriever(store, StubEmbedder()).retrieve("r", "q", k=4)
    assert store.requests == [4]


def test_one_search_is_enough_when_the_first_pool_already_holds_k_content_chunks():
    store = ScriptedStore([hit(n, PROSE) for n in range(1, 60)])
    results = Retriever(store, StubEmbedder(), exclude_non_content=True).retrieve("r", "q", k=3)
    assert [r["page"] for r in results] == [1, 2, 3]
    assert store.requests == [13]                      # first pool = max(3 * 3, 3 + 10); no wasteful second scan
    store = ScriptedStore([hit(n, PROSE) for n in range(1, 60)])
    Retriever(store, StubEmbedder(), exclude_non_content=True).retrieve("r", "q", k=6)
    assert store.requests == [18]                      # max(6 * 3, 6 + 10)
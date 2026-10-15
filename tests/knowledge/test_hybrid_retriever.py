"""HybridRetriever: RRF over dense + BM25 candidates. Every number below can be checked by hand.

The store holds hand-built 2-D unit vectors, so the cosine of each chunk with the query (1, 0) is known exactly:
    A (0 deg)   cos 1.0       "alpha beta"
    B (30 deg)  cos 0.866...  "gamma"
    C (60 deg)  cos 0.5       "delta"
    D (90 deg)  cos 0.0       "epsilon"
Dense order is therefore A, B, C, D.
"""

import math

import numpy as np
import pytest

from rafeeq.knowledge.retrieval import HybridRetriever, Retriever, build_retriever, reciprocal_rank_fusion
from rafeeq.knowledge.vector_store import FAISSVectorStore
from test_content_filter import index_page

K = 60


def unit(degrees):
    r = math.radians(degrees)
    return np.array([math.cos(r), math.sin(r)], dtype=np.float32)


class FixedEmbedder:
    """Every query embeds to (1, 0); only the query WORDS differ between tests (they drive BM25)."""
    dimension = 2

    def embed_query(self, text):
        return unit(0)


def chunk(chunk_id, text, room="room_A", page=1, file_name="book.pdf"):
    return {"chunk_id": chunk_id, "room_id": room, "file_name": file_name, "page": page, "block_type": "text", "text": text, "topic_id": "t"}


def make_store(extra=()):
    store = FAISSVectorStore(2)
    rows = [(0, chunk("A", "alpha beta", page=1)), (30, chunk("B", "gamma", page=2)), (60, chunk("C", "delta", page=3)),
            (90, chunk("D", "epsilon", page=4))] + list(extra)
    store.add(np.stack([unit(d) for d, _ in rows]), [c for _, c in rows])
    return store


def ids(results):
    return [r["chunk_id"] for r in results]


# ------------------------------------------------------------------ fusion arithmetic

def test_rrf_scores_are_the_sum_of_one_over_k_plus_rank_with_equal_weights():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert reciprocal_rank_fusion([["x", "x", "y"]], k=60) == {"x": 1 / 61, "y": 1 / 63}   # a repeated item counts once, at its first position
    assert reciprocal_rank_fusion([[], []]) == {}
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


# ------------------------------------------------------------------ ranking, score and fusion score

def test_a_lexical_match_can_outrank_a_better_cosine_and_the_fusion_value_is_kept_apart_from_the_score():
    results = HybridRetriever(make_store(), FixedEmbedder()).retrieve("room_A", "delta", k=4)
    assert ids(results) == ["C", "A", "B", "D"]                                  # fusion order, not cosine order
    by_id = {r["chunk_id"]: r for r in results}
    assert by_id["C"]["fusion_score"] == pytest.approx(1 / (K + 3) + 1 / (K + 1))  # dense rank 3 + lexical rank 1
    assert by_id["A"]["fusion_score"] == pytest.approx(1 / (K + 1))
    assert by_id["B"]["fusion_score"] == pytest.approx(1 / (K + 2))
    assert by_id["D"]["fusion_score"] == pytest.approx(1 / (K + 4))
    assert [by_id[c]["score"] for c in "ABCD"] == pytest.approx([1.0, math.cos(math.radians(30)), 0.5, 0.0], abs=1e-6)  # true cosine
    assert by_id["C"]["score"] != by_id["C"]["fusion_score"]
    assert (by_id["C"]["dense_rank"], by_id["C"]["lexical_rank"]) == (3, 1)
    assert (by_id["A"]["dense_rank"], by_id["A"]["lexical_rank"], by_id["A"]["lexical_score"]) == (1, None, None)
    assert by_id["C"]["lexical_score"] > 0


def test_scores_are_exactly_what_dense_search_reports_for_the_same_chunks():
    store = make_store()
    dense = {h.metadata["chunk_id"]: h.score for h in store.search(unit(0), top_k=4, room_id="room_A")}
    for r in HybridRetriever(store, FixedEmbedder()).retrieve("room_A", "delta gamma", k=4):
        assert r["score"] == dense[r["chunk_id"]]


def test_without_any_lexical_overlap_the_result_equals_the_dense_retriever():
    store = make_store()
    hybrid = HybridRetriever(store, FixedEmbedder()).retrieve("room_A", "zzzz qqqq", k=3)
    dense = Retriever(store, FixedEmbedder()).retrieve("room_A", "zzzz qqqq", k=3)
    assert ids(hybrid) == ids(dense) == ["A", "B", "C"]
    assert [r["score"] for r in hybrid] == [r["score"] for r in dense]
    assert all(r["lexical_rank"] is None and r["lexical_score"] is None for r in hybrid)
    assert [r["fusion_score"] for r in hybrid] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_a_candidate_found_only_by_bm25_gets_its_exact_cosine_from_the_store():
    # candidate_pool=2 and k=2 -> the dense list is [A, B]; "delta" (C) is found only lexically.
    results = HybridRetriever(make_store(), FixedEmbedder(), candidate_pool=2).retrieve("room_A", "delta", k=2)
    assert ids(results) == ["A", "C"]            # A (dense 1) and C (lexical 1) tie at 1/61 -> higher cosine first; B has only 1/62
    c = results[1]
    assert c["dense_rank"] is None and c["lexical_rank"] == 1
    assert c["score"] == pytest.approx(0.5, abs=1e-6)             # not None, not the fusion value
    assert c["fusion_score"] == pytest.approx(1 / 61)


class TieStableStore(FAISSVectorStore):
    """Equal cosines are returned in page order, so the dense list does not depend on how a FAISS build orders ties."""

    def search(self, query_embedding, top_k=5, **filters):
        hits = super().search(query_embedding, top_k=max(len(self), top_k), **filters)
        hits.sort(key=lambda h: (-round(h.score, 6), h.metadata["page"]))
        return hits[:top_k]


def test_equal_fusion_and_equal_cosine_are_ordered_by_chunk_id_not_by_insertion_order():
    store = TieStableStore(2)
    # same vector, so cosine ties; "m" is inserted first so the dense list is [m, b], while BM25 prefers the shorter "b": [b, m]
    store.add(np.stack([unit(0), unit(0)]), [chunk("m", "foo bar filler words here"), chunk("b", "bar", page=2)])
    results = HybridRetriever(store, FixedEmbedder()).retrieve("room_A", "bar", k=2)
    assert [(r["dense_rank"], r["lexical_rank"]) for r in results] == [(2, 1), (1, 2)]       # b: dense 2, lexical 1 ; m: dense 1, lexical 2
    assert results[0]["fusion_score"] == results[1]["fusion_score"] == 1 / 61 + 1 / 62       # a genuine tie
    assert ids(results) == ["b", "m"]                                                       # chunk_id order, not the dense (insertion) order [m, b]


def test_k_larger_than_the_pool_and_than_the_room():
    r = HybridRetriever(make_store(), FixedEmbedder(), candidate_pool=1).retrieve("room_A", "delta", k=4)
    assert ids(r) == ["C", "A", "B", "D"]                          # the pool is raised to k, never lowered: all four are candidates
    assert len(HybridRetriever(make_store(), FixedEmbedder()).retrieve("room_A", "alpha", k=50)) == 4


# ------------------------------------------------------------------ room isolation

def test_another_room_never_contributes_to_either_candidate_list():
    foreign = (0, chunk("X", "delta delta delta alpha", room="room_B", page=9))     # perfect cosine AND a perfect lexical match
    store = make_store(extra=[foreign])
    results = HybridRetriever(store, FixedEmbedder()).retrieve("room_A", "delta alpha", k=10)
    assert "X" not in ids(results) and len(results) == 4
    other = HybridRetriever(store, FixedEmbedder()).retrieve("room_B", "delta alpha", k=10)
    assert ids(other) == ["X"]
    assert HybridRetriever(store, FixedEmbedder()).retrieve("no_such_room", "delta", k=3) == []


def test_a_dense_hit_from_the_wrong_room_fails_loudly():
    class LeakyStore:
        def __init__(self, inner):
            self._inner = inner

        def __len__(self):
            return len(self._inner)

        def room_chunks(self, room_id):
            return self._inner.room_chunks(room_id)

        def similarities(self, *a):
            return self._inner.similarities(*a)

        def search(self, query, top_k=5, room_id=None, **kw):
            return self._inner.search(query, top_k=top_k, room_id="room_B")

    store = make_store(extra=[(0, chunk("X", "foreign", room="room_B"))])
    with pytest.raises(RuntimeError, match="room isolation violated"):
        HybridRetriever(LeakyStore(store), FixedEmbedder()).retrieve("room_A", "alpha", k=2)


def test_a_lexical_candidate_from_the_wrong_room_fails_loudly():
    class LeakyLexicon:
        def __init__(self, inner):
            self._inner = inner

        def __len__(self):
            return len(self._inner)

        def room_chunks(self, room_id):
            return self._inner.room_chunks("room_B")                     # the lexical source ignores the requested room

        def similarities(self, *a):
            return self._inner.similarities(*a)

        def search(self, *a, **kw):
            return self._inner.search(*a, **kw)

    store = make_store(extra=[(0, chunk("X", "delta foreign", room="room_B"))])
    with pytest.raises(RuntimeError, match="room isolation violated"):
        HybridRetriever(LeakyLexicon(store), FixedEmbedder()).retrieve("room_A", "delta", k=2)


# ------------------------------------------------------------------ index-page exclusion

def test_exclude_non_content_removes_index_pages_from_both_lists_before_ranks_are_assigned():
    junk = (0, chunk("IDX", index_page([("delta", "Delta Section, Delta Chapter")] * 12), page=99))   # cosine 1.0 and best BM25 for "delta"
    store = make_store(extra=[junk])
    plain = HybridRetriever(store, FixedEmbedder()).retrieve("room_A", "delta", k=5)
    assert "IDX" in ids(plain)
    clean = HybridRetriever(store, FixedEmbedder(), exclude_non_content=True).retrieve("room_A", "delta", k=5)
    assert "IDX" not in ids(clean)
    assert ids(clean) == ["C", "A", "B", "D"]                                       # identical to the store without the index page
    by_id = {r["chunk_id"]: r for r in clean}
    assert (by_id["A"]["dense_rank"], by_id["C"]["lexical_rank"]) == (1, 1)         # ranks counted without the removed page
    assert by_id["C"]["fusion_score"] == pytest.approx(1 / 63 + 1 / 61)


# ------------------------------------------------------------------ contract, validation, freshness

def test_result_contract_extends_the_dense_one_without_changing_it():
    store = make_store()
    dense = Retriever(store, FixedEmbedder()).retrieve("room_A", "alpha", k=1)[0]
    hybrid = HybridRetriever(store, FixedEmbedder()).retrieve("room_A", "alpha", k=1)[0]
    assert set(dense) == {"chunk_id", "text", "file_name", "page", "topic_id", "score"}
    assert set(hybrid) == set(dense) | {"fusion_score", "dense_rank", "lexical_rank", "lexical_score"}
    assert {key: hybrid[key] for key in dense} == dense


def test_invalid_arguments_are_rejected_like_the_dense_retriever_does():
    retriever = HybridRetriever(make_store(), FixedEmbedder())
    for room, query, k in [("", "q", 1), (None, "q", 1), ("room_A", " ", 1), ("room_A", None, 1), ("room_A", "q", 0), ("room_A", "q", True), ("room_A", "q", "3")]:
        with pytest.raises(ValueError):
            retriever.retrieve(room, query, k)
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            HybridRetriever(make_store(), FixedEmbedder(), rrf_k=bad)
        with pytest.raises(ValueError):
            HybridRetriever(make_store(), FixedEmbedder(), candidate_pool=bad)
    with pytest.raises(TypeError):
        HybridRetriever(object(), FixedEmbedder())


def test_the_lexical_index_is_rebuilt_when_the_store_grows():
    store = make_store()
    retriever = HybridRetriever(store, FixedEmbedder())
    assert ids(retriever.retrieve("room_A", "zeta", k=5)) == ["A", "B", "C", "D"]          # nothing matches yet
    store.add(np.stack([unit(80)]), [chunk("E", "zeta", page=5)])
    after = retriever.retrieve("room_A", "zeta", k=5)
    assert after[0]["chunk_id"] == "E" and after[0]["lexical_rank"] == 1


def test_describe_reports_the_fixed_design_values():
    d = HybridRetriever(make_store(), FixedEmbedder(), exclude_non_content=True).describe()
    assert d["rrf_k"] == 60 and d["weights"] == "equal" and d["candidate_pool"] == 50 and d["exclude_non_content"] is True


# ------------------------------------------------------------------ the switch

def test_dense_is_the_default_and_hybrid_is_opt_in():
    store = make_store()
    default = build_retriever(store, FixedEmbedder())
    assert type(default) is Retriever and default.exclude_non_content is False
    assert type(build_retriever(store, FixedEmbedder(), mode="dense", exclude_non_content=True)) is Retriever
    hybrid = build_retriever(store, FixedEmbedder(), mode="hybrid", rrf_k=60)
    assert type(hybrid) is HybridRetriever
    with pytest.raises(ValueError):
        build_retriever(store, FixedEmbedder(), mode="sparse")
    with pytest.raises(ValueError):
        build_retriever(store, FixedEmbedder(), mode="dense", rrf_k=10)           # hybrid-only option on dense: refused, not ignored


def test_hybrid_works_as_the_module_level_default_retriever():
    from rafeeq.knowledge.retrieval import retrieve, set_default_retriever

    set_default_retriever(HybridRetriever(make_store(), FixedEmbedder()))
    try:
        assert retrieve("room_A", "delta", k=1)[0]["chunk_id"] == "C"
    finally:
        set_default_retriever(None)


def test_the_answerer_gate_still_works_on_cosine_when_fusion_ranks_a_low_cosine_chunk_first():
    from rafeeq.knowledge.answering.answerer import Answerer

    class RecordingLLM:
        def __init__(self):
            self.prompt = None

        def generate(self, system, prompt):
            self.prompt = prompt
            return "The answer is alpha [1]."

    store = make_store()
    retriever = HybridRetriever(store, FixedEmbedder())
    assert ids(retriever.retrieve("room_A", "epsilon", k=2))[0] == "D"            # fusion puts D (cosine 0.0) first ...
    llm = RecordingLLM()
    result = Answerer(retriever, llm, k=2, min_score=0.35).answer("room_A", "epsilon")
    excerpts = llm.prompt.split("Question:")[0]
    assert "alpha beta" in excerpts and "page 4" not in excerpts and "[2]" not in excerpts   # ... but the cosine gate keeps it out of the prompt
    assert result["grounded"] is True and result["citations"] == [{"file_name": "book.pdf", "page": 1}]

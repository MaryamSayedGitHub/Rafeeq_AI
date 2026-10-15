"""Unit tests for FAISSVectorStore. Tiny fake vectors only: no model is loaded."""

import json

import numpy as np
import pytest

from rafeeq.knowledge.vector_store import (
    DuplicateChunkError,
    FAISSVectorStore,
    SearchResult,
    VectorStoreLoadError,
)

DIM = 8


def unit(*values: float) -> np.ndarray:
    """Normalised float32 vector of length DIM from the leading components."""
    v = np.zeros(DIM, dtype=np.float32)
    v[: len(values)] = values
    return v / np.linalg.norm(v)


def axis(i: int) -> np.ndarray:
    return unit(*([0.0] * i + [1.0]))


def make_chunk(room="A", doc="docA", n=1, page=1, **extra) -> dict:
    chunk = {
        "chunk_id": f"{room}-{doc}-c{n}",
        "room_id": room,
        "file_name": f"{doc}.pdf",
        "page": page,
        "block_type": "text",
        "text": f"text of {room}/{doc}/{n}",
    }
    chunk.update(extra)
    return chunk


@pytest.fixture()
def store() -> FAISSVectorStore:
    return FAISSVectorStore(DIM)


@pytest.fixture()
def multi_room_store() -> FAISSVectorStore:
    """Room A: book + lecture (3 chunks). Room B: one book chunk. Different directions."""
    s = FAISSVectorStore(DIM)
    chunks = [
        make_chunk("A", "book", 1, page=10, document_type="book"),
        make_chunk("A", "book", 2, page=11, document_type="book"),
        make_chunk("A", "lecture1", 1, page=1, document_type="lecture"),
        make_chunk("B", "book", 1, page=5, document_type="book"),
    ]
    vectors = np.stack([axis(0), unit(1, 1), axis(1), axis(0)])  # B's chunk equals A/book/1
    s.add(vectors, chunks)
    return s


# ---------- initialization ----------

def test_empty_store_works(store):
    assert len(store) == 0
    assert store.dimension == DIM
    assert store.search(axis(0)) == []


@pytest.mark.parametrize("bad", [0, -3, 2.5, "8", None, True])
def test_invalid_dimension_rejected(bad):
    with pytest.raises(ValueError, match="dimension"):
        FAISSVectorStore(bad)


# ---------- add ----------

def test_add_vectors_and_metadata_counts_match(store):
    added = store.add(np.stack([axis(0), axis(1), axis(2)]), [make_chunk(n=i) for i in range(3)])
    assert added == 3
    assert len(store) == 3
    assert len(store._records) == len(store) == store._index.ntotal


def test_add_accepts_float64_and_converts(store):
    store.add(np.stack([axis(0)]).astype(np.float64), [make_chunk()])
    assert len(store) == 1


def test_add_can_be_called_multiple_times(store):
    store.add(np.stack([axis(0)]), [make_chunk(n=1)])
    store.add(np.stack([axis(1)]), [make_chunk(n=2)])
    assert len(store) == 2
    assert store.search(axis(1), top_k=1)[0].metadata["chunk_id"] == "A-docA-c2"


def test_mismatched_lengths_raise_and_nothing_is_added(store):
    with pytest.raises(ValueError, match="does not match number of chunks"):
        store.add(np.stack([axis(0), axis(1)]), [make_chunk(n=1)])
    with pytest.raises(ValueError, match="does not match number of chunks"):
        store.add(np.stack([axis(0)]), [make_chunk(n=1), make_chunk(n=2)])
    assert len(store) == 0 and store._records == []


def test_wrong_embedding_dimension_rejected(store):
    wrong = np.ones((1, DIM + 1), dtype=np.float32) / np.sqrt(DIM + 1)
    with pytest.raises(ValueError, match="does not match store dimension"):
        store.add(wrong, [make_chunk()])


def test_1d_embedding_rejected_with_hint(store):
    with pytest.raises(ValueError, match="2D"):
        store.add(axis(0), [make_chunk()])


def test_empty_inputs_rejected(store):
    with pytest.raises(ValueError, match="must not be empty"):
        store.add(np.empty((0, DIM), dtype=np.float32), [])
    with pytest.raises(ValueError, match="must not be empty"):
        store.add(np.stack([axis(0)]), [])


def test_non_normalised_embeddings_rejected(store):
    with pytest.raises(ValueError, match="L2-normalised"):
        store.add(np.full((1, DIM), 2.0, dtype=np.float32), [make_chunk()])


def test_nan_embeddings_rejected(store):
    bad = np.stack([axis(0)])
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        store.add(bad, [make_chunk()])


@pytest.mark.parametrize("missing", ["chunk_id", "room_id", "file_name", "page", "block_type", "text"])
def test_missing_metadata_field_rejected(store, missing):
    chunk = make_chunk()
    del chunk[missing]
    with pytest.raises(ValueError, match=missing):
        store.add(np.stack([axis(0)]), [chunk])
    assert len(store) == 0


@pytest.mark.parametrize("field,value", [("room_id", ""), ("text", "   "), ("page", 0), ("page", "3"), ("page", True)])
def test_invalid_metadata_value_rejected(store, field, value):
    with pytest.raises(ValueError, match=field):
        store.add(np.stack([axis(0)]), [make_chunk(**{field: value})])


def test_non_dict_chunk_rejected(store):
    with pytest.raises(TypeError, match="must be a dict"):
        store.add(np.stack([axis(0)]), ["not a dict"])


def test_failed_add_is_atomic_when_a_later_chunk_is_invalid(store):
    good, bad = make_chunk(n=1), make_chunk(n=2)
    del bad["text"]
    with pytest.raises(ValueError):
        store.add(np.stack([axis(0), axis(1)]), [good, bad])
    assert len(store) == 0


def test_document_id_defaults_to_file_name_and_extra_fields_are_preserved(store):
    store.add(np.stack([axis(0)]), [make_chunk(source_url="http://x", section="3.2")])
    meta = store.search(axis(0))[0].metadata
    assert meta["document_id"] == "docA.pdf"
    assert meta["source_url"] == "http://x"
    assert meta["section"] == "3.2"


def test_input_chunks_are_not_mutated(store):
    chunk = make_chunk()
    snapshot = dict(chunk)
    store.add(np.stack([axis(0)]), [chunk])
    assert chunk == snapshot and "document_id" not in chunk


# ---------- duplicates ----------

def test_adding_same_chunk_twice_is_rejected(store):
    store.add(np.stack([axis(0)]), [make_chunk()])
    with pytest.raises(DuplicateChunkError, match="append-only"):
        store.add(np.stack([axis(0)]), [make_chunk()])
    assert len(store) == 1


def test_duplicate_inside_one_batch_is_rejected_atomically(store):
    with pytest.raises(DuplicateChunkError):
        store.add(np.stack([axis(0), axis(1)]), [make_chunk(n=1), make_chunk(n=1)])
    assert len(store) == 0


def test_same_chunk_id_in_different_rooms_is_not_a_duplicate(store):
    a = make_chunk("A", n=1, chunk_id="same")
    b = make_chunk("B", n=1, chunk_id="same")
    store.add(np.stack([axis(0), axis(0)]), [a, b])
    assert len(store) == 2


# ---------- search ----------

def test_nearest_vector_is_returned_first_with_score_text_and_metadata(store):
    store.add(np.stack([axis(0), axis(1), axis(2)]), [make_chunk(n=i, page=i + 1) for i in range(3)])
    results = store.search(axis(1), top_k=3)
    assert all(isinstance(r, SearchResult) for r in results)
    best = results[0]
    assert best.metadata["chunk_id"] == "A-docA-c1"
    assert best.text == "text of A/docA/1"
    assert best.score == pytest.approx(1.0, abs=1e-5)
    assert best.metadata["page"] == 2
    assert best.metadata["file_name"] == "docA.pdf"
    assert best.metadata["room_id"] == "A"
    assert "text" not in best.metadata


def test_scores_are_cosine_similarities_sorted_descending(store):
    store.add(np.stack([axis(0), unit(1, 1), axis(1)]), [make_chunk(n=i) for i in range(3)])
    results = store.search(axis(0), top_k=3)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[1].score == pytest.approx(np.sqrt(0.5), abs=1e-5)  # cos 45 degrees
    assert results[2].score == pytest.approx(0.0, abs=1e-5)


def test_top_k_limits_results(store):
    store.add(np.stack([axis(i) for i in range(5)]), [make_chunk(n=i) for i in range(5)])
    assert len(store.search(axis(0), top_k=2)) == 2
    assert len(store.search(axis(0), top_k=5)) == 5


def test_top_k_larger_than_store_returns_everything(store):
    store.add(np.stack([axis(0), axis(1)]), [make_chunk(n=i) for i in range(2)])
    assert len(store.search(axis(0), top_k=50)) == 2


@pytest.mark.parametrize("bad", [0, -1, 2.5, "3", None, True])
def test_invalid_top_k_rejected(store, bad):
    with pytest.raises(ValueError, match="top_k"):
        store.search(axis(0), top_k=bad)


def test_invalid_top_k_rejected_even_on_empty_store(store):
    with pytest.raises(ValueError, match="top_k"):
        store.search(axis(0), top_k=0)


def test_query_wrong_dimension_rejected(store):
    store.add(np.stack([axis(0)]), [make_chunk()])
    with pytest.raises(ValueError, match="does not match store dimension"):
        store.search(np.ones(DIM + 2, dtype=np.float32) / np.sqrt(DIM + 2))


def test_query_must_be_1d(store):
    with pytest.raises(ValueError, match="1D"):
        store.search(np.stack([axis(0), axis(1)]))


def test_non_normalised_query_rejected(store):
    store.add(np.stack([axis(0)]), [make_chunk()])
    with pytest.raises(ValueError, match="L2-normalised"):
        store.search(np.full(DIM, 3.0, dtype=np.float32))


def test_query_shaped_1_by_dim_is_accepted(store):
    store.add(np.stack([axis(0)]), [make_chunk()])
    assert len(store.search(axis(0).reshape(1, -1))) == 1


# ---------- room filtering ----------

def test_no_filters_searches_the_whole_index(multi_room_store):
    results = multi_room_store.search(axis(0), top_k=10)
    assert {r.metadata["room_id"] for r in results} == {"A", "B"}
    assert len(results) == 4


def test_only_requested_room_is_returned(multi_room_store):
    for room in ("A", "B"):
        results = multi_room_store.search(axis(0), top_k=10, room_id=room)
        assert results and all(r.metadata["room_id"] == room for r in results)


def test_room_a_query_never_returns_room_b_even_if_b_is_the_closest_match(multi_room_store):
    # B/book/1 is an exact match for axis(0) but must not appear for room A.
    results = multi_room_store.search(axis(0), top_k=1, room_id="A")
    assert [r.metadata["room_id"] for r in results] == ["A"]


def test_nonexistent_room_returns_empty_list(multi_room_store):
    assert multi_room_store.search(axis(0), top_k=5, room_id="does-not-exist") == []


def test_room_with_fewer_chunks_than_top_k_returns_only_available(multi_room_store):
    results = multi_room_store.search(axis(0), top_k=10, room_id="B")
    assert len(results) == 1 and results[0].metadata["room_id"] == "B"


def test_room_with_many_chunks_returns_correct_top_k(multi_room_store):
    assert len(multi_room_store.search(axis(0), top_k=2, room_id="A")) == 2


def test_filter_finds_matches_buried_far_below_the_initial_candidate_pool(store):
    # 40 Room B chunks all closer to the query than the 3 Room A chunks.
    # top_k=2 -> initial pool is 20, which contains ONLY room B; the pool must grow.
    near = [unit(1.0, 0.001 * (i + 1)) for i in range(40)]
    far = [unit(0.1, 1.0, 0.001 * (i + 1)) for i in range(3)]
    store.add(np.stack(near + far),
              [make_chunk("B", n=i) for i in range(40)] + [make_chunk("A", n=i) for i in range(3)])
    results = store.search(axis(0), top_k=2, room_id="A")
    assert len(results) == 2
    assert all(r.metadata["room_id"] == "A" for r in results)
    assert results[0].score >= results[1].score


def test_filtered_results_are_the_true_top_k_of_that_room(store):
    vectors = [unit(1.0, 0.02 * i) for i in range(30)]  # decreasing similarity to axis(0)
    chunks = [make_chunk("B" if i % 3 else "A", n=i) for i in range(30)]
    store.add(np.stack(vectors), chunks)
    results = store.search(axis(0), top_k=3, room_id="A")
    expected = [f"A-docA-c{i}" for i in (0, 3, 6)]
    assert [r.metadata["chunk_id"] for r in results] == expected


# ---------- document filtering ----------

def test_document_filter_works(multi_room_store):
    # document_id defaults to file_name ("lecture1.pdf") when not given explicitly
    results = multi_room_store.search(axis(0), top_k=10, document_id="lecture1.pdf")
    assert len(results) == 1 and results[0].metadata["file_name"] == "lecture1.pdf"
    assert multi_room_store.search(axis(0), top_k=10, document_id="lecture1") == []


def test_explicit_document_id_is_used_for_filtering(store):
    store.add(np.stack([axis(0), axis(1)]),
              [make_chunk(n=1, document_id="hands_on_llm"), make_chunk(n=2, document_id="other")])
    results = store.search(axis(0), top_k=5, document_id="hands_on_llm")
    assert [r.metadata["document_id"] for r in results] == ["hands_on_llm"]


def test_document_type_filter_works(multi_room_store):
    results = multi_room_store.search(axis(0), top_k=10, document_type="lecture")
    assert len(results) == 1 and results[0].metadata["document_type"] == "lecture"


def test_unknown_document_returns_empty_list(multi_room_store):
    assert multi_room_store.search(axis(0), document_id="nope") == []


# ---------- combined filtering ----------

def test_room_and_document_filters_apply_together(multi_room_store):
    # "book.pdf" exists in rooms A and B; only A's must come back.
    results = multi_room_store.search(axis(0), top_k=10, room_id="A", document_id="book.pdf")
    assert len(results) == 2
    assert all(r.metadata["room_id"] == "A" and r.metadata["file_name"] == "book.pdf" for r in results)


def test_room_document_and_type_filters_apply_together(multi_room_store):
    results = multi_room_store.search(
        axis(0), top_k=10, room_id="A", document_id="lecture1.pdf", document_type="lecture")
    assert len(results) == 1
    assert multi_room_store.search(
        axis(0), top_k=10, room_id="A", document_id="lecture1.pdf", document_type="book") == []


def test_combined_filter_with_no_match_returns_empty(multi_room_store):
    assert multi_room_store.search(axis(0), room_id="B", document_id="lecture1.pdf") == []


@pytest.mark.parametrize("kwarg", ["room_id", "document_id", "document_type"])
def test_empty_string_filter_rejected(multi_room_store, kwarg):
    with pytest.raises(ValueError, match=kwarg):
        multi_room_store.search(axis(0), **{kwarg: ""})


# ---------- persistence ----------

def test_save_creates_both_files(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path / "vs")
    assert (tmp_path / "vs" / "index.faiss").is_file()
    assert (tmp_path / "vs" / "metadata.json").is_file()


def test_load_preserves_count_dimension_and_metadata(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    loaded = FAISSVectorStore.load(tmp_path)
    assert len(loaded) == len(multi_room_store) == 4
    assert loaded.dimension == DIM
    assert loaded._records == multi_room_store._records


def test_search_after_load_matches_search_before_save(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    loaded = FAISSVectorStore.load(tmp_path)
    for query in (axis(0), axis(1), unit(1, 1)):
        for kwargs in ({}, {"room_id": "A"}, {"room_id": "A", "document_id": "book.pdf"}):
            before = multi_room_store.search(query, top_k=4, **kwargs)
            after = loaded.search(query, top_k=4, **kwargs)
            assert [r.metadata for r in before] == [r.metadata for r in after]
            assert [r.text for r in before] == [r.text for r in after]
            assert [r.score for r in before] == pytest.approx([r.score for r in after])


def test_arabic_and_extra_fields_survive_save_and_load(store, tmp_path):
    store.add(np.stack([axis(0)]), [make_chunk(text="يعني إيه Self-Attention؟", note="ملاحظة")])
    store.save(tmp_path)
    assert "يعني" in (tmp_path / "metadata.json").read_text(encoding="utf-8")  # not \u-escaped
    hit = FAISSVectorStore.load(tmp_path).search(axis(0))[0]
    assert hit.text == "يعني إيه Self-Attention؟"
    assert hit.metadata["note"] == "ملاحظة"


def test_loaded_store_accepts_more_data_and_still_rejects_duplicates(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    loaded = FAISSVectorStore.load(tmp_path)
    loaded.add(np.stack([axis(5)]), [make_chunk("C", "new", 1)])
    assert len(loaded) == 5
    assert loaded.search(axis(5), top_k=1)[0].metadata["room_id"] == "C"
    with pytest.raises(DuplicateChunkError):
        loaded.add(np.stack([axis(0)]), [make_chunk("A", "book", 1, page=10, document_type="book")])


def test_save_empty_store_and_load_it(store, tmp_path):
    store.save(tmp_path)
    loaded = FAISSVectorStore.load(tmp_path)
    assert len(loaded) == 0 and loaded.search(axis(0)) == []


def test_saving_twice_overwrites_cleanly(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    multi_room_store.add(np.stack([axis(6)]), [make_chunk("C", n=1)])
    multi_room_store.save(tmp_path)
    assert len(FAISSVectorStore.load(tmp_path)) == 5
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.faiss", "metadata.json"]


def test_load_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="index.faiss"):
        FAISSVectorStore.load(tmp_path / "nope")


def test_load_missing_metadata_file_raises(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    (tmp_path / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        FAISSVectorStore.load(tmp_path)


def test_load_missing_index_file_raises(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    (tmp_path / "index.faiss").unlink()
    with pytest.raises(FileNotFoundError, match="index.faiss"):
        FAISSVectorStore.load(tmp_path)


def test_load_corrupt_json_raises(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    (tmp_path / "metadata.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(VectorStoreLoadError, match="not valid JSON"):
        FAISSVectorStore.load(tmp_path)


def _edit_metadata(path, mutate):
    meta_path = path / "metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    mutate(payload)
    meta_path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_detects_index_and_metadata_out_of_sync(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    _edit_metadata(tmp_path, lambda p: p["records"].pop())
    with pytest.raises(VectorStoreLoadError, match="out of sync"):
        FAISSVectorStore.load(tmp_path)


def test_load_detects_dimension_mismatch(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    _edit_metadata(tmp_path, lambda p: p.update(dimension=DIM + 1))
    with pytest.raises(VectorStoreLoadError, match="dimension"):
        FAISSVectorStore.load(tmp_path)


def test_load_rejects_unknown_format_version(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    _edit_metadata(tmp_path, lambda p: p.update(format_version=99))
    with pytest.raises(VectorStoreLoadError, match="format_version"):
        FAISSVectorStore.load(tmp_path)


def test_load_rejects_invalid_record(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    _edit_metadata(tmp_path, lambda p: p["records"][0].pop("page"))
    with pytest.raises(VectorStoreLoadError, match="invalid record"):
        FAISSVectorStore.load(tmp_path)


def test_load_rejects_duplicate_identities(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    _edit_metadata(tmp_path, lambda p: p["records"].__setitem__(1, dict(p["records"][0])))
    with pytest.raises(VectorStoreLoadError, match="duplicate"):
        FAISSVectorStore.load(tmp_path)


def test_load_rejects_garbage_index_file(multi_room_store, tmp_path):
    multi_room_store.save(tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"this is not a faiss index")
    with pytest.raises(VectorStoreLoadError, match="could not read FAISS index"):
        FAISSVectorStore.load(tmp_path)


# ---------- compatibility with the real chunker output ----------

def test_accepts_chunker_output_unchanged(store):
    from rafeeq.knowledge.ingestion.chunker import chunk_pages

    pages = [{"page": 1, "text": "alpha beta gamma"}, {"page": 2, "text": ""}, {"page": 3, "text": "delta"}]
    chunks = chunk_pages(pages, "llm_101", "Hands-On Large Language Models.pdf")
    store.add(np.stack([axis(0), axis(1)]), chunks)
    hit = store.search(axis(1), top_k=1, room_id="llm_101")[0]
    assert hit.metadata["page"] == 3
    assert hit.metadata["file_name"] == "Hands-On Large Language Models.pdf"
    assert hit.metadata["chunk_id"] == chunks[1]["chunk_id"]

# ------------------------------------------- read-only helpers used by hybrid retrieval

def _unit_pair(degrees):
    import math

    r = math.radians(degrees)
    return np.array([math.cos(r), math.sin(r)], dtype=np.float32)


def _two_room_store():
    store = FAISSVectorStore(2)
    rows = [("room_A", "a1", "first", 0), ("room_A", "a2", "second", 45), ("room_B", "b1", "foreign", 90)]
    chunks = [{"chunk_id": cid, "room_id": room, "file_name": "f.pdf", "page": i + 1, "block_type": "text", "text": text}
              for i, (room, cid, text, _) in enumerate(rows)]
    store.add(np.stack([_unit_pair(d) for *_, d in rows]), chunks)
    return store


def test_room_chunks_returns_copies_of_one_rooms_records_with_text():
    store = _two_room_store()
    chunks = store.room_chunks("room_A")
    assert [c["chunk_id"] for c in chunks] == ["a1", "a2"] and [c["text"] for c in chunks] == ["first", "second"]
    assert all(c["room_id"] == "room_A" and c["document_id"] == "f.pdf" for c in chunks)
    chunks[0]["text"] = "changed"
    assert store.room_chunks("room_A")[0]["text"] == "first"                    # a copy: the store is not affected
    assert store.room_chunks("nobody") == []
    with pytest.raises(ValueError):
        store.room_chunks("")


def test_similarities_equal_what_search_reports_and_never_cross_rooms():
    store = _two_room_store()
    query = _unit_pair(10)
    searched = {h.metadata["chunk_id"]: h.score for h in store.search(query, top_k=5, room_id="room_A")}
    got = store.similarities(query, "room_A", [("f.pdf", "a1"), ("f.pdf", "a2"), ("f.pdf", "b1"), ("f.pdf", "missing")])
    assert got == {("f.pdf", "a1"): searched["a1"], ("f.pdf", "a2"): searched["a2"]}   # b1 is in another room, "missing" is unknown
    with pytest.raises(ValueError):
        store.similarities(np.array([2.0, 0.0], dtype=np.float32), "room_A", [("f.pdf", "a1")])   # not normalised
    with pytest.raises(ValueError):
        store.similarities(np.zeros(3, dtype=np.float32), "room_A", [])                            # wrong dimension


def test_similarities_see_chunks_added_after_the_first_call():
    store = _two_room_store()
    assert store.similarities(_unit_pair(0), "room_A", [("f.pdf", "a3")]) == {}
    store.add(np.stack([_unit_pair(0)]), [{"chunk_id": "a3", "room_id": "room_A", "file_name": "f.pdf", "page": 9, "block_type": "text", "text": "late"}])
    assert store.similarities(_unit_pair(0), "room_A", [("f.pdf", "a3")])[("f.pdf", "a3")] == pytest.approx(1.0)


# ---------- remove a document ----------
# (a chunk without document_id gets its file_name as document_id, so "book" is stored as "book.pdf")

def test_remove_document_drops_only_that_document_in_that_room(multi_room_store):
    assert multi_room_store.remove_document("A", "book.pdf") == 2
    assert len(multi_room_store) == 2
    left = {(r.metadata["room_id"], r.metadata["document_id"]) for r in multi_room_store.search(axis(0), top_k=10)}
    assert left == {("A", "lecture1.pdf"), ("B", "book.pdf")}  # room B's "book" is untouched
    hit = multi_room_store.search(axis(1), top_k=1, room_id="A")[0]
    assert hit.metadata["document_id"] == "lecture1.pdf" and hit.score == pytest.approx(1.0)


def test_remove_unknown_document_changes_nothing_and_ids_can_be_reused(multi_room_store, tmp_path):
    assert multi_room_store.remove_document("A", "nope") == 0
    assert len(multi_room_store) == 4
    multi_room_store.remove_document("A", "lecture1.pdf")
    multi_room_store.add(np.stack([axis(2)]), [make_chunk("A", "lecture1", 1)])  # same identity, added again
    multi_room_store.save(tmp_path)
    again = FAISSVectorStore.load(tmp_path)
    assert len(again) == 4
    assert again.search(axis(2), top_k=1, room_id="A")[0].metadata["document_id"] == "lecture1.pdf"


def test_removing_everything_leaves_a_working_empty_store(store, tmp_path):
    store.add(np.stack([axis(0)]), [make_chunk("A", "doc", 1)])
    assert store.remove_document("A", "doc.pdf") == 1
    assert len(store) == 0 and store.search(axis(0)) == []
    store.save(tmp_path)
    assert len(FAISSVectorStore.load(tmp_path)) == 0

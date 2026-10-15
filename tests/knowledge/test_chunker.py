import copy

import pytest

from rafeeq.knowledge.ingestion.chunker import chunk_pages

ROOM = "room-A"
FILE = "book.pdf"


def words(prefix: str, n: int) -> str:
    """n distinct words: 'p3w0 p3w1 ...' (prefix identifies the page)."""
    return " ".join(f"{prefix}w{i}" for i in range(n))


def page(number: int, n_words: int) -> dict:
    return {"page": number, "text": words(f"p{number}", n_words)}


def run(pages, **kwargs):
    return chunk_pages(pages, ROOM, FILE, **kwargs)


# ---------- metadata structure ----------

def test_chunk_has_exact_metadata_structure():
    chunks = run([page(10, 20)])
    assert len(chunks) == 1
    assert set(chunks[0].keys()) == {
        "chunk_id", "room_id", "file_name", "page", "block_type", "text",
    }


def test_room_id_and_file_name_preserved():
    chunks = chunk_pages([page(1, 900), page(2, 10)], "room-xyz", "lecture1.pdf")
    assert len(chunks) > 2
    assert all(c["room_id"] == "room-xyz" for c in chunks)
    assert all(c["file_name"] == "lecture1.pdf" for c in chunks)


def test_block_type_is_text():
    assert all(c["block_type"] == "text" for c in run([page(1, 900), page(2, 5)]))


def test_page_number_preserved_including_non_consecutive():
    chunks = run([page(5, 10), page(9, 10), page(12, 700)])
    assert [c["page"] for c in chunks if c["page"] == 5] == [5]
    assert {c["page"] for c in chunks} == {5, 9, 12}
    assert [c["page"] for c in chunks] == sorted(c["page"] for c in chunks)


# ---------- page boundaries ----------

def test_small_page_stays_one_chunk_with_same_text():
    p = page(1, 120)
    chunks = run([p])
    assert len(chunks) == 1
    assert chunks[0]["text"] == p["text"]


def test_page_with_exactly_chunk_size_words_is_one_chunk():
    assert len(run([page(1, 500)])) == 1


def test_page_with_chunk_size_plus_one_words_is_split():
    chunks = run([page(1, 501)])
    assert len(chunks) == 2
    assert len(chunks[0]["text"].split()) == 500
    assert len(chunks[1]["text"].split()) == 81  # 80 overlap + 1 new word


def test_small_adjacent_pages_are_not_merged():
    chunks = run([page(1, 50), page(2, 50), page(3, 50)])
    assert [c["page"] for c in chunks] == [1, 2, 3]
    for c in chunks:
        assert len(c["text"].split()) == 50


def test_no_chunk_contains_text_from_multiple_pages():
    pages = [page(1, 1300), page(2, 30), page(3, 999), page(4, 500)]
    chunks = run(pages)
    for c in chunks:
        prefixes = {w.split("w")[0] for w in c["text"].split()}
        assert prefixes == {f"p{c['page']}"}


def test_trailing_small_page_not_merged_with_previous_large_page():
    chunks = run([page(1, 700), page(2, 3)])
    assert chunks[-1]["page"] == 2
    assert chunks[-1]["text"] == page(2, 3)["text"]


# ---------- splitting and overlap ----------

def test_large_page_is_split_into_expected_windows():
    # 1200 words, size 500, overlap 80 -> step 420 -> windows start at 0, 420, 840
    chunks = run([page(1, 1200)])
    assert [len(c["text"].split()) for c in chunks] == [500, 500, 360]
    assert chunks[0]["text"].split()[0] == "p1w0"
    assert chunks[1]["text"].split()[0] == "p1w420"
    assert chunks[2]["text"].split()[0] == "p1w840"
    assert chunks[2]["text"].split()[-1] == "p1w1199"


def test_consecutive_chunks_overlap_by_exactly_overlap_words():
    chunks = run([page(1, 1500)])
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["text"].split()[-80:] == nxt["text"].split()[:80]


def test_custom_chunk_size_and_overlap():
    chunks = run([page(1, 12)], chunk_size=5, overlap=2)
    # step 3 -> starts 0, 3, 6, 9 ; last window 9..12
    assert [c["text"].split() for c in chunks] == [
        [f"p1w{i}" for i in range(0, 5)],
        [f"p1w{i}" for i in range(3, 8)],
        [f"p1w{i}" for i in range(6, 11)],
        [f"p1w{i}" for i in range(9, 12)],
    ]


def test_zero_overlap_gives_disjoint_chunks_that_cover_everything():
    p = page(1, 1100)
    chunks = run([p], overlap=0)
    assert [len(c["text"].split()) for c in chunks] == [500, 500, 100]
    rebuilt = " ".join(c["text"] for c in chunks)
    assert rebuilt == p["text"]


def test_no_words_lost_with_overlap():
    p = page(1, 1777)
    chunks = run([p])
    rebuilt = chunks[0]["text"].split()
    for c in chunks[1:]:
        rebuilt += c["text"].split()[80:]  # drop the overlapping prefix
    assert rebuilt == p["text"].split()


def test_last_chunk_is_never_fully_contained_in_previous():
    for n in (501, 580, 581, 920, 921, 1000):
        chunks = run([page(1, n)])
        assert chunks[-1]["text"].split()[-1] == f"p1w{n - 1}"
        new_words = set(chunks[-1]["text"].split()) - set(chunks[-2]["text"].split())
        assert new_words, f"redundant trailing chunk for n={n}"


def test_internal_newlines_are_preserved_and_ends_stripped():
    text = "  alpha beta\n\ngamma   delta \n"
    chunks = run([{"page": 1, "text": text}])
    assert chunks[0]["text"] == "alpha beta\n\ngamma   delta"


# ---------- empty pages ----------

def test_empty_page_produces_no_chunks():
    assert run([{"page": 1, "text": ""}]) == []


def test_whitespace_only_page_produces_no_chunks():
    assert run([{"page": 1, "text": " \n\t  \n"}]) == []


def test_empty_pages_between_text_pages_are_skipped_without_side_effects():
    chunks = run([page(1, 10), {"page": 2, "text": ""}, page(3, 10)])
    assert [c["page"] for c in chunks] == [1, 3]


def test_empty_page_list_returns_empty_list():
    assert run([]) == []


# ---------- deterministic IDs ----------

def test_chunk_ids_are_deterministic():
    pages = [page(1, 1300), page(2, 20)]
    assert [c["chunk_id"] for c in run(pages)] == [c["chunk_id"] for c in run(pages)]
    assert run(pages) == run(copy.deepcopy(pages))


def test_chunk_ids_are_unique():
    chunks = run([page(1, 1300), page(2, 20), page(3, 2000)])
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_differ_across_rooms_and_files():
    pages = [page(1, 10)]
    a = chunk_pages(pages, "room-A", "book.pdf")[0]["chunk_id"]
    b = chunk_pages(pages, "room-B", "book.pdf")[0]["chunk_id"]
    c = chunk_pages(pages, "room-A", "other.pdf")[0]["chunk_id"]
    assert len({a, b, c}) == 3


def test_chunk_id_does_not_depend_on_other_pages():
    alone = run([page(2, 10)])[0]["chunk_id"]
    with_neighbours = [c for c in run([page(1, 900), page(2, 10)]) if c["page"] == 2][0]["chunk_id"]
    assert alone == with_neighbours


# ---------- input is not mutated ----------

def test_input_pages_not_mutated():
    pages = [page(1, 700), {"page": 2, "text": ""}]
    snapshot = copy.deepcopy(pages)
    run(pages)
    assert pages == snapshot


# ---------- validation ----------

@pytest.mark.parametrize("bad_size", [0, -1, -500])
def test_chunk_size_must_be_positive(bad_size):
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        run([page(1, 10)], chunk_size=bad_size, overlap=0)


def test_overlap_must_not_be_negative():
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        run([page(1, 10)], overlap=-1)


@pytest.mark.parametrize("bad_overlap", [500, 501, 1000])
def test_overlap_must_be_smaller_than_chunk_size(bad_overlap):
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        run([page(1, 10)], chunk_size=500, overlap=bad_overlap)


@pytest.mark.parametrize("bad", [10.5, "500", None, True])
def test_chunk_size_must_be_an_integer(bad):
    with pytest.raises(ValueError, match="chunk_size must be an integer"):
        run([page(1, 10)], chunk_size=bad)


def test_overlap_must_be_an_integer():
    with pytest.raises(ValueError, match="overlap must be an integer"):
        run([page(1, 10)], overlap=2.5)


def test_validation_happens_even_when_there_are_no_pages():
    with pytest.raises(ValueError):
        run([], chunk_size=0)


@pytest.mark.parametrize("bad_room", ["", "   ", None])
def test_room_id_required(bad_room):
    with pytest.raises(ValueError, match="room_id"):
        chunk_pages([page(1, 10)], bad_room, FILE)


@pytest.mark.parametrize("bad_file", ["", "  ", None])
def test_file_name_required(bad_file):
    with pytest.raises(ValueError, match="file_name"):
        chunk_pages([page(1, 10)], ROOM, bad_file)


def test_malformed_page_entry_raises_clear_error():
    with pytest.raises(ValueError, match="'page' and 'text'"):
        run([{"page": 1}])


def test_duplicate_page_numbers_rejected():
    with pytest.raises(ValueError, match="duplicate page number 3"):
        run([page(3, 5), page(3, 5)])


def test_page_number_must_be_one_indexed_int():
    with pytest.raises(ValueError, match="integer >= 1"):
        run([{"page": 0, "text": "x"}])
"""Ingestion pipeline tests: real chunker, topic manager and FAISS store;
fake parser input / embedder (no BGE-M3, no PDF needed except where noted)."""

from pathlib import Path

import numpy as np
import pytest

from rafeeq.knowledge.ingestion.pipeline import IngestionStats, NoTextExtractedError, ingest_file
from rafeeq.knowledge.ingestion.scanned_pages import ScannedPagesNotAvailableError
from rafeeq.knowledge.topics import list_topics
from rafeeq.knowledge.vector_store import DuplicateChunkError, FAISSVectorStore

ROOM = "llm_101"
PDF = "data/raw/hands_on_llm.pdf"
BOOK_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "hands_on_llm.pdf"

SELF_ATTENTION = "Self-attention allows each token to attend to other tokens."
CNN = "Convolutional neural networks are useful for image processing."
GRADIENT = "Gradient descent is used to optimize neural networks."
PAGES = [{"page": 1, "text": SELF_ATTENTION}, {"page": 2, "text": CNN}, {"page": 3, "text": GRADIENT}]
GOOD = "This page has plenty of real extractable text content."


def parser_returning(pages):
    return lambda path: [dict(p) for p in pages]


def words(prefix, n):
    return " ".join(f"{prefix}w{i}" for i in range(n))


def ingest(store, embedder, pages=PAGES, **kwargs):
    kwargs.setdefault("parse", parser_returning(pages))
    return ingest_file(kwargs.pop("path", PDF), kwargs.pop("room_id", ROOM), store, embedder, **kwargs)


def all_hits(store, embedder, room_id=ROOM, k=100):
    return store.search(embedder.embed_query("self attention neural token"), top_k=k, room_id=room_id)


# ---------- basic flow and statistics ----------

def test_ingest_returns_useful_statistics(empty_store, hashing_embedder):
    stats = ingest(empty_store, hashing_embedder)
    assert isinstance(stats, IngestionStats)
    assert stats.room_id == ROOM
    assert stats.file_name == "hands_on_llm.pdf"
    assert stats.document_id == "hands_on_llm"
    assert stats.total_pages == 3 and stats.pages_with_text == 3
    assert stats.pages_needing_extraction == [] and stats.pages_extracted == [] and stats.pages_unextracted == []
    assert stats.chunk_count == 3 and stats.vectors_added == 3 and stats.vectors_in_store == 3
    assert sum(stats.topic_counts.values()) == 3
    assert stats.saved_to is None


def test_every_chunk_has_room_id_topic_id_and_preserved_metadata(empty_store, hashing_embedder):
    ingest(empty_store, hashing_embedder)
    hits = all_hits(empty_store, hashing_embedder)
    assert len(hits) == 3
    listed_topics = {t["topic_id"] for t in list_topics(ROOM)}
    for hit in hits:
        assert hit.metadata["room_id"] == ROOM
        assert hit.metadata["topic_id"] in listed_topics
        assert hit.metadata["file_name"] == "hands_on_llm.pdf"
        assert hit.metadata["block_type"] == "text"
        assert hit.metadata["document_id"] == "hands_on_llm"
    assert {h.metadata["page"]: h.text for h in hits} == {1: SELF_ATTENTION, 2: CNN, 3: GRADIENT}


def test_topics_are_assigned_after_chunking(empty_store, hashing_embedder):
    ingest(empty_store, hashing_embedder)
    by_page = {h.metadata["page"]: h.metadata["topic_id"] for h in all_hits(empty_store, hashing_embedder)}
    assert by_page == {1: "topic_attention", 2: "topic_deep_learning_basics", 3: "topic_deep_learning_basics"}


def test_chunks_never_cross_page_boundaries(empty_store, hashing_embedder):
    pages = [{"page": 1, "text": words("p1", 700)}, {"page": 2, "text": words("p2", 30)},
             {"page": 3, "text": words("p3", 520)}]
    stats = ingest(empty_store, hashing_embedder, pages=pages)
    hits = all_hits(empty_store, hashing_embedder)
    assert stats.chunk_count == len(hits) == 2 + 1 + 2
    for hit in hits:
        assert {w.split("w")[0] for w in hit.text.split()} == {f"p{hit.metadata['page']}"}


def test_embeddings_correspond_exactly_to_chunks(empty_store, hashing_embedder):
    ingest(empty_store, hashing_embedder)
    # exactly one embedder call, with exactly the chunk texts, in page order
    assert hashing_embedder.document_calls == [[SELF_ATTENTION, CNN, GRADIENT]]
    # and each stored vector belongs to its own chunk: querying with a page's text finds that page first
    for page in PAGES:
        top = empty_store.search(hashing_embedder.embed_query(page["text"]), top_k=1, room_id=ROOM)[0]
        assert top.metadata["page"] == page["page"] and top.text == page["text"]


def test_empty_pages_are_skipped_not_indexed(empty_store, hashing_embedder):
    pages = [{"page": 1, "text": SELF_ATTENTION}, {"page": 2, "text": ""}, {"page": 3, "text": GRADIENT}]
    stats = ingest(empty_store, hashing_embedder, pages=pages)
    assert stats.chunk_count == 2
    assert sorted(h.metadata["page"] for h in all_hits(empty_store, hashing_embedder)) == [1, 3]


def test_document_id_and_type_can_be_set(empty_store, hashing_embedder):
    stats = ingest(empty_store, hashing_embedder, document_id="llm-book", document_type="book")
    assert stats.document_id == "llm-book"
    hits = empty_store.search(hashing_embedder.embed_query("attention"), top_k=5, room_id=ROOM,
                              document_id="llm-book", document_type="book")
    assert len(hits) == 3
    assert empty_store.search(hashing_embedder.embed_query("attention"), top_k=5, document_type="lecture") == []


def test_chunk_size_and_overlap_are_passed_to_the_chunker(empty_store, hashing_embedder):
    stats = ingest(empty_store, hashing_embedder, pages=[{"page": 1, "text": words("p1", 12)}],
                   chunk_size=5, overlap=2)
    assert stats.chunk_count == 4


# ---------- rooms ----------

def test_same_file_in_two_rooms_stays_isolated(empty_store, hashing_embedder):
    ingest(empty_store, hashing_embedder, room_id="room_A")
    ingest(empty_store, hashing_embedder, room_id="room_B")
    assert len(empty_store) == 6
    query = hashing_embedder.embed_query("self attention")
    for room in ("room_A", "room_B"):
        hits = empty_store.search(query, top_k=10, room_id=room)
        assert len(hits) == 3 and {h.metadata["room_id"] for h in hits} == {room}


def test_ingesting_the_same_file_into_the_same_room_twice_is_rejected(empty_store, hashing_embedder):
    ingest(empty_store, hashing_embedder)
    with pytest.raises(DuplicateChunkError):
        ingest(empty_store, hashing_embedder)
    assert len(empty_store) == 3


def test_invalid_room_id_rejected_before_any_work(empty_store, hashing_embedder):
    with pytest.raises(ValueError, match="room_id"):
        ingest(empty_store, hashing_embedder, room_id="")
    assert hashing_embedder.document_calls == [] and len(empty_store) == 0


# ---------- scanned-page hook ----------

class RecordingExtractor:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result or [], error, []

    def __call__(self, file_path, pages):
        self.calls.append((file_path, list(pages)))
        if self.error:
            raise self.error
        return self.result


MIXED = [{"page": 1, "text": GOOD}, {"page": 2, "text": ""}, {"page": 3, "text": ""}]


def test_pages_without_text_are_sent_to_the_hook_and_results_are_indexed(empty_store, hashing_embedder):
    extractor = RecordingExtractor(result=[{"page": 2, "text": SELF_ATTENTION, "bbox": [0, 0, 1, 1]}])
    stats = ingest(empty_store, hashing_embedder, pages=MIXED, scanned_page_extractor=extractor)
    assert extractor.calls == [(PDF, [2, 3])]
    assert stats.pages_needing_extraction == [2, 3]
    assert stats.pages_extracted == [2] and stats.pages_unextracted == [3]
    assert stats.pages_with_text == 1 and stats.chunk_count == 2
    hit = empty_store.search(hashing_embedder.embed_query("self attention"), top_k=1, room_id=ROOM)[0]
    assert hit.metadata["page"] == 2 and hit.text == SELF_ATTENTION and hit.metadata["topic_id"] == "topic_attention"


def test_pages_with_text_do_not_trigger_the_hook(empty_store, hashing_embedder):
    extractor = RecordingExtractor()
    ingest(empty_store, hashing_embedder, scanned_page_extractor=extractor)
    assert extractor.calls == []


def test_default_stub_does_not_break_ingestion_and_reports_unextracted_pages(empty_store, hashing_embedder):
    stats = ingest(empty_store, hashing_embedder, pages=MIXED)  # default extractor = M4 stub
    assert stats.pages_needing_extraction == [2, 3] and stats.pages_extracted == []
    assert stats.pages_unextracted == [2, 3]
    assert stats.chunk_count == 1


def test_extractor_raising_not_implemented_is_tolerated(empty_store, hashing_embedder):
    extractor = RecordingExtractor(error=ScannedPagesNotAvailableError("later"))
    stats = ingest(empty_store, hashing_embedder, pages=MIXED, scanned_page_extractor=extractor)
    assert stats.pages_unextracted == [2, 3] and len(empty_store) == 1


def test_hook_can_be_disabled(empty_store, hashing_embedder):
    stats = ingest(empty_store, hashing_embedder, pages=MIXED, scanned_page_extractor=None)
    assert stats.pages_needing_extraction == [2, 3] and stats.pages_extracted == []


def test_short_parser_text_is_kept_when_extraction_gives_nothing(empty_store, hashing_embedder):
    pages = [{"page": 1, "text": GOOD}, {"page": 2, "text": "Figure 3"}]  # too short to count as "usable"
    stats = ingest(empty_store, hashing_embedder, pages=pages)
    assert stats.pages_unextracted == [2] and stats.chunk_count == 2


def test_other_extractor_errors_propagate_and_nothing_is_indexed(empty_store, hashing_embedder):
    extractor = RecordingExtractor(error=RuntimeError("vision model crashed"))
    with pytest.raises(RuntimeError, match="vision model crashed"):
        ingest(empty_store, hashing_embedder, pages=MIXED, scanned_page_extractor=extractor)
    assert len(empty_store) == 0


def test_extractor_returning_an_unrequested_page_is_rejected(empty_store, hashing_embedder):
    extractor = RecordingExtractor(result=[{"page": 1, "text": "hijacked"}])
    with pytest.raises(ValueError, match="not requested"):
        ingest(empty_store, hashing_embedder, pages=MIXED, scanned_page_extractor=extractor)
    assert len(empty_store) == 0


# ---------- failure handling leaves the store untouched ----------

def test_a_file_with_no_text_raises_and_indexes_nothing(empty_store, hashing_embedder):
    with pytest.raises(NoTextExtractedError, match="no text"):
        ingest(empty_store, hashing_embedder, pages=[{"page": 1, "text": ""}, {"page": 2, "text": " "}])
    assert len(empty_store) == 0 and hashing_embedder.document_calls == []


class ShortChangingEmbedder:
    dimension = 256

    def embed_documents(self, texts):
        return np.zeros((len(texts) - 1, 256), dtype=np.float32)


def test_embedding_count_mismatch_is_refused(empty_store):
    with pytest.raises(RuntimeError, match="refusing to index"):
        ingest(empty_store, ShortChangingEmbedder())
    assert len(empty_store) == 0


class UnnormalisedEmbedder:
    def embed_documents(self, texts):
        return np.full((len(texts), 256), 5.0, dtype=np.float32)


def test_store_validation_failure_leaves_the_store_empty(empty_store):
    with pytest.raises(ValueError, match="L2-normalised"):
        ingest(empty_store, UnnormalisedEmbedder())
    assert len(empty_store) == 0


# ---------- persistence ----------

def test_save_dir_persists_the_store(empty_store, hashing_embedder, tmp_path):
    stats = ingest(empty_store, hashing_embedder, save_dir=tmp_path / "vs")
    assert stats.saved_to == str(tmp_path / "vs")
    loaded = FAISSVectorStore.load(tmp_path / "vs")
    assert len(loaded) == 3
    top = loaded.search(hashing_embedder.embed_query("self attention"), top_k=1, room_id=ROOM)[0]
    assert top.metadata["page"] == 1 and top.metadata["topic_id"] == "topic_attention"


def test_nothing_is_saved_unless_requested(empty_store, hashing_embedder, tmp_path):
    ingest(empty_store, hashing_embedder)
    assert list(tmp_path.iterdir()) == []


# ---------- real PDF through the real parser ----------

def test_ingest_a_real_small_pdf(empty_store, hashing_embedder, tmp_path):
    import fitz

    pdf_path = tmp_path / "mini_course.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), SELF_ATTENTION)
    doc.new_page()  # blank page in the middle
    doc.new_page().insert_text((72, 72), GRADIENT)
    doc.save(str(pdf_path))
    doc.close()

    stats = ingest_file(pdf_path, ROOM, empty_store, hashing_embedder)  # real parse_pdf
    assert stats.file_name == "mini_course.pdf" and stats.total_pages == 3
    assert stats.pages_needing_extraction == [2] and stats.pages_unextracted == [2]
    assert stats.chunk_count == 2
    assert sorted(h.metadata["page"] for h in all_hits(empty_store, hashing_embedder)) == [1, 3]


@pytest.mark.skipif(not BOOK_PATH.exists(), reason="data/raw/hands_on_llm.pdf not present")
def test_the_real_book_can_be_ingested(empty_store, hashing_embedder):
    stats = ingest_file(BOOK_PATH, "llm_101", empty_store, hashing_embedder)
    print(f"\npages={stats.total_pages} chunks={stats.chunk_count} unextracted={len(stats.pages_unextracted)}")
    print("topics:", stats.topic_counts)
    assert stats.total_pages > 0 and stats.chunk_count > 0
    assert sum(stats.topic_counts.values()) == stats.chunk_count
    assert len(stats.topic_counts) >= 3  # the keyword taxonomy should find several topics in a real LLM book
    assert stats.vectors_in_store == stats.chunk_count
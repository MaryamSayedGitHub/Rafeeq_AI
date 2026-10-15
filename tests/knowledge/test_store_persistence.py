"""Safe load / create / save rules for the persisted FAISS store.

Uses the REAL FAISSVectorStore with the deterministic HashingEmbedder (no model, no
network) and temporary directories. The most important rule under test: a missing,
incomplete or corrupt store is an ERROR that leaves the files untouched, never a reason
to silently start an empty store that a later save would write over the real one.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from kb_fakes import HashingEmbedder
from rafeeq.knowledge import store_persistence as sp
from rafeeq.knowledge.ingestion.chunker import chunk_pages
from rafeeq.knowledge.store_persistence import (
    DocumentAlreadyIngestedError,
    PersistenceError,
    StoreNotFoundError,
    document_exists,
    ensure_dimension,
    find_leftover_files,
    load_existing_store,
    load_or_create_store,
    save_store_safely,
    store_inventory,
)
from rafeeq.knowledge.topics import TopicManager
from rafeeq.knowledge.vector_store import FAISSVectorStore, VectorStoreLoadError
from rafeeq.knowledge.vector_store import faiss_store

BOOK = [
    "Self-attention allows each token to attend to other tokens.",
    "Convolutional neural networks are useful for image processing.",
    "Gradient descent is used to optimize neural networks.",
]


def make_store(rooms, embedder=None):
    """rooms = {room_id: {file_name: [page text, ...]}} -> populated real store."""
    embedder = embedder or HashingEmbedder()
    store = FAISSVectorStore(embedder.dimension)
    manager = TopicManager()
    for room_id, files in rooms.items():
        for file_name, texts in files.items():
            pages = [{"page": i + 1, "text": t} for i, t in enumerate(texts)]
            chunks = manager.tag_chunks(room_id, chunk_pages(pages, room_id, file_name))
            store.add(embedder.embed_documents([c["text"] for c in chunks]), chunks)
    return store


def snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(Path(directory).iterdir()) if p.is_file()}


class IndexOnlySave:
    """The dangerous half-save: writes the NEW index.faiss next to the OLD metadata.json, then fails."""

    def __init__(self, store):
        self._store = store

    def __len__(self):
        return len(self._store)

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self._store.save(tmp)
            shutil.copy2(Path(tmp) / "index.faiss", Path(directory) / "index.faiss")
        raise OSError("disk full while writing metadata.json")


class SilentNoSave:
    """A save() that claims nothing and writes nothing."""

    def __init__(self, store):
        self._store = store

    def __len__(self):
        return len(self._store)

    def save(self, directory):
        pass


class InterruptedSave(IndexOnlySave):
    def save(self, directory):
        try:
            super().save(directory)
        except OSError:
            raise KeyboardInterrupt


# ------------------------------------------------------------------ constants

def test_file_names_match_the_store_module():
    assert sp.INDEX_FILE_NAME == faiss_store.INDEX_FILE_NAME
    assert sp.METADATA_FILE_NAME == faiss_store.METADATA_FILE_NAME


# ----------------------------------------------------------- load / create

def test_missing_directory_creates_a_new_empty_store(tmp_path):
    store, created = load_or_create_store(tmp_path / "nope", 256, store_cls=FAISSVectorStore)
    assert created is True and len(store) == 0 and store.dimension == 256
    assert not (tmp_path / "nope").exists()  # creating the object writes nothing


def test_empty_existing_directory_creates_a_new_store(tmp_path):
    (tmp_path / "notes.txt").write_text("unrelated file")
    store, created = load_or_create_store(tmp_path, 256, store_cls=FAISSVectorStore)
    assert created is True and len(store) == 0


def test_dimension_callable_is_only_called_when_a_new_store_is_needed(tmp_path):
    calls = []

    def dimension():
        calls.append(1)
        return 256

    load_or_create_store(tmp_path / "new", dimension, store_cls=FAISSVectorStore)
    assert calls == [1]

    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path / "saved")
    store, created = load_or_create_store(tmp_path / "saved", dimension, store_cls=FAISSVectorStore)
    assert created is False and len(store) == 3
    assert calls == [1]  # the model would not have been loaded to open an existing store


def test_complete_store_is_loaded_with_the_same_content(tmp_path):
    embedder = HashingEmbedder()
    original = make_store({"room_A": {"book.pdf": BOOK}}, embedder)
    original.save(tmp_path)

    loaded, created = load_or_create_store(tmp_path, 256, store_cls=FAISSVectorStore)

    assert created is False and len(loaded) == len(original) == 3
    query = embedder.embed_query("What does self-attention do?")
    before = [(r.text, r.score, r.metadata) for r in original.search(query, top_k=3, room_id="room_A")]
    after = [(r.text, r.score, r.metadata) for r in loaded.search(query, top_k=3, room_id="room_A")]
    assert before == after and before[0][0] == BOOK[0]


def test_load_existing_store_never_creates_one(tmp_path):
    with pytest.raises(StoreNotFoundError, match="no vector store found"):
        load_existing_store(tmp_path / "nothing", store_cls=FAISSVectorStore)
    assert issubclass(StoreNotFoundError, PersistenceError)


@pytest.mark.parametrize("missing", ["index.faiss", "metadata.json"])
def test_one_missing_file_is_an_error_and_nothing_is_created_or_changed(tmp_path, missing):
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    (tmp_path / missing).unlink()
    before = snapshot(tmp_path)

    with pytest.raises(PersistenceError, match="incomplete"):
        load_or_create_store(tmp_path, 256, store_cls=FAISSVectorStore)

    assert snapshot(tmp_path) == before  # no empty store was created over the survivor


def test_corrupt_metadata_is_an_error_and_files_are_untouched(tmp_path):
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    (tmp_path / "metadata.json").write_text("{ this is not json", encoding="utf-8")
    before = snapshot(tmp_path)

    with pytest.raises(PersistenceError, match="corrupt"):
        load_or_create_store(tmp_path, 256, store_cls=FAISSVectorStore)

    assert snapshot(tmp_path) == before


def test_corrupt_index_is_an_error_and_files_are_untouched(tmp_path):
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"not a faiss index")
    before = snapshot(tmp_path)

    with pytest.raises(PersistenceError, match="corrupt"):
        load_or_create_store(tmp_path, 256, store_cls=FAISSVectorStore)

    assert snapshot(tmp_path) == before


def test_index_and_metadata_that_disagree_are_an_error(tmp_path):
    small, big = tmp_path / "small", tmp_path / "big"
    make_store({"room_A": {"book.pdf": BOOK}}).save(small)
    make_store({"room_A": {"book.pdf": BOOK, "other.pdf": BOOK}}).save(big)
    shutil.copy2(big / "metadata.json", small / "metadata.json")  # 3 vectors, 6 records
    before = snapshot(small)

    with pytest.raises(PersistenceError, match="out of sync"):
        load_or_create_store(small, 256, store_cls=FAISSVectorStore)

    assert snapshot(small) == before


def test_a_path_that_is_a_file_is_an_error(tmp_path):
    target = tmp_path / "store"
    target.write_text("i am a file")
    with pytest.raises(PersistenceError, match="not a directory"):
        load_or_create_store(target, 256, store_cls=FAISSVectorStore)
    with pytest.raises(PersistenceError, match="not a directory"):
        save_store_safely(make_store({"room_A": {"book.pdf": BOOK}}), target, store_cls=FAISSVectorStore)
    assert target.read_text() == "i am a file"


def test_error_mentions_backups_left_by_an_interrupted_save(tmp_path):
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    shutil.copy2(tmp_path / "index.faiss", tmp_path / "index.faiss.bak")
    (tmp_path / "metadata.json").unlink()

    with pytest.raises(PersistenceError, match=r"index\.faiss\.bak"):
        load_existing_store(tmp_path, store_cls=FAISSVectorStore)
    assert [p.name for p in find_leftover_files(tmp_path)] == ["index.faiss.bak"]


def test_ensure_dimension_rejects_a_model_mismatch(tmp_path):
    store = FAISSVectorStore(8)
    ensure_dimension(store, 8, tmp_path)
    with pytest.raises(PersistenceError, match="dimension 8 .* 1024"):
        ensure_dimension(store, 1024, tmp_path)


# --------------------------------------------------------------- inspection

def test_document_exists_is_room_and_document_specific():
    store = make_store({"room_A": {"book.pdf": BOOK}, "room_B": {"other.pdf": BOOK}})
    assert document_exists(store, "room_A", "book.pdf") is True  # document_id defaults to the file name
    assert document_exists(store, "room_B", "book.pdf") is False
    assert document_exists(store, "room_A", "other.pdf") is False
    assert document_exists(store, "room_C", "book.pdf") is False


def test_document_exists_on_an_empty_store_is_false():
    assert document_exists(FAISSVectorStore(256), "room_A", "book.pdf") is False


def test_store_inventory_counts_chunks_per_room_and_document(tmp_path):
    make_store({"room_B": {"b.pdf": BOOK}, "room_A": {"a.pdf": BOOK, "z.pdf": BOOK[:1]}}).save(tmp_path)
    assert store_inventory(tmp_path) == {"room_A": {"a.pdf": 3, "z.pdf": 1}, "room_B": {"b.pdf": 3}}


def test_store_inventory_rejects_unreadable_metadata(tmp_path):
    (tmp_path / "metadata.json").write_text("nope", encoding="utf-8")
    with pytest.raises(PersistenceError, match="cannot read"):
        store_inventory(tmp_path)
    (tmp_path / "metadata.json").write_text('{"records": 5}', encoding="utf-8")
    with pytest.raises(PersistenceError, match="no 'records' list"):
        store_inventory(tmp_path)


# ------------------------------------------------------------------- saving

def test_save_creates_the_directory_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "deep" / "vector_store"
    save_store_safely(make_store({"room_A": {"book.pdf": BOOK}}), target, store_cls=FAISSVectorStore)
    assert sorted(p.name for p in target.iterdir()) == ["index.faiss", "metadata.json"]
    assert len(FAISSVectorStore.load(target)) == 3


def test_second_document_is_added_without_losing_the_first(tmp_path):
    embedder = HashingEmbedder()
    save_store_safely(make_store({"room_A": {"first.pdf": BOOK}}, embedder), tmp_path, store_cls=FAISSVectorStore)

    store, created = load_or_create_store(tmp_path, 256, store_cls=FAISSVectorStore)
    manager = TopicManager()
    pages = [{"page": 1, "text": "Transformers use positional encodings to represent token order."}]
    chunks = manager.tag_chunks("room_A", chunk_pages(pages, "room_A", "second.pdf"))
    store.add(embedder.embed_documents([c["text"] for c in chunks]), chunks)
    save_store_safely(store, tmp_path, store_cls=FAISSVectorStore)

    assert store_inventory(tmp_path) == {"room_A": {"first.pdf": 3, "second.pdf": 1}}
    assert len(FAISSVectorStore.load(tmp_path)) == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.faiss", "metadata.json"]  # no .bak left


def test_half_written_save_really_breaks_the_store_without_the_rollback(tmp_path):
    """Documents WHY save_store_safely exists: new index + old metadata cannot be loaded."""
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    grown = make_store({"room_A": {"book.pdf": BOOK, "more.pdf": BOOK}})
    with pytest.raises(OSError):
        IndexOnlySave(grown).save(tmp_path)
    with pytest.raises(VectorStoreLoadError, match="out of sync"):
        FAISSVectorStore.load(tmp_path)


def test_failed_save_restores_the_previous_store_byte_for_byte(tmp_path):
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    before = snapshot(tmp_path)
    grown = make_store({"room_A": {"book.pdf": BOOK, "more.pdf": BOOK}})

    with pytest.raises(PersistenceError, match="previous store files were restored"):
        save_store_safely(IndexOnlySave(grown), tmp_path, store_cls=FAISSVectorStore)

    assert snapshot(tmp_path) == before  # also proves no .bak / .tmp files remain
    assert len(FAISSVectorStore.load(tmp_path)) == 3


def test_save_that_writes_nothing_is_detected_by_the_reload_check(tmp_path):
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    before = snapshot(tmp_path)
    grown = make_store({"room_A": {"book.pdf": BOOK, "more.pdf": BOOK}})

    with pytest.raises(PersistenceError, match="verification failed: 6 vectors .* 3"):
        save_store_safely(SilentNoSave(grown), tmp_path, store_cls=FAISSVectorStore)

    assert snapshot(tmp_path) == before


def test_failed_first_save_leaves_nothing_behind(tmp_path):
    target = tmp_path / "vector_store"
    store = make_store({"room_A": {"book.pdf": BOOK}})

    with pytest.raises(PersistenceError, match="No store was written"):
        save_store_safely(IndexOnlySave(store), target, store_cls=FAISSVectorStore)
    assert not target.exists()

    with pytest.raises(PersistenceError, match="verification failed|failed"):
        save_store_safely(SilentNoSave(store), target, store_cls=FAISSVectorStore)
    assert not target.exists()


def test_failed_first_save_in_an_existing_directory_keeps_the_other_files(tmp_path):
    (tmp_path / "notes.txt").write_text("keep me")
    with pytest.raises(PersistenceError):
        save_store_safely(IndexOnlySave(make_store({"room_A": {"book.pdf": BOOK}})), tmp_path, store_cls=FAISSVectorStore)
    assert [p.name for p in tmp_path.iterdir()] == ["notes.txt"]


def test_interrupt_during_save_restores_the_files_and_still_interrupts(tmp_path):
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    before = snapshot(tmp_path)
    grown = make_store({"room_A": {"book.pdf": BOOK, "more.pdf": BOOK}})

    with pytest.raises(KeyboardInterrupt):
        save_store_safely(InterruptedSave(grown), tmp_path, store_cls=FAISSVectorStore)

    assert snapshot(tmp_path) == before


def test_save_refuses_to_touch_an_incomplete_store(tmp_path):
    make_store({"room_A": {"book.pdf": BOOK}}).save(tmp_path)
    (tmp_path / "metadata.json").unlink()
    before = snapshot(tmp_path)

    with pytest.raises(PersistenceError, match="refusing to save"):
        save_store_safely(make_store({"room_A": {"x.pdf": BOOK}}), tmp_path, store_cls=FAISSVectorStore)

    assert snapshot(tmp_path) == before


def test_duplicate_error_type_is_a_value_error():
    assert issubclass(DocumentAlreadyIngestedError, ValueError)
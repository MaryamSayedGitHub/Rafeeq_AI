"""scripts/ingest_book.py and scripts/query_book.py: argument parsing, orchestration, restart behavior.

No model is loaded: the deterministic HashingEmbedder stands in for BGE-M3 and the PDF parser is
replaced by a function returning page texts. The vector store is the REAL FAISSVectorStore working
in pytest's temporary directories. "Restart" is simulated by calling the query script's main() with a
brand-new embedder object, so everything it knows must come from the files on disk.
"""

import importlib.util
import io
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from kb_fakes import HashingEmbedder
from rafeeq.knowledge.vector_store import FAISSVectorStore

SCRIPTS = Path(__file__).resolve().parents[2] / "rafeeq" / "knowledge" / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest_book = load_script("ingest_book")
query_book = load_script("query_book")

BOOK_A = [
    "Self-attention allows each token to attend to other tokens.",
    "Convolutional neural networks are useful for image processing.",
    "Gradient descent is used to optimize neural networks.",
]
BOOK_B = [
    "Transformers use positional encodings to represent token order.",
    "Fine-tuning adapts a pretrained model with low rank adapters.",
]
PAGES = {"hands_on_llm.pdf": BOOK_A, "other_course.pdf": BOOK_B}


def parse_pages(path):
    """Stands in for parse_pdf: looks the page texts up by file name."""
    return [{"page": i + 1, "text": t} for i, t in enumerate(PAGES[Path(path).name])]


class Setup:
    def __init__(self, tmp_path):
        self.store_dir = tmp_path / "vector_store"
        self.pdf_dir = tmp_path / "pdfs"
        self.pdf_dir.mkdir()

    def pdf(self, name):
        path = self.pdf_dir / name
        path.write_bytes(b"%PDF-1.4 placeholder; the parser is replaced in these tests")
        return path

    def ingest(self, name="hands_on_llm.pdf", *extra, embedder=None, store_cls=FAISSVectorStore, parse=parse_pages):
        argv = [str(self.pdf(name)), "--store-dir", str(self.store_dir), *extra]
        out, err = io.StringIO(), io.StringIO()
        code = ingest_book.main(argv, embedder=embedder or HashingEmbedder(), store_cls=store_cls, parse=parse, out=out, err=err)
        return code, out.getvalue(), err.getvalue()

    def query(self, text, *extra, embedder=None):
        argv = [*([text] if text is not None else []), "--store-dir", str(self.store_dir), *extra]
        out, err = io.StringIO(), io.StringIO()
        code = query_book.main(argv, embedder=embedder or HashingEmbedder(), store_cls=FAISSVectorStore, out=out, err=err)
        return code, out.getvalue(), err.getvalue()

    def snapshot(self):
        if not self.store_dir.exists():
            return None
        return {p.name: p.read_bytes() for p in sorted(self.store_dir.iterdir())}


def number(report, label):
    return int(re.search(rf"{label}:\s+(\d+)", report).group(1))


class FailingEmbedder(HashingEmbedder):
    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        raise RuntimeError("model exploded")


class ShortEmbedder(HashingEmbedder):
    def embed_documents(self, texts):
        return super().embed_documents(texts)[:-1]


class ForbiddenEmbedder(HashingEmbedder):
    """Fails the test if anything tries to embed (or even ask for the dimension... via calls)."""

    def embed_documents(self, texts):
        raise AssertionError("the embedder must not be used here")

    def embed_query(self, text):
        raise AssertionError("the embedder must not be used here")


class DiskFullOnSave(FAISSVectorStore):
    def save(self, directory):
        raise OSError("disk full")


class HalfSave(FAISSVectorStore):
    """Writes the new index.faiss but fails before metadata.json: the dangerous half-save."""

    def save(self, directory):
        import shutil
        import tempfile

        Path(directory).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            super().save(tmp)
            shutil.copy2(Path(tmp) / "index.faiss", Path(directory) / "index.faiss")
        raise OSError("disk full while writing metadata.json")


# ------------------------------------------------------------ argument parsing

def test_ingest_defaults_point_at_the_book_and_a_stable_store_directory():
    args = ingest_book.build_parser().parse_args([])
    assert Path(args.pdf).parts[-3:] == ("data", "raw", "hands_on_llm.pdf")
    assert Path(args.store_dir).parts[-2:] == ("data", "vector_store")
    assert args.room_id == "llm_course" and args.document_id is None and args.device == "cpu"


def test_ingest_explicit_arguments_are_parsed():
    args = ingest_book.build_parser().parse_args(
        ["x.pdf", "--room-id", "r1", "--store-dir", "s", "--document-id", "d", "--document-type", "book",
         "--chunk-size", "100", "--overlap", "10", "--device", "cuda", "--batch-size", "4"]
    )
    assert (args.pdf, args.room_id, args.store_dir, args.document_id, args.document_type) == ("x.pdf", "r1", "s", "d", "book")
    assert (args.chunk_size, args.overlap, args.device, args.batch_size) == (100, 10, "cuda", 4)


@pytest.mark.parametrize("bad", [["--chunk-size", "0"], ["--chunk-size", "x"], ["--overlap", "-1"], ["--room-id", " "], ["--batch-size", "0"]])
def test_ingest_rejects_bad_arguments_with_usage_status(bad):
    with pytest.raises(SystemExit) as info:
        ingest_book.build_parser().parse_args(bad)
    assert info.value.code == 2


def test_query_defaults_and_arguments():
    parser = query_book.build_parser()
    args = parser.parse_args(["what is attention?"])
    assert args.room_id == "llm_course" and args.top_k == 6 and Path(args.store_dir).parts[-2:] == ("data", "vector_store")
    args = parser.parse_args(["q", "--room-id", "r", "-k", "3", "--max-chars", "50"])
    assert (args.room_id, args.top_k, args.max_chars) == ("r", 3, 50)


def test_query_requires_a_query_unless_listing(tmp_path):
    for argv in ([], ["  "]):
        with pytest.raises(SystemExit) as info:
            query_book.main(argv)
        assert info.value.code == 2
    with pytest.raises(SystemExit):
        query_book.build_parser().parse_args(["q", "-k", "0"])


# ------------------------------------------------------------------- ingesting

def test_ingest_saves_both_files_and_reports_accurate_statistics(tmp_path):
    env = Setup(tmp_path)
    embedder = HashingEmbedder()

    code, out, err = env.ingest(embedder=embedder)

    assert code == 0 and err == ""
    assert sorted(env.snapshot()) == ["index.faiss", "metadata.json"]
    for expected in ("hands_on_llm.pdf", "llm_course", str(env.store_dir), "topic_attention"):
        assert expected in out
    chunk_count = number(out, "Chunks")
    topic_total = sum(int(m) for m in re.findall(r"^\s{4}topic_\w+: (\d+)$", out, flags=re.M))
    assert number(out, "Pages") == 3
    assert chunk_count == number(out, "Vectors added") == number(out, "Vectors in store") == topic_total == 3
    assert [len(call) for call in embedder.document_calls] == [chunk_count]  # one embedding per chunk, one call
    reloaded = FAISSVectorStore.load(env.store_dir)
    assert len(reloaded) == chunk_count
    assert "hands_on_llm: 3 chunks" in out


def test_statistics_with_a_multi_chunk_page(tmp_path):
    env = Setup(tmp_path)
    long_page = " ".join(f"word{i}" for i in range(120))
    parse = lambda path: [{"page": 1, "text": long_page}, {"page": 2, "text": BOOK_A[0]}]  # noqa: E731

    code, out, _ = env.ingest("hands_on_llm.pdf", "--chunk-size", "50", "--overlap", "10", parse=parse)

    assert code == 0
    chunks = number(out, "Chunks")
    assert chunks > 3  # 120 words at 50/10 -> 3 chunks, plus the second page
    assert number(out, "Vectors added") == number(out, "Vectors in store") == chunks == len(FAISSVectorStore.load(env.store_dir))


def test_document_type_is_stored(tmp_path):
    env = Setup(tmp_path)
    assert env.ingest("hands_on_llm.pdf", "--document-type", "book")[0] == 0
    store = FAISSVectorStore.load(env.store_dir)
    hit = store.search(HashingEmbedder().embed_query("self-attention"), top_k=1)[0]
    assert hit.metadata["document_type"] == "book" and hit.metadata["document_id"] == "hands_on_llm"


def test_scanned_page_limitation_is_reported_not_hidden(tmp_path):
    env = Setup(tmp_path)
    parse = lambda path: [{"page": 1, "text": BOOK_A[0]}, {"page": 2, "text": "fig"}, {"page": 3, "text": BOOK_A[1]}]  # noqa: E731

    code, out, _ = env.ingest(parse=parse)

    assert code == 0
    assert "WARNING: 1 page(s) have no usable text layer" in out and "pages 2" in out
    assert "not installed yet" in out


def test_pdf_without_any_text_fails_and_stores_nothing(tmp_path):
    env = Setup(tmp_path)
    code, out, err = env.ingest(parse=lambda path: [{"page": 1, "text": ""}, {"page": 2, "text": "  "}])
    assert code == 1 and "Ingestion complete" not in out
    assert "no text could be extracted" in err and "scanned-page extractor" in err
    assert env.snapshot() is None


def test_missing_pdf_fails_before_touching_anything(tmp_path):
    env = Setup(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    code = ingest_book.main([str(tmp_path / "ghost.pdf"), "--store-dir", str(env.store_dir)],
                            embedder=ForbiddenEmbedder(), store_cls=FAISSVectorStore, parse=parse_pages, out=out, err=err)
    assert code == 1 and "PDF not found" in err.getvalue() and "ghost.pdf" in err.getvalue()
    assert env.snapshot() is None


# ----------------------------------------------- persistence / restart / adding

def test_a_second_pdf_is_added_without_losing_the_first(tmp_path):
    env = Setup(tmp_path)
    assert env.ingest("hands_on_llm.pdf")[0] == 0

    code, out, _ = env.ingest("other_course.pdf")

    assert code == 0
    assert "Loaded existing store" in out and "3 vectors" in out
    assert number(out, "Vectors added") == 2 and number(out, "Vectors in store") == 5
    assert "hands_on_llm: 3 chunks" in out and "other_course: 2 chunks" in out
    assert len(FAISSVectorStore.load(env.store_dir)) == 5
    assert sorted(env.snapshot()) == ["index.faiss", "metadata.json"]  # no backups left behind


def test_retrieval_after_restart_finds_the_right_page_for_english_and_arabic(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    env.ingest("other_course.pdf")

    for question in ("What does self-attention do?", "يعني إيه Self-Attention؟"):
        code, out, err = env.query(question, "-k", "3")  # a brand-new embedder and a store loaded from disk
        assert code == 0 and err == ""
        first = out.split("1. ", 1)[1].splitlines()[0]
        assert "file=hands_on_llm.pdf" in first and "page=1" in first and "topic=topic_attention" in first
        assert re.search(r"score=\d\.\d{4}", first)
        assert BOOK_A[0] in out  # the chunk text is shown


def test_query_is_read_only(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    before = env.snapshot()
    assert env.query("self-attention")[0] == 0
    assert env.snapshot() == before


def test_rooms_are_isolated_in_the_query_results(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf", "--room-id", "room_A")
    env.ingest("other_course.pdf", "--room-id", "room_B")

    _, out_a, _ = env.query("token", "--room-id", "room_A", "-k", "10")
    _, out_b, _ = env.query("token", "--room-id", "room_B", "-k", "10")

    assert "file=hands_on_llm.pdf" in out_a and "other_course.pdf" not in out_a
    assert "file=other_course.pdf" in out_b and "hands_on_llm.pdf" not in out_b


def test_the_same_pdf_can_be_ingested_into_a_different_room(tmp_path):
    env = Setup(tmp_path)
    assert env.ingest("hands_on_llm.pdf", "--room-id", "room_A")[0] == 0
    code, out, _ = env.ingest("hands_on_llm.pdf", "--room-id", "room_B")
    assert code == 0 and number(out, "Vectors in store") == 6


def test_unknown_room_reports_no_results_and_lists_the_rooms(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf", "--room-id", "room_A")
    code, out, err = env.query("self-attention", "--room-id", "room_Z")
    assert code == 1 and out == ""
    assert "no results for room 'room_Z'" in err and "room_A" in err


def test_query_k_limits_the_number_of_results(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    _, out, _ = env.query("neural networks", "-k", "1")
    assert "Top 1 result(s)" in out and "2. " not in out


def test_query_truncates_long_text(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    _, out, _ = env.query("self-attention", "-k", "1", "--max-chars", "20")
    preview = [line for line in out.splitlines() if line.startswith("   ")][0].strip()
    assert preview.endswith("...") and len(preview) == 20


def test_list_shows_rooms_and_documents_without_using_the_model(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf", "--room-id", "room_A")
    env.ingest("other_course.pdf", "--room-id", "room_B")
    code, out, _ = env.query(None, "--list", embedder=ForbiddenEmbedder())
    assert code == 0
    assert "5 vectors" in out and "room room_A" in out and "hands_on_llm: 3 chunks" in out and "other_course: 2 chunks" in out


def test_query_without_a_store_says_to_ingest_first(tmp_path):
    env = Setup(tmp_path)
    code, out, err = env.query("anything", embedder=ForbiddenEmbedder())
    assert code == 1 and "no vector store found" in err and "ingest_book.py" in err
    assert env.snapshot() is None


def test_query_on_an_empty_store_reports_it(tmp_path):
    env = Setup(tmp_path)
    FAISSVectorStore(256).save(env.store_dir)
    code, _, err = env.query("anything", embedder=ForbiddenEmbedder())
    assert code == 1 and "is empty (0 vectors)" in err


def test_query_with_a_different_embedding_model_is_refused(tmp_path):
    env = Setup(tmp_path)
    store = FAISSVectorStore(8)
    vector = np.zeros((1, 8), dtype=np.float32)
    vector[0, 0] = 1.0
    store.add(vector, [{"chunk_id": "c0", "room_id": "llm_course", "file_name": "x.pdf", "page": 1, "block_type": "text", "text": "hello"}])
    store.save(env.store_dir)
    code, _, err = env.query("hello")
    assert code == 1 and "dimension 8" in err and "256" in err


# ---------------------------------------------------------------- duplicates

def test_reingesting_the_same_document_is_refused_before_any_embedding(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    before = env.snapshot()
    embedder = HashingEmbedder()

    code, out, err = env.ingest("hands_on_llm.pdf", embedder=embedder)

    assert code == 1 and "Ingestion complete" not in out
    assert "already in room 'llm_course'" in err and "--document-id" in err and "append-only" in err
    assert embedder.document_calls == []  # refused cheaply: the slow embedding never started
    assert env.snapshot() == before


def test_a_new_document_id_keeps_both_versions(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    code, out, _ = env.ingest("hands_on_llm.pdf", "--document-id", "hands_on_llm_v2")
    assert code == 0 and number(out, "Vectors in store") == 6
    assert "hands_on_llm: 3 chunks" in out and "hands_on_llm_v2: 3 chunks" in out


def test_the_store_level_duplicate_rule_is_still_reported_clearly(tmp_path, monkeypatch):
    """If the cheap pre-check is bypassed, the store's own (room, document, chunk) rule must still stop it."""
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    before = env.snapshot()
    monkeypatch.setattr(ingest_book, "document_exists", lambda *args, **kwargs: False)

    code, out, err = env.ingest("hands_on_llm.pdf")

    assert code == 1 and "duplicate chunk" in err and "Nothing was stored" in err and "Ingestion complete" not in out
    assert env.snapshot() == before


# ------------------------------------------------------------------ failures

def test_embedding_failure_leaves_the_existing_store_untouched(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    before = env.snapshot()

    code, out, err = env.ingest("other_course.pdf", embedder=FailingEmbedder())

    assert code == 1 and "Ingestion complete" not in out
    assert "RuntimeError: model exploded" in err
    assert env.snapshot() == before
    assert len(FAISSVectorStore.load(env.store_dir)) == 3


def test_embedding_failure_on_a_first_run_creates_no_files(tmp_path):
    env = Setup(tmp_path)
    code, _, err = env.ingest(embedder=FailingEmbedder())
    assert code == 1 and "model exploded" in err
    assert env.snapshot() is None


def test_wrong_number_of_embeddings_is_refused_and_nothing_is_stored(tmp_path):
    env = Setup(tmp_path)
    code, _, err = env.ingest(embedder=ShortEmbedder())
    assert code == 1 and "refusing to index" in err
    assert env.snapshot() is None


def test_save_failure_is_reported_and_never_claimed_as_success(tmp_path):
    env = Setup(tmp_path)
    code, out, err = env.ingest(store_cls=DiskFullOnSave)
    assert code == 1 and "Ingestion complete" not in out
    assert "saving the vector store" in err and "disk full" in err
    assert env.snapshot() is None


def test_half_written_save_is_rolled_back_so_the_old_store_still_loads(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    before = env.snapshot()

    code, out, err = env.ingest("other_course.pdf", store_cls=HalfSave)

    assert code == 1 and "Ingestion complete" not in out
    assert "previous store files were restored" in err
    assert env.snapshot() == before
    assert len(FAISSVectorStore.load(env.store_dir)) == 3
    assert env.ingest("other_course.pdf")[0] == 0  # and the next normal run works


def test_corrupt_store_is_reported_and_never_overwritten(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    (env.store_dir / "metadata.json").write_text("{ broken", encoding="utf-8")
    before = snapshot = env.snapshot()
    embedder = HashingEmbedder()

    code, out, err = env.ingest("other_course.pdf", embedder=embedder)

    assert code == 1 and "corrupt" in err and "no empty store was created" in err
    assert embedder.document_calls == [] and env.snapshot() == before


@pytest.mark.parametrize("missing", ["index.faiss", "metadata.json"])
def test_incomplete_store_is_reported_and_never_overwritten(tmp_path, missing):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    (env.store_dir / missing).unlink()
    before = env.snapshot()

    code, _, err = env.ingest("other_course.pdf")

    assert code == 1 and "incomplete" in err and missing in err
    assert env.snapshot() == before


def test_ingest_into_a_store_built_with_another_model_is_refused(tmp_path):
    env = Setup(tmp_path)
    store = FAISSVectorStore(8)
    vector = np.zeros((1, 8), dtype=np.float32)
    vector[0, 0] = 1.0
    store.add(vector, [{"chunk_id": "c0", "room_id": "r", "file_name": "x.pdf", "page": 1, "block_type": "text", "text": "hello"}])
    store.save(env.store_dir)
    before = env.snapshot()

    code, _, err = env.ingest()

    assert code == 1 and "dimension 8" in err
    assert env.snapshot() == before


def test_debug_flag_prints_the_traceback(tmp_path):
    env = Setup(tmp_path)
    code, _, err = env.ingest("hands_on_llm.pdf", "--debug", embedder=FailingEmbedder())
    assert code == 1 and "Traceback" in err and "model exploded" in err


def test_leftover_files_from_an_interrupted_save_produce_a_warning(tmp_path):
    env = Setup(tmp_path)
    env.ingest("hands_on_llm.pdf")
    (env.store_dir / "index.faiss.bak").write_bytes(b"old")
    code, out, _ = env.ingest("other_course.pdf")
    assert code == 0 and "warning: files from an interrupted save" in out
"""Integration: scripts/ingest_book.py and scripts/query_book.py as REAL separate processes,
with the REAL BGE-M3 model, REAL PyMuPDF parsing and the REAL FAISS files on disk.

This is the "restart" proof: every command below is a brand-new Python process, so the query can
only succeed if index.faiss + metadata.json were really written and can be loaded again.

Slow (each process that embeds loads the model). Run separately:
    uv run pytest tests/knowledge/integration/test_ingest_query_bge_m3.py -v -s
Only structure and rankings are asserted, never score thresholds.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

ROOT = Path(__file__).resolve().parents[3]
INGEST = ROOT / "rafeeq" / "knowledge" / "scripts" / "ingest_book.py"
QUERY = ROOT / "rafeeq" / "knowledge" / "scripts" / "query_book.py"

COURSE_A = [
    "Self-attention allows each token to attend to other tokens.",
    "Convolutional neural networks are useful for image processing.",
    "Gradient descent is used to optimize neural networks.",
]
COURSE_B = [
    "Transformers use positional encodings to represent token order.",
    "Fine-tuning adapts a pretrained model with low rank adapters.",
]
EN_QUERY = "What does self-attention do?"
AR_QUERY = "يعني إيه Self-Attention؟"


def show(text: str) -> None:
    enc = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc))


def run(script: Path, *args) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, str(script), *map(str, args)],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=3600,
    )


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    base = tmp_path_factory.mktemp("kb_e2e")
    pdf_a = make_pdf(base / "hands_on_llm.pdf", COURSE_A)
    pdf_b = make_pdf(base / "other_course.pdf", COURSE_B)
    store = base / "vector_store"

    first = run(INGEST, pdf_a, "--room-id", "room_A", "--store-dir", store)
    show(first.stdout + first.stderr)
    second = run(INGEST, pdf_b, "--room-id", "room_B", "--store-dir", store)  # a new process adds to the saved store
    show(second.stdout + second.stderr)
    return {"pdf_a": pdf_a, "pdf_b": pdf_b, "store": store, "first": first, "second": second}


def test_first_ingest_persists_both_files(workspace):
    first = workspace["first"]
    assert first.returncode == 0, first.stderr
    assert "No saved store" in first.stdout and "Vectors added:     3" in first.stdout
    assert sorted(p.name for p in workspace["store"].iterdir()) == ["index.faiss", "metadata.json"]


def test_second_pdf_is_added_to_the_saved_store_without_losing_the_first(workspace):
    second = workspace["second"]
    assert second.returncode == 0, second.stderr
    assert "Loaded existing store" in second.stdout and "3 vectors" in second.stdout
    assert "Vectors in store:  5" in second.stdout
    assert "hands_on_llm: 3 chunks" in second.stdout and "other_course: 2 chunks" in second.stdout


@pytest.mark.parametrize("query", [EN_QUERY, AR_QUERY])
def test_a_new_process_retrieves_the_right_page(workspace, query):
    result = run(QUERY, query, "--room-id", "room_A", "-k", "3", "--store-dir", workspace["store"])
    show(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    first = result.stdout.split("1. ", 1)[1].splitlines()[0]
    assert "file=hands_on_llm.pdf" in first and "page=1" in first and "topic=topic_attention" in first


def test_rooms_stay_isolated_across_processes(workspace):
    in_b = run(QUERY, AR_QUERY, "--room-id", "room_B", "-k", "10", "--store-dir", workspace["store"])
    assert in_b.returncode == 0 and "other_course.pdf" in in_b.stdout and "hands_on_llm.pdf" not in in_b.stdout
    nowhere = run(QUERY, AR_QUERY, "--room-id", "room_Z", "--store-dir", workspace["store"])
    assert nowhere.returncode == 1 and "no results for room 'room_Z'" in nowhere.stderr


def test_reingesting_is_refused_and_leaves_the_files_untouched(workspace):
    before = snapshot(workspace["store"])
    again = run(INGEST, workspace["pdf_a"], "--room-id", "room_A", "--store-dir", workspace["store"])
    assert again.returncode == 1 and "already in room 'room_A'" in again.stderr
    assert snapshot(workspace["store"]) == before
"""run_eval.py --exclude-non-content: the A/B switch changes retrieval only, and says so in the results."""

import io
import json
from pathlib import Path

from eval_helpers import load_script, save_store
from kb_fakes import HashingEmbedder
from rafeeq.knowledge.evaluation.dataset import BenchmarkItem, dump_benchmark
from test_content_filter import index_page

run_eval = load_script("run_eval")
validate_benchmark = load_script("validate_benchmark")

ROOM = "llm_course"
QUESTION = "What is self-attention?"
CONTENT = "Self-attention allows each token to attend to other tokens in the sequence."
PAGES = [
    "Convolutional neural networks are useful for image processing.",
    index_page([("self-attention", "Self Attention Layers, Self Attention Heads")] * 12),   # index-like, repeats the question's words
    CONTENT,
    "Gradient descent is used to optimize neural networks.",
]


def snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(Path(directory).iterdir())}


def setup(tmp_path):
    store_dir = tmp_path / "store"
    save_store(store_dir, {ROOM: {"hands_on_llm.pdf": PAGES}})
    benchmark = tmp_path / "bench.jsonl"
    dump_benchmark([BenchmarkItem(
        id="q1", language="en", qtype="definition", question=QUESTION, answerable=True, file_name="hands_on_llm.pdf",
        evidence_quotes=[CONTENT], reference_answer="Each token attends to the others.", key_facts=[["attend"]])], benchmark)
    code = validate_benchmark.main(["--benchmark", str(benchmark), "--store-dir", str(store_dir), "--write"], out=io.StringIO(), err=io.StringIO())
    assert code == 0
    return store_dir, benchmark


def run(tmp_path, store_dir, benchmark, name, *extra):
    out_dir = tmp_path / name
    code = run_eval.main(["--benchmark", str(benchmark), "--store-dir", str(store_dir), "--room-id", ROOM, "--out-dir", str(out_dir), *extra],
                         embedder=HashingEmbedder(), out=io.StringIO(), err=io.StringIO())
    assert code == 0
    return json.loads((out_dir / "results.json").read_text(encoding="utf-8")), out_dir


def test_the_switch_moves_the_content_page_up_and_is_recorded(tmp_path):
    store_dir, benchmark = setup(tmp_path)
    before_files = snapshot(store_dir)

    base, _ = run(tmp_path, store_dir, benchmark, "base")
    variant, variant_dir = run(tmp_path, store_dir, benchmark, "variant", "--exclude-non-content")

    base_row, variant_row = base["per_question"][0], variant["per_question"][0]
    assert [h["page"] for h in base_row["retrieved"]][:2] == [2, 3]            # the index-like page outranks the content page
    assert base_row["first_relevant_rank"] == 2 and base_row["hit@1"] == 0.0
    assert [h["page"] for h in variant_row["retrieved"]][0] == 3
    assert variant_row["first_relevant_rank"] == 1 and variant_row["hit@1"] == 1.0
    assert 2 not in [h["page"] for h in variant_row["retrieved"]]
    assert base["meta"]["retrieval"] == "dense (unchanged baseline behaviour)"
    assert variant["meta"]["retrieval"] == "dense, index-like pages excluded"
    assert "dense, index-like pages excluded" in (variant_dir / "report.md").read_text(encoding="utf-8")
    assert snapshot(store_dir) == before_files                                  # the store is only read


def test_without_the_flag_nothing_changes_relative_to_the_plain_retriever(tmp_path):
    store_dir, benchmark = setup(tmp_path)
    first, _ = run(tmp_path, store_dir, benchmark, "one")
    second, _ = run(tmp_path, store_dir, benchmark, "two")
    assert first["per_question"][0]["retrieved"] == second["per_question"][0]["retrieved"]
    assert "exclude" not in first["meta"]["retrieval"]
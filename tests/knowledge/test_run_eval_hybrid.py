"""run_eval.py --retrieval-mode: dense stays the default and unchanged; hybrid is recorded and auditable."""

import io
import json

import pytest

from eval_helpers import load_script, save_store
from kb_fakes import HashingEmbedder
from rafeeq.knowledge.evaluation.dataset import BenchmarkItem, dump_benchmark

run_eval = load_script("run_eval")
validate_benchmark = load_script("validate_benchmark")

ROOM = "llm_course"
CONTENT = "Self-attention allows each token to attend to other tokens in the sequence."
PAGES = [
    "Convolutional neural networks are useful for image processing.",
    CONTENT,
    "Gradient descent is used to optimize neural networks.",
    "Retrieval-augmented generation fetches documents before answering a question.",
]


def setup(tmp_path):
    store_dir = tmp_path / "store"
    save_store(store_dir, {ROOM: {"hands_on_llm.pdf": PAGES}})
    benchmark = tmp_path / "bench.jsonl"
    dump_benchmark([
        BenchmarkItem(id="q1", language="en", qtype="definition", question="What is self-attention?", answerable=True, file_name="hands_on_llm.pdf",
                      evidence_quotes=[CONTENT], reference_answer="Each token attends to the others.", key_facts=[["attend"]]),
        BenchmarkItem(id="o1", language="en", qtype="out_of_scope", question="Which sequence won the football world cup?", answerable=False, absent_terms=["zebra"]),
    ], benchmark)
    code = validate_benchmark.main(["--benchmark", str(benchmark), "--store-dir", str(store_dir), "--write"], out=io.StringIO(), err=io.StringIO())
    assert code == 0
    return store_dir, benchmark


def run(tmp_path, store_dir, benchmark, name, *extra):
    out_dir = tmp_path / name
    out, err = io.StringIO(), io.StringIO()
    code = run_eval.main(["--benchmark", str(benchmark), "--store-dir", str(store_dir), "--room-id", ROOM, "--out-dir", str(out_dir), *extra],
                         embedder=HashingEmbedder(), out=out, err=err)
    results = json.loads((out_dir / "results.json").read_text(encoding="utf-8")) if code == 0 else None
    return code, results, err.getvalue(), out_dir


def row(results, qid):
    return next(r for r in results["per_question"] if r["id"] == qid)


def test_dense_is_the_default_and_its_records_have_no_hybrid_fields(tmp_path):
    store_dir, benchmark = setup(tmp_path)
    code, results, _, _ = run(tmp_path, store_dir, benchmark, "default")
    assert code == 0
    assert results["meta"]["retrieval"] == "dense (unchanged baseline behaviour)"
    assert all(set(h) == {"rank", "chunk_id", "file_name", "page", "topic_id", "score", "text_preview", "relevant_page", "relevant_chunk"}
               for h in row(results, "q1")["retrieved"])
    explicit = run(tmp_path, store_dir, benchmark, "explicit", "--retrieval-mode", "dense")[1]
    assert explicit["per_question"] == results["per_question"]                 # naming the default changes nothing


def test_hybrid_is_labelled_and_keeps_the_fusion_audit_fields_next_to_the_true_cosine(tmp_path):
    store_dir, benchmark = setup(tmp_path)
    code, results, _, out_dir = run(tmp_path, store_dir, benchmark, "hybrid", "--retrieval-mode", "hybrid")
    assert code == 0
    assert results["meta"]["retrieval"] == "hybrid (dense + BM25, RRF k=60, equal weights, candidate pool 50)"
    hits = row(results, "q1")["retrieved"]
    assert hits[0]["page"] == 2 and row(results, "q1")["first_relevant_rank"] == 1
    for h in hits:
        assert {"fusion_score", "dense_rank", "lexical_rank", "lexical_score"} <= set(h)
        assert -1.0 <= h["score"] <= 1.0 and h["fusion_score"] < 0.05           # cosine and fusion value are different quantities
    assert hits[0]["lexical_rank"] == 1 and hits[0]["dense_rank"] == 1
    # the abstention gate and its sweep still use the cosine of the top result, not the fusion value
    o1 = row(results, "o1")
    cosines = [h["score"] for h in o1["retrieved"]]
    assert o1["top1_score"] == cosines[0]                                          # the cosine of the top-ranked chunk ...
    assert o1["max_score"] == max(cosines) > o1["top1_score"] > 0.1                # ... which fusion can place below a higher-cosine chunk
    assert o1["gate_passed"] is False                                              # 0.1667 < the default min_score 0.35
    sweep = {row_["threshold"]: row_ for row_ in results["threshold_sweep"]}
    assert sweep[0.15]["out_of_scope_pass_rate"] == 1.0 and sweep[0.2]["out_of_scope_pass_rate"] == 0.0   # uses max_score (0.1667), as the Answerer's gate does
    assert "hybrid (dense + BM25" in (out_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("extra, label", [
    ([], "dense (unchanged baseline behaviour)"),
    (["--exclude-non-content"], "dense, index-like pages excluded"),
    (["--retrieval-mode", "hybrid"], "hybrid (dense + BM25, RRF k=60, equal weights, candidate pool 50)"),
    (["--retrieval-mode", "hybrid", "--exclude-non-content"], "hybrid (dense + BM25, RRF k=60, equal weights, candidate pool 50), index-like pages excluded"),
])
def test_the_four_modes_are_distinguishable_in_the_run_metadata(tmp_path, extra, label):
    store_dir, benchmark = setup(tmp_path)
    code, results, _, _ = run(tmp_path, store_dir, benchmark, "m", *extra)
    assert code == 0 and results["meta"]["retrieval"] == label


def test_fusion_parameters_are_accepted_for_hybrid_and_refused_for_dense(tmp_path):
    store_dir, benchmark = setup(tmp_path)
    code, results, _, _ = run(tmp_path, store_dir, benchmark, "p", "--retrieval-mode", "hybrid", "--rrf-k", "30", "--candidate-pool", "10")
    assert code == 0 and "RRF k=30" in results["meta"]["retrieval"] and "candidate pool 10" in results["meta"]["retrieval"]
    code, _, err, out_dir = run(tmp_path, store_dir, benchmark, "q", "--rrf-k", "30")
    assert code == 1 and "only apply with --retrieval-mode hybrid" in err and not out_dir.exists()


def test_existing_results_are_never_overwritten(tmp_path):
    store_dir, benchmark = setup(tmp_path)
    code, first, _, out_dir = run(tmp_path, store_dir, benchmark, "keep")
    assert code == 0
    before = (out_dir / "results.json").read_bytes()
    code, _, err, _ = run(tmp_path, store_dir, benchmark, "keep", "--retrieval-mode", "hybrid")
    assert code == 1 and "already exists" in err and "never overwritten" in err
    assert (out_dir / "results.json").read_bytes() == before


def test_the_store_is_only_read(tmp_path):
    store_dir, benchmark = setup(tmp_path)
    snapshot = {p.name: p.read_bytes() for p in store_dir.iterdir()}
    assert run(tmp_path, store_dir, benchmark, "h", "--retrieval-mode", "hybrid")[0] == 0
    assert {p.name: p.read_bytes() for p in store_dir.iterdir()} == snapshot

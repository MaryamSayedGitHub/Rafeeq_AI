"""scripts/heldout_report.py: only real, like-for-like runs on the frozen held-out benchmark can be compared."""

import copy
import io
import json

import pytest

from eval_helpers import load_script
from rafeeq.knowledge.evaluation.dataset import BenchmarkItem
from rafeeq.knowledge.evaluation.runner import aggregate_retrieval, score_retrieval

report = load_script("heldout_report")
KS = [1, 3, 5]
PREFIX = "abcdef012345"


def item(id, pages, qtype="definition"):
    return BenchmarkItem(id=id, language="en", qtype=qtype, question=f"question {id}", answerable=True, status="verified",
                         file_name="book.pdf", expected_pages=list(pages), evidence_quotes=["x" * 30], reference_answer="r", key_facts=[["f"]])


def oos(id):
    return BenchmarkItem(id=id, language="en", qtype="out_of_scope", question=f"off {id}", answerable=False, status="verified", absent_terms=["zebra"])


def hits(*pages, top=0.9):
    return [{"file_name": "book.pdf", "page": p, "score": top - 0.1 * i, "chunk_id": f"book.pdf-p{p}", "text": f"t{p}", "topic_id": "t"} for i, p in enumerate(pages)]


ITEMS = [item("s1", [10]), item("s2", [20]), item("m1", [30, 31], "multi_passage"), oos("x1")]
RETR = {
    "dense": {"s1": hits(10, 1, 2, 3, 4), "s2": hits(1, 20, 2, 3, 4), "m1": hits(30, 1, 31, 2, 3), "x1": hits(9, 9, 9, 9, 9, top=0.3)},
    "hybrid": {"s1": hits(1, 2, 10, 3, 4), "s2": hits(20, 1, 2, 3, 4), "m1": hits(30, 31, 1, 2, 3), "x1": hits(9, 9, 9, 9, 9, top=0.5)},
}


def run(config, retrieval=None):
    mode = "hybrid" if config.startswith("hybrid") else "dense"
    records = [score_retrieval(it, RETR[mode][it.id], KS, 0.35) for it in ITEMS]
    if retrieval is None:
        retrieval = "hybrid (dense + BM25, RRF k=60, equal weights, candidate pool 50)" if mode == "hybrid" else "dense (unchanged baseline behaviour)"
    meta = {"benchmark_file": f"eval/benchmark/hands_on_llm_heldout_v1.jsonl (sha256 {PREFIX})", "retrieval": retrieval, "embedder": "BAAI/bge-m3 on cpu",
            "store": "s", "room": "r", "ks": "1 3 5", "documents in room": "book.pdf"}
    return {"meta": meta, "retrieval": aggregate_retrieval(records, KS), "per_question": records}


def runs():
    docs = {c: run(c) for c in report.CONFIGS}
    docs["dense_exclude"]["meta"]["retrieval"] = "dense, index-like pages excluded"
    docs["hybrid_exclude"]["meta"]["retrieval"] = "hybrid (dense + BM25, RRF k=60, equal weights, candidate pool 50), index-like pages excluded"
    return docs


def write(tmp_path, docs):
    paths = {}
    for cfg, doc in docs.items():
        p = tmp_path / f"{cfg}.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        paths[cfg] = str(p)
    return paths


def freeze(tmp_path):
    f = tmp_path / "freeze.sha256"
    f.write_text(f"{PREFIX}0000000000000000000000000000000000000000000000000000  hands_on_llm_heldout_v1.jsonl\n", encoding="utf-8")
    return str(f)


def argv(paths, tmp_path, *extra):
    out = []
    for cfg, p in paths.items():
        out += ["--" + cfg.replace("_", "-"), p]
    return out + ["--freeze-file", freeze(tmp_path), *extra]


def test_sign_test_values():
    assert report.sign_test_p(0, 0) == 1.0
    assert report.sign_test_p(5, 0) == pytest.approx(0.0625)
    assert report.sign_test_p(3, 1) == pytest.approx(0.625)
    assert report.sign_test_p(4, 4) == 1.0


def test_a_good_set_of_runs_produces_a_verified_report(tmp_path):
    buf = io.StringIO()
    assert report.main(argv(write(tmp_path, runs()), tmp_path), out=buf) == 0
    text = buf.getvalue()
    assert "VERIFIED" in text and "REPLAY" not in text.replace("not a replay", "")
    assert "Multi-passage questions" in text and "Full Recall@5" in text
    assert "Dense  ->  Hybrid" in text and "Dense + exclusion  ->  Hybrid + exclusion" in text
    # s1 got worse (rank 1 -> 3), s2 got better (2 -> 1): one regression, one improvement
    assert "improved: **1**, regressed: **1**" in text


def test_subset_metrics_use_the_per_question_rows(tmp_path):
    docs = runs()
    multi = report.subset_metrics([r for r in docs["dense"]["per_question"] if r["qtype"] == "multi_passage"])
    assert multi["full_recall@5"] == 1.0 and multi["hit@1"] == 1.0 and multi["mrr@5"] == 1.0
    single = report.subset_metrics([r for r in docs["hybrid"]["per_question"] if r["answerable"] and r["qtype"] != "multi_passage"])
    assert single["hit@1"] == 0.5 and single["hit@3"] == 1.0


@pytest.mark.parametrize("breakage", ["wrong_benchmark", "wrong_hash", "replay", "wrong_mode", "wrong_exclusion", "wrong_pool", "other_embedder", "other_questions"])
def test_incomparable_inputs_are_refused(tmp_path, breakage):
    docs = runs()
    d = docs["hybrid"]
    if breakage == "wrong_benchmark":
        d["meta"]["benchmark_file"] = "eval/benchmark/hands_on_llm_v1.jsonl (sha256 c872d23dcc16)"
    elif breakage == "wrong_hash":
        d["meta"]["benchmark_file"] = "eval/benchmark/hands_on_llm_heldout_v1.jsonl (sha256 000000000000)"
    elif breakage == "replay":
        d["meta"]["retrieval"] = "hybrid REPLAY (offline)"
    elif breakage == "wrong_mode":
        docs["dense"]["meta"]["retrieval"] = d["meta"]["retrieval"]
    elif breakage == "wrong_exclusion":
        d["meta"]["retrieval"] += ", index-like pages excluded"
    elif breakage == "wrong_pool":
        d["meta"]["retrieval"] = "hybrid (dense + BM25, RRF k=60, equal weights, candidate pool 20)"
    elif breakage == "other_embedder":
        d["meta"]["embedder"] = "all-MiniLM"
    elif breakage == "other_questions":
        d["per_question"] = d["per_question"][:-1]
    err = io.StringIO()
    old = report.sys.stderr
    report.sys.stderr = err
    try:
        assert report.main(argv(write(tmp_path, docs), tmp_path), out=io.StringIO()) == 2
    finally:
        report.sys.stderr = old
    assert "refusing" in err.getvalue()


def test_an_existing_report_file_is_never_overwritten(tmp_path):
    target = tmp_path / "report.md"
    target.write_text("precious", encoding="utf-8")
    err = io.StringIO()
    old = report.sys.stderr
    report.sys.stderr = err
    try:
        assert report.main(argv(write(tmp_path, runs()), tmp_path, "--out", str(target)), out=io.StringIO()) == 2
    finally:
        report.sys.stderr = old
    assert target.read_text(encoding="utf-8") == "precious"

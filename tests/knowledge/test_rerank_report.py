"""scripts/rerank_report.py on SYNTHETIC result dicts (shape of run_eval results.json); no real run is involved."""
import pytest

from eval_helpers import load_script

rr = load_script("rerank_report")
BENCH = "eval/benchmark/hands_on_llm_heldout_v1.jsonl (sha256 6b0e57f23ff7)"


def run(label, ranks):
    overall = {"n": len(ranks), "hit@1": 0.5, "hit@3": 0.5, "hit@5": 1.0, "mrr@5": 0.6}
    per = [{"id": i, "answerable": True, "first_relevant_rank": r} for i, r in ranks.items()]
    return {"meta": {"benchmark_file": BENCH, "store": "s", "room": "r", "embedder": "BAAI/bge-m3 on cuda", "ks": "1 3 5", "retrieval": label},
            "retrieval": {"overall": overall}, "per_question": per}


H = "hybrid (dense + BM25, RRF k=60, equal weights, candidate pool 50)"


def runs(**over):
    base = {"dense": run("dense (unchanged baseline behaviour)", {"h02": 4, "h07": None}), "hybrid": run(H, {"h02": 3, "h07": 5}),
            "hybrid_rerank": run(H + " + Cross-Encoder rerank (BAAI/bge-reranker-v2-m3, pool 50)", {"h02": 1, "h07": 5})}
    base.update(over)
    return base


def test_render_reports_focus_questions_and_direction():
    text = rr.render(runs())
    assert "h02 **(focus)** | 4 | 3 | 1 | improved" in text and "h07 **(focus)** | - | 5 | 5 | same" in text
    assert "Not RAGAS" in text


def test_check_accepts_matching_runs():
    rr.check(runs())


def test_check_rejects_reranked_baseline_and_unreranked_candidate_and_other_benchmark():
    with pytest.raises(SystemExit):
        rr.check(runs(hybrid=runs()["hybrid_rerank"]))
    with pytest.raises(SystemExit):
        rr.check(runs(hybrid_rerank=run(H, {"h02": 1})))
    bad = runs(); bad["dense"]["meta"]["benchmark_file"] = "x/hands_on_llm_v1.jsonl (sha256 55d970cf3a23)"
    with pytest.raises(SystemExit):
        rr.check(bad)


def test_refuses_existing_output(tmp_path):
    out = tmp_path / "o.md"; out.write_text("keep")
    assert rr.main(["--dense", "a", "--hybrid", "b", "--hybrid-rerank", "c", "--out", str(out)]) == 1 and out.read_text() == "keep"

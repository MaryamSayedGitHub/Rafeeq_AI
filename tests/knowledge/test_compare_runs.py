"""scripts/compare_runs.py on small hand-built results documents (numbers worked out by hand)."""

import io
import json

import pytest

from eval_helpers import load_script

compare_runs = load_script("compare_runs")


def row(qid, rank, pages, *, answerable=True, qtype="definition", language="en", top1=None):
    return {"id": qid, "answerable": answerable, "qtype": qtype, "language": language, "first_relevant_rank": rank,
            "top1_score": top1, "retrieved": [{"page": p} for p in pages]}


def doc(rows, overall, multi, *, retrieval="dense (unchanged baseline behaviour)", ks="1 3 5", sweep=None, benchmark="b.jsonl (sha256 aaa)"):
    return {
        "meta": {"retrieval": retrieval, "run_at_utc": "2026-10-09 12:00:00", "ks": ks, "benchmark_file": benchmark,
                 "store": "s (10 vectors)", "embedder": "bge-m3"},
        "retrieval": {"overall": overall, "by_qtype": {"multi_passage": multi}, "by_language": {}},
        "per_question": rows,
        "threshold_sweep": sweep or [],
    }


# ranks before: q1=4, q2=1, q3=1, q4=none ; after: q1=2, q2=none, q3=1, q4=none
BEFORE = doc(
    [row("q1", 4, [9, 8, 7, 1]), row("q2", 1, [2, 5]), row("q3", 1, [3]), row("q4", None, [6]),
     row("o1", None, [40], answerable=False, qtype="out_of_scope", top1=0.66), row("o2", None, [41], answerable=False, qtype="out_of_scope", top1=0.45)],
    {"n": 4, "hit@1": 0.5, "hit@1_ci95": [0.1, 0.9], "hit@3": 0.5, "hit@5": 0.75, "mrr@5": 0.6},
    {"n": 2, "recall@5": 0.5, "full_recall@5": 0.0},
    sweep=[{"threshold": 0.5, "answerable_pass_rate": 1.0, "out_of_scope_pass_rate": 0.5},
           {"threshold": 0.7, "answerable_pass_rate": 0.25, "out_of_scope_pass_rate": 0.0}],
)
AFTER = doc(
    [row("q1", 2, [9, 1]), row("q2", None, [5, 6]), row("q3", 1, [3]), row("q4", None, [7]),
     row("o1", None, [42], answerable=False, qtype="out_of_scope", top1=0.58), row("o2", None, [41], answerable=False, qtype="out_of_scope", top1=0.45)],
    {"n": 4, "hit@1": 0.25, "hit@1_ci95": [0.0, 0.7], "hit@3": 0.5, "hit@5": 0.5, "mrr@5": 0.35},
    {"n": 2, "recall@5": 0.75, "full_recall@5": 0.5},
    retrieval="dense, index-like pages excluded",
    sweep=[{"threshold": 0.5, "answerable_pass_rate": 1.0, "out_of_scope_pass_rate": 0.0},
           {"threshold": 0.7, "answerable_pass_rate": 0.25, "out_of_scope_pass_rate": 0.0}],
)


def test_metric_deltas_are_after_minus_before():
    cmp = compare_runs.compare(BEFORE, AFTER)
    overall = cmp["metrics"]["all answerable"]
    assert overall["n"] == 4
    assert overall["hit@1"] == {"before": 0.5, "after": 0.25, "delta": pytest.approx(-0.25)}
    assert overall["hit@3"]["delta"] == pytest.approx(0.0)
    assert overall["hit@5"]["delta"] == pytest.approx(-0.25)
    assert overall["mrr@5"]["delta"] == pytest.approx(-0.25)
    assert "hit@1_ci95" not in overall                                # interval lists are not metrics
    multi = cmp["metrics"]["type: multi_passage"]
    assert multi["recall@5"]["delta"] == pytest.approx(0.25) and multi["full_recall@5"]["delta"] == pytest.approx(0.5)


def test_questions_are_sorted_into_improved_regressed_and_unchanged():
    ans = compare_runs.compare(BEFORE, AFTER)["answerable"]
    assert [(r["id"], r["rank_before"], r["rank_after"]) for r in ans["improved"]] == [("q1", "4", "2")]
    assert [(r["id"], r["rank_before"], r["rank_after"]) for r in ans["regressed"]] == [("q2", "1", "none")]
    assert ans["unchanged"] == 2                                       # q3 stayed at 1, q4 stayed a miss
    assert ans["improved"][0]["pages_before"] == [9, 8, 7, 1] and ans["improved"][0]["pages_after"] == [9, 1]


def test_out_of_scope_top1_scores_are_compared_and_never_counted_as_answerable():
    cmp = compare_runs.compare(BEFORE, AFTER)
    by_id = {r["id"]: r for r in cmp["out_of_scope"]}
    assert set(by_id) == {"o1", "o2"}
    assert by_id["o1"]["delta"] == pytest.approx(-0.08) and by_id["o1"]["top1_page_before"] == 40 and by_id["o1"]["top1_page_after"] == 42
    assert by_id["o2"]["delta"] == pytest.approx(0.0)
    assert cmp["threshold_sweep"][0] == {"threshold": 0.5, "answerable_before": 1.0, "answerable_after": 1.0,
                                         "out_of_scope_before": 0.5, "out_of_scope_after": 0.0}


def test_identical_runs_show_no_change():
    cmp = compare_runs.compare(BEFORE, BEFORE)
    assert cmp["warnings"] == []
    assert cmp["answerable"]["improved"] == [] and cmp["answerable"]["regressed"] == []
    assert all(m["delta"] == 0 for g in cmp["metrics"].values() for k, m in g.items() if k != "n")


def test_mismatched_runs_produce_warnings_and_ignore_unshared_questions():
    other = json.loads(json.dumps(AFTER))
    other["meta"].update(ks="1 3 5 10", benchmark_file="c.jsonl (sha256 bbb)", embedder="other")
    other["per_question"].append(row("q9", 1, [1]))
    warnings = " | ".join(compare_runs.compare(BEFORE, other)["warnings"])
    assert "different benchmark files" in warnings and "cut-offs differ" in warnings and "different embedders" in warnings
    assert "only one run (ignored): q9" in warnings


def test_report_text_states_the_numbers_and_the_caveat():
    text = compare_runs.render(compare_runs.compare(BEFORE, AFTER))
    assert "- before: dense (unchanged baseline behaviour)" in text and "- after: dense, index-like pages excluded" in text
    assert "| hit@1 | 50.0% | 25.0% | -25.0 pp |" in text
    assert "| mrr@5 | 0.600 | 0.350 | -0.250 |" in text
    assert "| hit@3 | 50.0% | 50.0% | 0 |" in text
    assert "Improved: **1**, regressed: **1**, unchanged rank: 2" in text
    assert "| q2 | definition | en | 1 -> none | 2 5 | 5 6 |" in text
    assert "| o1 | 0.660 | 0.580 | -0.080 | 40 -> 42 |" in text
    assert "one question moves a Hit rate by 3.3 points" in text


def test_main_prints_writes_and_rejects_bad_input(tmp_path):
    b, a, out = tmp_path / "b.json", tmp_path / "a.json", tmp_path / "cmp.md"
    b.write_text(json.dumps(BEFORE), encoding="utf-8")
    a.write_text(json.dumps(AFTER), encoding="utf-8")
    buffer = io.StringIO()
    assert compare_runs.main([str(b), str(a), "--out", str(out)], out=buffer) == 0
    assert "# Retrieval comparison" in buffer.getvalue() and out.read_text(encoding="utf-8").startswith("# Retrieval comparison")
    with pytest.raises(SystemExit, match="cannot read"):
        compare_runs.main([str(tmp_path / "missing.json"), str(a)])
    (tmp_path / "x.json").write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(SystemExit, match="not a run_eval.py results.json"):
        compare_runs.main([str(tmp_path / "x.json"), str(a)])
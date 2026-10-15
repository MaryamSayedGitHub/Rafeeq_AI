"""scripts/rescore_run.py: ground-truth changes are scored against a stored run without re-running retrieval."""

import io
import json

from eval_helpers import load_script
from rafeeq.knowledge.evaluation.dataset import BenchmarkItem, dump_benchmark
from rafeeq.knowledge.evaluation.runner import aggregate_retrieval, score_retrieval

rescore_run = load_script("rescore_run")
KS = [1, 3, 5]


def item(id="q1", pages=(10,), question=None, status="verified", **kw):
    base = dict(id=id, language="en", qtype="definition", question=question or f"question {id}", answerable=True, status=status,
                file_name="book.pdf", expected_pages=list(pages), evidence_quotes=["x" * 30], reference_answer="reference", key_facts=[["fact"]])
    base.update(kw)
    return BenchmarkItem(**base)


def oos(id="o1"):
    return BenchmarkItem(id=id, language="en", qtype="out_of_scope", question=f"off topic {id}", answerable=False, status="verified", absent_terms=["zebra"])


def hits(*pages, file_name="book.pdf"):
    return [{"file_name": file_name, "page": p, "score": 0.9 - 0.1 * i, "chunk_id": f"{file_name}-p{p}", "text": f"text {p}", "topic_id": "t"}
            for i, p in enumerate(pages)]


def make_run(items, retrieved):
    """A results.json-like document built with the real scorer, so rows carry text_preview like the stored ones."""
    records = [score_retrieval(it, retrieved[it.id], KS, 0.35) for it in items]
    return {"retrieval": {"ks": KS}, "per_question": records}


ITEMS = [item("q1", pages=(10,)), item("q2", pages=(20,)), oos("o1")]
RETRIEVED = {"q1": hits(5, 6, 10, 7, 8), "q2": hits(20, 1, 2, 3, 4), "o1": hits(9, 9, 9, 9, 9)}


def write(tmp_path, name, items):
    path = tmp_path / name
    dump_benchmark(items, path)
    return path


def run_main(tmp_path, run, benchmarks, *extra):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(run), encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    argv = [str(results)]
    for b in benchmarks:
        argv += ["--benchmark", str(b)]
    code = rescore_run.main(argv + list(extra), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_same_labels_reproduce_the_recorded_numbers():
    run = make_run(ITEMS, RETRIEVED)
    records, skipped = rescore_run.rescore(run, ITEMS, KS)
    assert skipped == []
    assert aggregate_retrieval(records, KS)["overall"] == aggregate_retrieval(run["per_question"], KS)["overall"]
    assert [r["first_relevant_rank"] for r in records if r["answerable"]] == [3, 1]


def test_changed_labels_change_only_the_labelled_questions_and_never_the_retrieved_pages():
    run = make_run(ITEMS, RETRIEVED)
    widened = [item("q1", pages=(10, 7)), item("q2", pages=(20,)), oos("o1")]
    before, _ = rescore_run.rescore(run, ITEMS, KS)
    after, _ = rescore_run.rescore(run, widened, KS)
    by_before = {r["id"]: r for r in before}
    by_after = {r["id"]: r for r in after}
    assert by_before["q1"]["first_relevant_rank"] == 3 and by_after["q1"]["first_relevant_rank"] == 3
    assert by_before["q1"]["recall@5"] == 1.0 and by_after["q1"]["recall@5"] == 1.0   # both pages 10 and 7 are in the top 5
    assert by_after["q1"]["expected_pages"] == [10, 7]
    assert by_before["q2"] == by_after["q2"]
    for r in after:
        assert [h["page"] for h in r["retrieved"]] == [h["page"] for h in RETRIEVED[r["id"]]]


def test_a_label_that_matches_an_earlier_rank_improves_hit_and_rank():
    run = make_run(ITEMS, RETRIEVED)
    relabelled = [item("q1", pages=(10, 6)), item("q2", pages=(20,)), oos("o1")]
    before, _ = rescore_run.rescore(run, ITEMS, KS)
    after, _ = rescore_run.rescore(run, relabelled, KS)
    b = {r["id"]: r for r in before}["q1"]
    a = {r["id"]: r for r in after}["q1"]
    assert (b["first_relevant_rank"], a["first_relevant_rank"]) == (3, 2)
    assert (b["hit@1"], a["hit@1"]) == (0.0, 0.0) and (b["hit@3"], a["hit@3"]) == (1.0, 1.0)
    assert b["rr"] == 1 / 3 and a["rr"] == 0.5


def test_items_that_cannot_be_rescored_are_listed_and_never_guessed():
    run = make_run(ITEMS, RETRIEVED)
    items = [item("q1", pages=(10,), question="a different question"),       # text changed: stored retrieval is for another query
             item("q2", pages=(20,)),
             item("q3", pages=(30,)),                                         # not in the run
             item("q4", pages=(40,), status="draft"),                         # unverified items are not scored at all
             oos("o1")]
    records, skipped = rescore_run.rescore(run, items, KS)
    assert sorted(r["id"] for r in records) == ["o1", "q2"]
    assert dict(skipped) == {"q1": "question text differs from the run", "q3": "not in the run"}


def test_a_run_that_kept_fewer_chunks_than_the_largest_k_is_not_rescored():
    run = make_run(ITEMS, RETRIEVED)
    for record in run["per_question"]:
        record["retrieved"] = record["retrieved"][:2]
    records, skipped = rescore_run.rescore(run, ITEMS, KS)
    assert records == []
    assert {qid for qid, _ in skipped} == {"q1", "q2", "o1"}
    assert "need 5" in dict(skipped)["q1"]


def test_report_compares_columns_lists_the_changed_questions_and_states_the_caveat(tmp_path):
    run = make_run(ITEMS, RETRIEVED)
    ref = write(tmp_path, "ref.jsonl", ITEMS)
    new = write(tmp_path, "new.jsonl", [item("q1", pages=(10, 6)), item("q2", pages=(20,)), oos("o1")])
    code, out, err = run_main(tmp_path, run, [ref, new])
    assert code == 0 and err == ""
    assert "| metric | ref.jsonl | new.jsonl |" in out
    assert "## Questions that score differently under new.jsonl than under ref.jsonl" in out
    assert "| q1 | [10] -> [10, 6] | 3 -> 2 |" in out
    assert "| q2 |" not in out
    assert "compare runs only under the same labels" in out


def test_identical_labels_report_no_differences_and_out_writes_the_file(tmp_path):
    run = make_run(ITEMS, RETRIEVED)
    ref = write(tmp_path, "ref.jsonl", ITEMS)
    same = write(tmp_path, "same.jsonl", ITEMS)
    target = tmp_path / "sub" / "report.md"
    code, out, _ = run_main(tmp_path, run, [ref, same], "--out", str(target))
    assert code == 0
    assert "differently under same.jsonl than under ref.jsonl" in out and "\nnone\n" in out
    assert target.read_text(encoding="utf-8") == out


def test_errors_use_exit_codes_and_say_why(tmp_path):
    ref = write(tmp_path, "ref.jsonl", ITEMS)
    out, err = io.StringIO(), io.StringIO()
    assert rescore_run.main([str(tmp_path / "missing.json"), "--benchmark", str(ref)], out=out, err=err) == 1
    assert "cannot read a run" in err.getvalue()
    run = make_run(ITEMS, RETRIEVED)
    code, _, err_text = run_main(tmp_path, run, [tmp_path / "nope.jsonl"])
    assert code == 1 and "cannot load" in err_text
    other = write(tmp_path, "other.jsonl", [item("z9", pages=(1,), question="unrelated")])
    code, _, err_text = run_main(tmp_path, run, [other])
    assert code == 1 and "nothing in" in err_text
    assert rescore_run.main([str(tmp_path / "results.json")], out=io.StringIO(), err=io.StringIO()) == 2   # --benchmark is required

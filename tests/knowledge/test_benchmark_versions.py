"""Benchmark files may only change the way eval/BENCHMARK_LABELS.md says: ground-truth pages are widened, nothing else.

Holds whether hands_on_llm_v1.jsonl is the original or has had the q01/q21 widening applied in place.
"""

from pathlib import Path

from rafeeq.knowledge.evaluation.dataset import item_problems, load_benchmark

BENCH = Path(__file__).resolve().parents[2] / "rafeeq" / "knowledge" / "eval" / "benchmark"
V1, V11 = BENCH / "hands_on_llm_v1.jsonl", BENCH / "hands_on_llm_v1_1.jsonl"
RELABELLED = {"q01", "q21", "q08"}
LABEL_FIELDS = {"expected_pages", "expected_chunk_ids", "evidence_quotes", "notes"}


def test_every_shipped_benchmark_file_loads_and_is_complete_when_verified():
    for path in sorted(BENCH.glob("*.jsonl")):
        items = load_benchmark(path)
        assert len({i.id for i in items}) == len(items), path.name
        for item in items:
            if item.verified:
                assert item_problems(item) == [], (path.name, item.id)


def test_v1_1_differs_from_v1_only_by_widened_page_labels_of_the_documented_questions():
    old = {i.id: i.to_dict() for i in load_benchmark(V1)}
    new = {i.id: i.to_dict() for i in load_benchmark(V11)}
    assert set(old) == set(new)
    differing = {qid for qid in old if old[qid] != new[qid]}
    assert differing <= RELABELLED, sorted(differing - RELABELLED)
    for qid in old:
        changed = {k for k in old[qid] if old[qid][k] != new[qid][k]}
        assert changed <= LABEL_FIELDS, (qid, sorted(changed - LABEL_FIELDS))      # question, answer, key facts, language: untouched
        assert set(old[qid]["expected_pages"]) <= set(new[qid]["expected_pages"]), qid   # labels are widened, never removed
        assert set(old[qid]["evidence_quotes"]) <= set(new[qid]["evidence_quotes"]), qid


def test_the_documented_corrections_are_present():
    new = {i.id: i for i in load_benchmark(V11)}
    assert new["q01"].expected_pages == new["q21"].expected_pages == [135, 141, 142]
    assert new["q08"].expected_pages == [501, 502, 505]
    assert "v1.1" in new["q08"].notes and "BENCHMARK_LABELS.md" in new["q08"].notes

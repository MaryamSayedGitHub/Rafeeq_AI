"""The held-out benchmark is frozen: any edit to it (or to the dev benchmark it must stay separate from) fails here."""

import hashlib
import json
from pathlib import Path

from rafeeq.knowledge.evaluation.dataset import item_problems, load_benchmark

BENCH = Path(__file__).resolve().parents[2] / "rafeeq" / "knowledge" / "eval" / "benchmark"
STRICT, LENIENT = BENCH / "hands_on_llm_heldout_v1.jsonl", BENCH / "hands_on_llm_heldout_v1_lenient.jsonl"
V1, V11 = BENCH / "hands_on_llm_v1.jsonl", BENCH / "hands_on_llm_v1_1.jsonl"
FREEZE = BENCH / "hands_on_llm_heldout_v1.sha256"
V1_SHA12 = "c872d23dcc16"          # the hash every dev-benchmark run on record printed
LENIENT_ONLY = {"m09": 470, "m10": 431, "a06": 414}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen():
    out = {}
    for line in FREEZE.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        out[name] = digest
    return out


def test_both_held_out_files_still_match_their_freeze_hashes():
    recorded = frozen()
    assert set(recorded) == {STRICT.name, LENIENT.name}
    assert sha(STRICT) == recorded[STRICT.name]
    assert sha(LENIENT) == recorded[LENIENT.name]


def test_the_dev_benchmark_is_untouched_by_the_held_out_work():
    # Normalize line endings so Windows CRLF and Unix LF are treated equally.
    normalized = V1.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(normalized).hexdigest()[:12] == V1_SHA12


def test_composition_is_the_documented_one():
    items = load_benchmark(STRICT)
    single = [i for i in items if i.answerable and i.qtype != "multi_passage"]
    multi = [i for i in items if i.qtype == "multi_passage"]
    oos = [i for i in items if not i.answerable]
    assert (len(items), len(single), len(multi), len(oos)) == (54, 23, 19, 12)
    assert {i.language for i in single} == {i.language for i in multi} == {i.language for i in oos} == {"en", "ar"}
    assert all(len(i.expected_pages) == 1 for i in single)
    assert all(len(i.expected_pages) >= 2 for i in multi)


def test_every_item_is_verified_and_carries_its_evidence():
    for item in load_benchmark(STRICT):
        assert item.verified and item_problems(item) == [], item.id
        if item.answerable:
            assert item.file_name == "hands_on_llm.pdf" and item.evidence_quotes and item.expected_chunk_ids, item.id
            assert all(len(q) >= 25 for q in item.evidence_quotes), item.id
            pages_of_chunks = {int(c.rsplit("-p", 1)[1].split("-")[0]) for c in item.expected_chunk_ids}
            assert pages_of_chunks == set(item.expected_pages), item.id
        else:
            assert item.absent_terms, item.id


def test_it_is_separate_from_v1_and_v1_1():
    held = load_benchmark(STRICT)
    for dev_path in (V1, V11):
        dev = load_benchmark(dev_path)
        assert not {i.id for i in held} & {i.id for i in dev}, dev_path.name
        assert not {i.question for i in held} & {i.question for i in dev}, dev_path.name
    dev_pages = {p for i in load_benchmark(V11) for p in i.expected_pages}
    assert not dev_pages & {p for i in held for p in i.expected_pages}     # no page is labelled in both sets


def test_lenient_labels_add_exactly_three_documented_pages_and_nothing_else():
    strict = {i.id: i.to_dict() for i in load_benchmark(STRICT)}
    lenient = {i.id: i.to_dict() for i in load_benchmark(LENIENT)}
    assert set(strict) == set(lenient)
    for qid, s in strict.items():
        l = lenient[qid]
        if qid in LENIENT_ONLY:
            assert set(l["expected_pages"]) == set(s["expected_pages"]) | {LENIENT_ONLY[qid]}, qid
            assert set(s["evidence_quotes"]) < set(l["evidence_quotes"]), qid
        else:
            assert l["expected_pages"] == s["expected_pages"] and l["evidence_quotes"] == s["evidence_quotes"], qid
        for field in ("question", "language", "qtype", "answerable", "reference_answer", "key_facts", "absent_terms"):
            assert l[field] == s[field], (qid, field)


def test_the_jsonl_is_plain_utf8_with_unix_newlines():
    raw = STRICT.read_bytes()
    assert b"\r" not in raw
    for line in raw.decode("utf-8").splitlines():
        json.loads(line)

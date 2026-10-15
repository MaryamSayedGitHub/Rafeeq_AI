"""Benchmark schema validation, evidence resolution against stored chunks, and scripts/validate_benchmark.py."""

import dataclasses
import io
import json
import shutil

import pytest

from eval_helpers import BOOK, OTHER, load_script, save_store
from rafeeq.knowledge.evaluation.dataset import (
    BenchmarkError,
    BenchmarkItem,
    dump_benchmark,
    item_problems,
    load_benchmark,
    parse_benchmark,
)
from rafeeq.knowledge.evaluation.ground_truth import (
    PageIndex,
    find_quote,
    lexical_search,
    load_store_chunks,
    merge_overlapping,
    normalize_text,
    present_absent_terms,
    resolve_item,
    squash,
)
from rafeeq.knowledge.store_persistence import PersistenceError

validate_benchmark = load_script("validate_benchmark")
find_evidence = load_script("find_evidence")

QUOTE = "Retrieval-augmented generation fetches documents before answering"


def draft(**overrides):
    base = dict(id="q1", language="en", qtype="definition", question="What is RAG?", answerable=True, file_name="book.pdf")
    base.update(overrides)
    return BenchmarkItem(**base)


def verified(**overrides):
    base = dict(status="verified", expected_pages=[4], evidence_quotes=[QUOTE], reference_answer="RAG fetches documents first.", key_facts=[["documents"]])
    base.update(overrides)
    return draft(**base)


# ----------------------------------------------------------------- dataset

def test_a_bare_draft_item_is_valid():
    assert item_problems(draft()) == []


def test_verified_item_needs_everything_a_scorer_depends_on():
    problems = item_problems(draft(status="verified"))
    for needle in ("expected_pages", "evidence_quotes", "reference_answer", "key_facts"):
        assert any(needle in p for p in problems), needle
    assert item_problems(verified()) == []


def test_multi_passage_needs_two_pages():
    assert any("at least 2" in p for p in item_problems(verified(qtype="multi_passage")))
    assert item_problems(verified(qtype="multi_passage", expected_pages=[4, 9])) == []


def test_out_of_scope_rules():
    oos = draft(id="o1", qtype="out_of_scope", answerable=False, file_name=None, question="Capital of Mars?")
    assert item_problems(oos) == []
    assert any("expected_pages" in p for p in item_problems(dataclasses.replace(oos, expected_pages=[1])))
    assert any("verified out-of-scope" in p for p in item_problems(dataclasses.replace(oos, status="verified")))
    assert item_problems(dataclasses.replace(oos, status="verified", absent_terms=["Mars"])) == []
    assert any("out_of_scope" in p for p in item_problems(draft(qtype="out_of_scope")))   # answerable + out_of_scope
    assert any("out_of_scope" in p for p in item_problems(dataclasses.replace(oos, qtype="definition")))


def test_field_level_validation():
    assert any("language" in p for p in item_problems(draft(language="fr")))
    assert any("expected_pages" in p for p in item_problems(draft(expected_pages=[0])))
    assert any("duplicates" in p for p in item_problems(draft(expected_pages=[2, 2])))
    assert any("at least 25" in p for p in item_problems(draft(evidence_quotes=["too short"])))
    assert any("key_facts" in p for p in item_problems(draft(key_facts=[[]])))
    assert any("file_name" in p for p in item_problems(draft(file_name=None)))


def test_parse_reports_line_numbers_unknown_fields_and_duplicates():
    good = json.dumps(draft().to_dict())
    typo = json.dumps({**draft(id="q2").to_dict(), "expected_page": [1]})
    with pytest.raises(BenchmarkError) as info:
        parse_benchmark([good, "", typo, "{not json", good])
    message = str(info.value)
    assert ":3:" in message and "expected_page" in message      # unknown field on line 3
    assert ":4:" in message and "invalid JSON" in message
    assert ":5:" in message and "duplicate id" in message


def test_parse_rejects_missing_required_fields_and_empty_files():
    with pytest.raises(BenchmarkError, match="missing required"):
        parse_benchmark([json.dumps({"id": "q1"})])
    with pytest.raises(BenchmarkError, match="no items"):
        parse_benchmark(["", "  "])


def test_dump_and_load_round_trip_keeps_arabic(tmp_path):
    items = [draft(), draft(id="q2", language="ar", question="ما هو الانتباه الذاتي؟")]
    path = tmp_path / "b.jsonl"
    dump_benchmark(items, path)
    assert "ما هو الانتباه الذاتي؟" in path.read_text(encoding="utf-8")   # not \u-escaped
    assert load_benchmark(path) == items


def test_shipped_benchmark_is_valid_and_never_claims_unchecked_ground_truth():
    """Holds at every stage: while items are drafts and after scripts/validate_benchmark.py promotes them."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "rafeeq" / "knowledge" / "eval" / "benchmark" / "hands_on_llm_v1.jsonl"
    items = load_benchmark(path)                                        # raises on any schema problem
    assert any(i.answerable for i in items) and any(not i.answerable for i in items)
    assert {i.language for i in items} == {"en", "ar"}
    for item in items:
        if item.status == "draft":                                      # a draft may not carry hand-typed ground truth
            assert not item.expected_pages and not item.expected_chunk_ids and item.reference_answer is None, item.id
        else:                                                           # verified: must be complete (also enforced by load_benchmark)
            assert item_problems(item) == [], item.id


# ------------------------------------------------------------- text matching

def test_normalisation_is_forgiving_about_extraction_noise_only():
    assert squash("Self-attention") == squash("self attention") == squash("SELF–\nATTENTION") == "selfattention"
    assert squash("ﬁne-tuning") == squash("fine tuning")                      # ligature
    assert squash("الانتباهُ الذاتيّ") == squash("الانتباه الذاتي")             # diacritics
    assert squash("أحمد") == squash("احمد")                                    # alef forms
    assert squash("الانتبـــاه") == squash("الانتباه")                           # tatweel (kashida) is a letter-like character: must be removed
    assert normalize_text("الانتباهُ الذاتيّ") == "الانتباه الذاتي"              # used by the keyword search, which does not squash
    assert squash("attention") != squash("attentions")
    assert normalize_text("  A   B ") == "a b"


def test_merge_overlapping_rebuilds_text_without_the_repeated_overlap():
    words = [f"w{i}" for i in range(1, 31)]
    chunk1, chunk2 = " ".join(words[:20]), " ".join(words[14:])          # 6 words of overlap
    assert merge_overlapping([chunk1, chunk2]).split() == words
    assert merge_overlapping(["a b c", "x y z"]) == "a b c x y z"        # no overlap
    assert merge_overlapping([]) == ""


def record(chunk_id, page, text, file_name="book.pdf"):
    return {"chunk_id": chunk_id, "file_name": file_name, "page": page, "text": text, "room_id": "r"}


def two_chunk_page():
    words = [f"w{i}" for i in range(1, 31)]
    return [
        record("x-p0003-c000", 3, " ".join(words[:20])),
        record("x-p0003-c001", 3, " ".join(words[14:])),
        record("x-p0004-c000", 4, "Unrelated text about something else entirely here."),
    ]


def test_find_quote_reports_chunk_ids_or_page_level_only_when_it_straddles_chunks():
    index = PageIndex.build(two_chunk_page())
    inside = find_quote(index, "book.pdf", "w2 w3 w4 w5 w6 w7 w8 w9")
    assert inside == {3: ["x-p0003-c000"]}
    straddling = find_quote(index, "book.pdf", " ".join(f"w{i}" for i in range(10, 26)))   # w10..w25 is in neither chunk
    assert straddling == {3: []}
    assert find_quote(index, "book.pdf", "not in the book at all") == {}
    assert find_quote(index, "other.pdf", "w2 w3 w4 w5 w6 w7 w8 w9") == {}


# --------------------------------------------------------------- resolution

def book_index():
    records = [record(f"b-p{n:04d}-c000", n, text) for n, text in enumerate(BOOK, start=1)]
    records += [record(f"o-p{n:04d}-c000", n, text, "other.pdf") for n, text in enumerate(OTHER, start=1)]
    return PageIndex.build(records)


def test_resolve_item_derives_pages_and_chunks_from_the_quote():
    result = resolve_item(draft(evidence_quotes=[QUOTE]), book_index())
    assert result.ok and result.pages == [4] and result.chunk_ids == ["b-p0004-c000"]


def test_resolve_item_checks_declared_pages():
    index = book_index()
    assert resolve_item(verified(), index).ok
    wrong = resolve_item(verified(expected_pages=[2]), index)
    assert not wrong.ok and "is on page(s) [4] but expected_pages are [2]" in wrong.problems[0]
    extra = resolve_item(verified(expected_pages=[4, 6]), index)
    assert not extra.ok and any("not supported by any quote" in p for p in extra.problems)


def test_resolve_item_reports_missing_quotes_files_and_ambiguity():
    index = book_index()
    assert "quote not found" in resolve_item(draft(evidence_quotes=["this sentence is nowhere in the book"]), index).problems[0]
    assert "not in the store" in resolve_item(draft(file_name="missing.pdf", evidence_quotes=[QUOTE]), index).problems[0]
    assert "no evidence_quotes" in resolve_item(draft(), index).problems[0]
    repeated = PageIndex.build([record("a-p0001-c000", 1, "The same sentence appears twice in this book."),
                                record("a-p0002-c000", 2, "The same sentence appears twice in this book.")])
    ambiguous = resolve_item(draft(evidence_quotes=["The same sentence appears twice"]), repeated)
    assert not ambiguous.ok and "several pages [1, 2]" in ambiguous.problems[0]
    chosen = resolve_item(draft(expected_pages=[2], evidence_quotes=["The same sentence appears twice"]), repeated)
    assert chosen.ok and chosen.pages == [2]


def test_multi_passage_pages_are_the_union_of_what_each_quote_supports():
    item = draft(qtype="multi_passage", evidence_quotes=[QUOTE, "LoRA adapts large models by training small"])
    result = resolve_item(item, book_index())
    assert result.ok and result.pages == [4, 5]


def test_out_of_scope_terms_are_checked_against_the_whole_store():
    index = book_index()
    assert present_absent_terms(index, ["koshari", "Mount Everest"]) == {}
    assert present_absent_terms(index, ["neural networks", "koshari"]) == {"neural networks": 2}  # pages 2 and 3


def test_lexical_search_ranks_the_matching_page_first_and_ignores_nonsense():
    records = [record(f"b-p{n:04d}-c000", n, text) for n, text in enumerate(BOOK, start=1)]
    hits = lexical_search(records, "retrieval augmented generation documents")
    assert hits[0]["page"] == 4 and "retrieval" in hits[0]["matched"]
    assert lexical_search(records, "zebra giraffe") == []
    assert lexical_search(records, "the and for") == []                           # only stop words
    assert all(h["file_name"] == "book.pdf" for h in lexical_search(records, "tokens", file_name="book.pdf"))


# ----------------------------------------------- reading a real saved store

def test_load_store_chunks_reads_the_saved_metadata(tmp_path):
    save_store(tmp_path, {"room_A": {"book.pdf": BOOK, "other.pdf": OTHER}})
    records = load_store_chunks(tmp_path)
    assert len(records) == 7 and {r["file_name"] for r in records} == {"book.pdf", "other.pdf"}
    assert all({"chunk_id", "page", "text", "room_id"} <= set(r) for r in records)
    index = PageIndex.build(records)
    assert find_quote(index, "book.pdf", QUOTE).keys() == {4}


def test_load_store_chunks_rejects_missing_or_broken_metadata(tmp_path):
    with pytest.raises(PersistenceError, match="cannot read"):
        load_store_chunks(tmp_path)
    (tmp_path / "metadata.json").write_text('{"records": [{"chunk_id": "a"}]}', encoding="utf-8")
    with pytest.raises(PersistenceError, match="lacks"):
        load_store_chunks(tmp_path)


# ----------------------------------------------- scripts/validate_benchmark.py

def write_benchmark(path, items):
    dump_benchmark(items, path)
    return path


def run_validate(*argv):
    out, err = io.StringIO(), io.StringIO()
    return validate_benchmark.main(list(argv), out=out, err=err), out.getvalue(), err.getvalue()


def make_benchmark_and_store(tmp_path):
    store_dir = tmp_path / "store"
    save_store(store_dir, {"room_A": {"book.pdf": BOOK, "other.pdf": OTHER}})
    complete = draft(id="q1", evidence_quotes=[QUOTE], reference_answer="RAG fetches documents first.", key_facts=[["documents"]])
    no_answer = draft(id="q2", evidence_quotes=["LoRA adapts large models by training small"])        # lacks reference_answer / key_facts
    bad_quote = draft(id="q3", evidence_quotes=["a sentence that the book does not contain at all"], reference_answer="x", key_facts=[["x"]])
    no_quotes = draft(id="q4")
    oos_ok = draft(id="o1", qtype="out_of_scope", answerable=False, file_name=None, question="Koshari recipe?", absent_terms=["koshari"])
    oos_bad = draft(id="o2", qtype="out_of_scope", answerable=False, file_name=None, question="How are neural networks trained?", absent_terms=["neural networks"])
    path = write_benchmark(tmp_path / "b.jsonl", [complete, no_answer, bad_quote, no_quotes, oos_ok, oos_bad])
    return path, store_dir


def test_validate_reports_without_writing_by_default(tmp_path):
    path, store_dir = make_benchmark_and_store(tmp_path)
    before = path.read_bytes()
    code, out, _ = run_validate("--benchmark", str(path), "--store-dir", str(store_dir))
    assert code == 0 and path.read_bytes() == before and not (tmp_path / "b.jsonl.bak").exists()
    assert "can be promoted" in out and "q1" in out
    assert "quote not found" in out and "no evidence_quotes" in out
    assert "DOES occur in the store" in out
    assert "0 of 6 items verified; 2 more can be promoted (re-run with --write)" in out   # q1 and o1; nothing is promoted without --write


def test_validate_write_promotes_only_what_really_verifies(tmp_path):
    path, store_dir = make_benchmark_and_store(tmp_path)
    original = path.read_bytes()
    code, out, _ = run_validate("--benchmark", str(path), "--store-dir", str(store_dir), "--write")
    assert code == 0
    assert (tmp_path / "b.jsonl.bak").read_bytes() == original            # previous version kept
    by_id = {i.id: i for i in load_benchmark(path)}
    assert by_id["q1"].status == "verified" and by_id["q1"].expected_pages == [4]
    assert len(by_id["q1"].expected_chunk_ids) == 1 and by_id["q1"].expected_chunk_ids[0].endswith("-p0004-c000")
    assert by_id["o1"].status == "verified"
    for stays_draft in ("q2", "q3", "q4", "o2"):
        assert by_id[stays_draft].status == "draft", stays_draft
    assert by_id["q2"].expected_pages == []                                # no half-filled ground truth either


def test_validate_flags_a_verified_item_whose_evidence_disappeared(tmp_path):
    path, store_dir = make_benchmark_and_store(tmp_path)
    run_validate("--benchmark", str(path), "--store-dir", str(store_dir), "--write")
    other_store = tmp_path / "other_store"
    save_store(other_store, {"room_A": {"book.pdf": ["Completely different book text lives here now."]}})
    code, out, _ = run_validate("--benchmark", str(path), "--store-dir", str(other_store))
    assert code == 1 and "BROKEN" in out and "previously verified item(s) BROKEN" in out


def test_validate_reports_bad_input(tmp_path):
    code, _, err = run_validate("--benchmark", str(tmp_path / "missing.jsonl"), "--store-dir", str(tmp_path))
    assert code == 1 and "cannot read benchmark" in err
    path, _ = make_benchmark_and_store(tmp_path)
    code, _, err = run_validate("--benchmark", str(path), "--store-dir", str(tmp_path / "no_store"))
    assert code == 1 and "cannot read" in err


def test_find_evidence_prints_pages_and_snippets(tmp_path):
    save_store(tmp_path, {"room_A": {"book.pdf": BOOK}})
    out, err = io.StringIO(), io.StringIO()
    assert find_evidence.main(["retrieval augmented generation", "--store-dir", str(tmp_path)], out=out, err=err) == 0
    assert "page    4" in out.getvalue() and "book.pdf" in out.getvalue()
    assert find_evidence.main(["zebra", "--store-dir", str(tmp_path)], out=io.StringIO(), err=err) == 1
    assert "no chunk contains" in err.getvalue()
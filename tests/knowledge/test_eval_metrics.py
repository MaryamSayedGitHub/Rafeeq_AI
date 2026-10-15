"""Metric definitions, checked against values worked out by hand (not against the code's own output)."""

import pytest

from rafeeq.knowledge.evaluation.metrics import (
    abstention_outcome,
    answer_body,
    citation_scores,
    first_relevant_rank,
    hit_at_k,
    key_fact_coverage,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
    wilson_interval,
)


def chunk(file_name, page, chunk_id="c"):
    return {"file_name": file_name, "page": page, "chunk_id": chunk_id}


# ------------------------------------------------------------------ relevance

def test_page_level_relevance_needs_the_right_document_and_page():
    retrieved = [chunk("book.pdf", 5), chunk("book.pdf", 10), chunk("other.pdf", 10), chunk("book.pdf", 20)]
    assert relevance_flags(retrieved, "book.pdf", [10, 20]) == [False, True, False, True]


def test_chunk_level_relevance_uses_chunk_ids_only():
    retrieved = [chunk("book.pdf", 10, "a"), chunk("book.pdf", 10, "b"), chunk("other.pdf", 10, "b")]
    assert relevance_flags(retrieved, "book.pdf", expected_chunk_ids=["b"], level="chunk") == [False, True, False]


def test_relevance_rejects_missing_ground_truth_and_bad_level():
    with pytest.raises(ValueError, match="at least one expected page"):
        relevance_flags([], "book.pdf", [])
    with pytest.raises(ValueError, match="at least one expected chunk id"):
        relevance_flags([], "book.pdf", [1], level="chunk")
    with pytest.raises(ValueError, match="level"):
        relevance_flags([], "book.pdf", [1], level="paragraph")


# ------------------------------------------------------------- rank metrics
# flags = [F, T, F, T, F]: relevant at ranks 2 and 4

FLAGS = [False, True, False, True, False]


def test_hit_at_k():
    assert [hit_at_k(FLAGS, k) for k in (1, 2, 3, 5)] == [0.0, 1.0, 1.0, 1.0]


def test_precision_at_k_divides_by_k():
    assert precision_at_k(FLAGS, 1) == 0.0
    assert precision_at_k(FLAGS, 2) == 0.5       # 1 of 2
    assert precision_at_k(FLAGS, 3) == pytest.approx(1 / 3)
    assert precision_at_k(FLAGS, 5) == pytest.approx(2 / 5)


def test_precision_keeps_k_as_denominator_when_fewer_results_exist():
    assert precision_at_k([True, True], 5) == pytest.approx(2 / 5)  # only two chunks came back


def test_reciprocal_rank_and_first_rank():
    assert reciprocal_rank(FLAGS) == 0.5
    assert first_relevant_rank(FLAGS) == 2
    assert reciprocal_rank(FLAGS, k=1) == 0.0   # first relevant is beyond the cut-off
    assert reciprocal_rank([False, False]) == 0.0 and first_relevant_rank([False, False]) is None
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)


def test_recall_counts_distinct_expected_units_once():
    units = [("b", 5), ("b", 10), ("b", 10), ("b", 20), ("b", 7)]   # page 10 appears twice
    expected = {("b", 10), ("b", 20)}
    assert recall_at_k(units, expected, 1) == 0.0
    assert recall_at_k(units, expected, 3) == 0.5    # page 10 found, 20 not yet; duplicates do not double count
    assert recall_at_k(units, expected, 4) == 1.0


def test_recall_for_a_single_expected_unit_is_binary():
    assert recall_at_k([("b", 1), ("b", 2)], [("b", 2)], 1) == 0.0
    assert recall_at_k([("b", 1), ("b", 2)], [("b", 2)], 2) == 1.0


@pytest.mark.parametrize("bad_k", [0, -1, True, 1.5, None])
def test_k_must_be_a_positive_integer(bad_k):
    with pytest.raises(ValueError, match="positive integer"):
        hit_at_k(FLAGS, bad_k)


def test_recall_rejects_empty_ground_truth():
    with pytest.raises(ValueError, match="at least one expected unit"):
        recall_at_k([("b", 1)], [], 1)


def test_mean():
    assert mean([1.0, 0.0, 0.5]) == 0.5 and mean([]) is None


def test_wilson_interval():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(50, 100)           # worked out: 0.5 +/- 0.0962
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    low, high = wilson_interval(30, 30)             # perfect score is still uncertain with n = 30
    assert high == pytest.approx(1.0) and 0.85 < low < 0.95
    assert wilson_interval(0, 30)[0] == pytest.approx(0.0, abs=1e-9)


# -------------------------------------------------------------- abstention

@pytest.mark.parametrize(
    "answerable,grounded,expected",
    [(True, True, "answered"), (True, False, "false_abstention"),
     (False, False, "correct_abstention"), (False, True, "unsupported_answer")],
)
def test_abstention_outcome_covers_all_four_cases(answerable, grounded, expected):
    assert abstention_outcome(answerable, grounded) == expected


# --------------------------------------------------------------- citations

def test_citation_scores_mixed_citations():
    citations = [{"file_name": "book.pdf", "page": 10}, {"file_name": "book.pdf", "page": 12}, {"file_name": "other.pdf", "page": 10}]
    result = citation_scores(citations, "book.pdf", [10, 11])
    assert result["n_citations"] == 3
    assert result["precision"] == pytest.approx(1 / 3)   # only (book, 10) is correct
    assert result["recall"] == 0.5                         # page 11 never cited
    assert result["hit"] == 1.0
    assert result["wrong_document"] == 1


def test_citation_scores_all_correct_and_none_cited():
    full = citation_scores([{"file_name": "b", "page": 1}, {"file_name": "b", "page": 2}], "b", [1, 2])
    assert (full["precision"], full["recall"], full["hit"]) == (1.0, 1.0, 1.0)
    nothing = citation_scores([], "b", [1])
    assert nothing["precision"] is None and nothing["recall"] == 0.0 and nothing["hit"] == 0.0


def test_duplicate_citations_count_once():
    result = citation_scores([{"file_name": "b", "page": 1}, {"file_name": "b", "page": 1}], "b", [1])
    assert result["n_citations"] == 1 and result["precision"] == 1.0


def test_citation_scoring_needs_expected_pages():
    with pytest.raises(ValueError):
        citation_scores([], "b", [])


# -------------------------------------------------------------- key facts

def test_answer_body_drops_sources_footer_and_markers():
    md = "Self-attention weighs tokens [1][2].\n\n**Sources:**\n- [1] book.pdf, page 3\n- [2] book.pdf, page 4"
    assert answer_body(md) == "Self-attention weighs tokens ."
    assert answer_body("no footer [3]") == "no footer"


def test_key_fact_coverage_counts_facts_with_any_spelling():
    facts = [["self-attention", "self attention"], ["token"], ["softmax"]]
    result = key_fact_coverage("Self Attention lets each Token look at the others.", facts)
    assert result["covered"] == 2 and result["total"] == 3 and result["coverage"] == pytest.approx(2 / 3)
    assert result["missing"] == [["softmax"]]


def test_key_fact_coverage_is_diacritic_and_alef_insensitive_for_arabic():
    facts = [["الانتباه الذاتي", "self-attention"], ["الرموز"]]
    answer = "يوفّر الانتباهُ الذاتيّ وزناً لكل الرّموز في الجملة"
    assert key_fact_coverage(answer, facts)["coverage"] == 1.0
    assert key_fact_coverage("إجابة أخرى", facts)["coverage"] == 0.0


def test_key_fact_coverage_needs_facts():
    with pytest.raises(ValueError):
        key_fact_coverage("anything", [])
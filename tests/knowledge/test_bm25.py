"""In-project BM25: tokenisation and scoring, checked against hand calculations."""

import math

import pytest

from rafeeq.knowledge.retrieval.bm25 import B, K1, BM25Index, tokenize


def test_tokenize_lowercases_splits_on_hyphens_and_drops_single_characters():
    assert tokenize("Self-Attention in a Transformer") == ["self", "attention", "in", "transformer"]
    assert tokenize("GPT-4 and x") == ["gpt", "and"]                          # "4" and "x" are single characters
    assert tokenize("") == [] and tokenize("  ... ") == []


def test_plural_s_is_removed_from_plain_english_words_only():
    assert tokenize("models tokens embeddings") == ["model", "token", "embedding"]
    assert tokenize("class loss bus analysis") == ["class", "loss", "bus", "analysis"]      # -ss, -us, -is are kept
    assert tokenize("ids") == ["ids"]                                                       # words of three letters or fewer are left alone


def test_arabic_is_normalised_so_spelling_variants_match():
    assert tokenize("الانتباهُ") == tokenize("الانتباه")                       # diacritics
    assert tokenize("الانتبـــاه") == tokenize("الانتباه")                     # tatweel
    assert tokenize("أحمد إبراهيم") == tokenize("احمد ابراهيم")                  # alef forms
    assert tokenize("مدرسة") == tokenize("مدرسه")                              # ta marbuta
    assert tokenize("ما هو الانتباه الذاتي (self-attention)")[-2:] == ["self", "attention"]


def test_tokenize_rejects_non_strings():
    with pytest.raises(TypeError):
        tokenize(None)


# Three documents, small enough to compute by hand.
DOCS = [["apple", "apple", "banana"], ["banana", "cherry"], ["cherry", "cherry", "cherry", "date"]]


def expected(tf, length, avgdl, n_t, n=3):
    idf = math.log(1 + (n - n_t + 0.5) / (n_t + 0.5))
    return idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avgdl))


def test_scores_match_the_okapi_formula_by_hand():
    index = BM25Index(DOCS)
    avgdl = (3 + 2 + 4) / 3
    scores = index.scores(["banana"])
    assert set(scores) == {0, 1}                                                # the cherry/date document does not match
    assert scores[0] == pytest.approx(expected(1, 3, avgdl, n_t=2))
    assert scores[1] == pytest.approx(expected(1, 2, avgdl, n_t=2))
    assert scores[1] > scores[0]                                                # same tf, shorter document wins
    assert index.idf("apple") == pytest.approx(math.log(1 + 2.5 / 1.5))
    assert index.idf("absent") == pytest.approx(math.log(1 + 3.5 / 0.5))


def test_scores_add_over_distinct_query_terms_and_ignore_repeated_ones():
    index = BM25Index(DOCS)
    both = index.scores(["apple", "banana"])
    assert both[0] == pytest.approx(index.scores(["apple"])[0] + index.scores(["banana"])[0])
    assert index.scores(["apple", "apple"]) == index.scores(["apple"])


def test_top_orders_by_score_then_position_and_never_returns_non_matches():
    index = BM25Index([["x", "y"], ["x", "y"], ["z"]])
    assert [p for p, _ in index.top(["x"])] == [0, 1]                           # equal scores: lower position first
    assert index.top(["nothing"]) == [] and index.top([]) == []                 # no overlap -> empty, not arbitrary
    assert len(index.top(["x"], n=1)) == 1


def test_empty_index_and_invalid_parameters():
    assert BM25Index([]).top(["x"]) == [] and len(BM25Index([])) == 0
    with pytest.raises(ValueError):
        BM25Index([["a"]], b=1.5)
    with pytest.raises(ValueError):
        BM25Index([["a"]], k1=-1)


def test_the_documented_parameters_are_pinned_with_a_literal_score():
    assert (K1, B) == (1.5, 0.75)
    # one document, one occurrence, N = n = 1 and |d| = avgdl: idf = ln(1 + 0.5/1.5), score = idf * 2.5 / (1 + 1.5)
    assert BM25Index([["only"]]).scores(["only"])[0] == pytest.approx(math.log(1 + 0.5 / 1.5))
    assert BM25Index([["a", "a"], ["b"]], k1=1.5, b=0.75).scores(["a"])[0] == pytest.approx(0.6931471805599453 * 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 2 / 1.5)))

"""Spot book pages that carry no teachable content, so retrieval can skip them.

The back-of-book *index* is the case that matters here: every line is a term
followed by section titles ("retrieval-augmented generation, Overview of Semantic
Search and RAG, ..."), so a page of it embeds close to almost any topical query and
takes top-k slots from the pages that actually explain the topic.

The test uses only four properties of the running text, none of which depends on
how the PDF extractor wrapped lines (chunks are stored as space-joined words):

* very few sentence endings (an index has no sentences),
* many capitalised words (section titles),
* few function words ("the", "is", "we", ...),
* many commas (the "term, Title, Title" pattern).

Thresholds were chosen on the one book this project indexes (``hands_on_llm.pdf``):
on its 598 pages no body page is flagged and about 96% of the index pages are.
They are a heuristic, not a guarantee: audit a new corpus with
``scripts/find_non_content.py`` before relying on the filter there. Text that is
mostly non-Latin (Arabic) has too few Latin words to be flagged.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

MIN_WORDS = 60
MAX_SENTENCE_ENDS_PER_100_WORDS = 1.0
MIN_CAPITALISED_SHARE = 0.25
MAX_FUNCTION_WORD_SHARE = 0.15
MIN_COMMAS_PER_WORD = 0.08

_FUNCTION_WORDS = frozenset("the is are we that this it can which of a to".split())
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENTENCE_END = re.compile(r"[a-z0-9\)\]”\"]\.(?: |$)")


def page_features(text: str) -> dict[str, float]:
    """The numbers :func:`is_non_content_page` decides on (all zero for text without Latin words)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    flat = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))
    words = _WORD.findall(flat)
    n = len(words)
    if n == 0:
        return {"words": 0, "sentence_ends_per_100": 0.0, "capitalised": 0.0, "function_words": 0.0, "commas_per_word": 0.0}
    return {
        "words": n,
        "sentence_ends_per_100": 100.0 * len(_SENTENCE_END.findall(flat)) / n,
        "capitalised": sum(w[0].isupper() for w in words) / n,
        "function_words": sum(w.lower() in _FUNCTION_WORDS for w in words) / n,
        "commas_per_word": flat.count(",") / n,
    }


def is_non_content_page(text: str) -> bool:
    """True for index-like text: no sentences, title-case terms, few function words, many commas."""
    f = page_features(text)
    return (
        f["words"] >= MIN_WORDS
        and f["sentence_ends_per_100"] < MAX_SENTENCE_ENDS_PER_100_WORDS
        and f["capitalised"] >= MIN_CAPITALISED_SHARE
        and f["function_words"] <= MAX_FUNCTION_WORD_SHARE
        and f["commas_per_word"] >= MIN_COMMAS_PER_WORD
    )


def filter_hits(hits: list[Any]) -> list[Any]:
    """Drop index-like hits (objects with a ``.text``), keeping order."""
    return [hit for hit in hits if not is_non_content_page(hit.text)]
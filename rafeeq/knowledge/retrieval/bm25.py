"""A small in-project BM25 (Okapi) index. No dependency, no model, no network.

    tokens = tokenize(text)
    index = BM25Index([tokenize(t) for t in texts])
    index.top(tokenize(query), n=50)   # [(document position, score), ...] best first, only documents that match

Scoring (standard Okapi BM25, Lucene-style non-negative idf)::

    idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
    score(q, d) = sum over distinct query terms t of
                  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

with k1 = 1.5 and b = 0.75. A document that shares no term with the query scores 0 and is NOT returned, so a
query with no lexical overlap (for example an Arabic question over an English book) yields an empty ranking
instead of an arbitrary one.

Tokenisation is deliberately simple and language-light: Unicode NFKC, case folding, Arabic diacritics and
tatweel removed, alef/ya/ta-marbuta variants folded, words split on anything that is not a letter or a digit
(so ``self-attention`` is ``self`` + ``attention``), single characters dropped, and a trailing plural ``s``
removed from English words longer than three letters. There is no stop-word list: idf already makes
frequent words nearly worthless.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Sequence

K1 = 1.5
B = 0.75

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_ARABIC_MARKS = re.compile("[ً-ٰٟـ]")  # harakat, superscript alef, tatweel
_ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})


def _stem(token: str) -> str:
    """Drop a plural -s from plain ASCII words (models -> model); leave everything else alone."""
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")) and token.isascii() and token.isalpha():
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-cased, normalised word tokens of ``text`` (see the module docstring)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = _ARABIC_MARKS.sub("", folded).translate(_ARABIC_FOLD)
    return [_stem(token) for token in _WORD.findall(folded) if len(token) > 1]


class BM25Index:
    """Immutable BM25 index over pre-tokenised documents; a document is its position in the input list."""

    def __init__(self, documents: Sequence[Sequence[str]], *, k1: float = K1, b: float = B) -> None:
        if not (k1 >= 0 and 0 <= b <= 1):
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1!r} b={b!r}")
        self._k1, self._b = float(k1), float(b)
        self._lengths = [len(doc) for doc in documents]
        self._count = len(self._lengths)
        self._avgdl = (sum(self._lengths) / self._count) if self._count else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                postings[term].append((position, tf))
        self._postings = dict(postings)

    def __len__(self) -> int:
        return self._count

    def idf(self, term: str) -> float:
        n = len(self._postings.get(term, ()))
        return math.log(1.0 + (self._count - n + 0.5) / (n + 0.5))

    def scores(self, query_tokens: Sequence[str]) -> dict[int, float]:
        """BM25 score of every document sharing at least one term with the query (others are absent = 0)."""
        out: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for position, tf in postings:
                norm = 1.0 - self._b + self._b * (self._lengths[position] / self._avgdl) if self._avgdl else 1.0
                out[position] += idf * tf * (self._k1 + 1.0) / (tf + self._k1 * norm)
        return dict(out)

    def top(self, query_tokens: Sequence[str], n: int | None = None) -> list[tuple[int, float]]:
        """Matching documents best first as ``(position, score)``; ties broken by position. ``n=None`` = all."""
        ranked = sorted(((p, s) for p, s in self.scores(query_tokens).items() if s > 0.0), key=lambda item: (-item[1], item[0]))
        return ranked if n is None else ranked[:n]

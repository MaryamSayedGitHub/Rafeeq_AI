"""Metric definitions. Pure functions on plain data; no models, no store, no network.

Relevance (the single definition used everywhere)
-------------------------------------------------
A retrieved chunk is RELEVANT to a question when
    - page level (primary):  chunk.file_name == item.file_name  AND  chunk.page in item.expected_pages
    - chunk level (strict, only for items that have expected_chunk_ids):
                             chunk.file_name == item.file_name  AND  chunk.chunk_id in item.expected_chunk_ids
Similarity scores play no part in relevance; they are never used as correctness labels.

Retrieval metrics at cut-off K (over the first K retrieved chunks; fewer are returned if the room has fewer)
    Hit@K        1 if at least one of the top K chunks is relevant, else 0
    Precision@K  (relevant chunks in the top K) / K. The denominator is always K. When a question has
                 few relevant chunks this cannot reach 1.0 for large K, so read it next to Recall@K.
    Recall@K     (distinct expected UNITS covered by the top K) / (number of expected units).
                 Unit = page (page level) or chunk id (chunk level). Several chunks from the same page
                 count once, so a multi-passage question is only fully recalled when every expected page
                 is covered.
    FullRecall@K 1 if Recall@K == 1 (all expected units found)
    RR / MRR     1 / (rank of the first relevant chunk within the first K), 0 if none; MRR = mean over questions

Abstention outcomes (per question): see ``abstention_outcome``.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

from rafeeq.knowledge.evaluation.ground_truth import squash


# ------------------------------------------------------------------ relevance

def relevance_flags(
    retrieved: Sequence[Mapping[str, Any]],
    file_name: str,
    expected_pages: Iterable[int] = (),
    expected_chunk_ids: Iterable[str] = (),
    *,
    level: str = "page",
) -> list[bool]:
    """One flag per retrieved chunk (rank order): is it relevant at ``level``?"""
    if level not in ("page", "chunk"):
        raise ValueError(f"level must be 'page' or 'chunk', got {level!r}")
    pages, ids = set(expected_pages), set(expected_chunk_ids)
    if level == "page" and not pages:
        raise ValueError("page-level relevance needs at least one expected page")
    if level == "chunk" and not ids:
        raise ValueError("chunk-level relevance needs at least one expected chunk id")
    flags = []
    for chunk in retrieved:
        same_file = chunk["file_name"] == file_name
        flags.append(same_file and (chunk["page"] in pages if level == "page" else chunk["chunk_id"] in ids))
    return flags


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    _check_k(k)
    return 1.0 if any(flags[:k]) else 0.0


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    _check_k(k)
    return sum(1 for f in flags[:k] if f) / k


def reciprocal_rank(flags: Sequence[bool], k: int | None = None) -> float:
    if k is not None:
        _check_k(k)
    for rank, flag in enumerate(flags[:k] if k is not None else flags, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def first_relevant_rank(flags: Sequence[bool]) -> int | None:
    for rank, flag in enumerate(flags, start=1):
        if flag:
            return rank
    return None


def recall_at_k(
    retrieved_units: Sequence[Any], expected_units: Iterable[Any], k: int
) -> float:
    """Fraction of distinct expected units appearing among the first ``k`` retrieved units."""
    _check_k(k)
    expected = set(expected_units)
    if not expected:
        raise ValueError("recall needs at least one expected unit")
    return len(expected & set(retrieved_units[:k])) / len(expected)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; None when n == 0.

    With ~30 questions these intervals are wide (a Hit@5 of 0.80 is compatible with roughly 0.63-0.91),
    which is why differences of a few points between two runs are not evidence of improvement.
    """
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ----------------------------------------------------------- abstention / RAG

def abstention_outcome(answerable: bool, grounded: bool) -> str:
    """Classify one answer attempt.

    answered           answerable question, the system answered (correctness is judged separately)
    false_abstention   answerable question, the system declined
    correct_abstention out-of-scope question, the system declined
    unsupported_answer out-of-scope question, the system answered anyway (the failure we most want to avoid)
    """
    if answerable:
        return "answered" if grounded else "false_abstention"
    return "unsupported_answer" if grounded else "correct_abstention"


def citation_scores(
    citations: Sequence[Mapping[str, Any]], file_name: str, expected_pages: Iterable[int]
) -> dict[str, Any]:
    """Citation correctness for one answer, judged against the verified pages.

    precision       cited (file, page) pairs that are expected / all cited pairs (None if nothing cited)
    recall          expected pages that were cited / expected pages
    hit             1.0 if at least one citation is a correct (file, page)
    wrong_document  number of citations pointing at a different document
    """
    expected = set(expected_pages)
    if not expected:
        raise ValueError("citation scoring needs at least one expected page")
    cited = {(c["file_name"], c["page"]) for c in citations}
    correct = {(f, p) for f, p in cited if f == file_name and p in expected}
    return {
        "n_citations": len(cited),
        "precision": (len(correct) / len(cited)) if cited else None,
        "recall": len({p for _, p in correct}) / len(expected),
        "hit": 1.0 if correct else 0.0,
        "wrong_document": sum(1 for f, _ in cited if f != file_name),
    }


SOURCES_SEPARATOR = "\n\n**Sources:**\n"  # exactly what Answerer.answer() appends
_MARKERS = re.compile(r"\[\d+\]")


def answer_body(answer_md: str) -> str:
    """The answer text without the generated 'Sources:' footer and without [n] markers."""
    body = answer_md.rsplit(SOURCES_SEPARATOR, 1)[0] if SOURCES_SEPARATOR in answer_md else answer_md
    return _MARKERS.sub("", body).strip()


def key_fact_coverage(answer_text: str, key_facts: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Deterministic correctness PROXY: how many required facts appear in the answer.

    A fact is covered when ANY of its listed spellings occurs in the answer (letters/digits only,
    case-, diacritic- and hyphen-insensitive). This checks that the expected content is present; it
    cannot detect contradictions or padding, so it is a lower-bound proxy, not a correctness judge.
    """
    if not key_facts:
        raise ValueError("key_fact_coverage needs at least one fact")
    haystack = squash(answer_text)
    missing = [list(alts) for alts in key_facts if not any(squash(a) and squash(a) in haystack for a in alts)]
    total = len(key_facts)
    return {"covered": total - len(missing), "total": total, "coverage": (total - len(missing)) / total, "missing": missing}
"""Benchmark dataset: one JSON object per line (JSONL), strictly validated.

Fields
------
id                  unique, e.g. "q01"
language            "en" | "ar"
qtype               "definition" | "explanation" | "comparison" | "multi_passage" | "out_of_scope"
question            the text sent to the system
answerable          False only for out-of-scope questions (qtype must then be "out_of_scope")
status              "draft"    ground truth NOT established; the item is never scored
                    "verified" every evidence quote was found in the stored chunks of the
                               expected pages (scripts/validate_benchmark.py sets this)
file_name           document the answer lives in, e.g. "hands_on_llm.pdf" (answerable items)
expected_pages      physical PDF pages (1-indexed, as stored in the chunk metadata) that
                    contain the supporting passages. Multi-passage items list every page.
expected_chunk_ids  chunk ids that contain a whole evidence quote (derived by the validator;
                    empty when a quote straddles a chunk boundary: page-level only)
evidence_quotes     short VERBATIM excerpts of the source text supporting the answer; the
                    validator searches the stored chunks for them. This is what makes the
                    ground truth checkable instead of trusted.
reference_answer    a short answer written from the evidence (needed to judge correctness)
key_facts           [[alt, alt], ...]: every inner list is one fact, any listed spelling
                    counts (use Arabic and English spellings for Arabic questions)
absent_terms        out-of-scope items: terms that must NOT occur anywhere in the store
notes               free text

Nothing here talks to FAISS, a model or the network.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

LANGUAGES = ("en", "ar")
QTYPES = ("definition", "explanation", "comparison", "multi_passage", "out_of_scope")
STATUSES = ("draft", "verified")
MIN_QUOTE_CHARS = 25  # shorter quotes match too easily to prove anything


class BenchmarkError(ValueError):
    """The benchmark file is malformed."""


@dataclass
class BenchmarkItem:
    id: str
    language: str
    qtype: str
    question: str
    answerable: bool
    status: str = "draft"
    file_name: str | None = None
    expected_pages: list[int] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    evidence_quotes: list[str] = field(default_factory=list)
    reference_answer: str | None = None
    key_facts: list[list[str]] = field(default_factory=list)
    absent_terms: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "BenchmarkItem":
        if not isinstance(data, dict):
            raise BenchmarkError(f"expected a JSON object, got {type(data).__name__}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise BenchmarkError(f"unknown field(s): {', '.join(unknown)} (typo?)")
        required = ("id", "language", "qtype", "question", "answerable")
        missing = [name for name in required if name not in data]
        if missing:
            raise BenchmarkError(f"missing required field(s): {', '.join(missing)}")
        return cls(**data)


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def item_problems(item: BenchmarkItem) -> list[str]:
    """Everything wrong with ``item``; empty list = valid. Verified items are held to more."""
    problems: list[str] = []
    if not _is_nonempty_str(item.id):
        problems.append("id must be a non-empty string")
    if item.language not in LANGUAGES:
        problems.append(f"language must be one of {LANGUAGES}, got {item.language!r}")
    if item.qtype not in QTYPES:
        problems.append(f"qtype must be one of {QTYPES}, got {item.qtype!r}")
    if not _is_nonempty_str(item.question):
        problems.append("question must be a non-empty string")
    if not isinstance(item.answerable, bool):
        problems.append("answerable must be true or false")
    if item.status not in STATUSES:
        problems.append(f"status must be one of {STATUSES}, got {item.status!r}")
    if isinstance(item.answerable, bool) and item.qtype in QTYPES and (item.qtype == "out_of_scope") == item.answerable:
        problems.append('qtype "out_of_scope" must go with answerable=false (and only then)')

    pages = item.expected_pages
    if not isinstance(pages, list) or any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in pages):
        problems.append("expected_pages must be a list of integers >= 1")
    elif len(set(pages)) != len(pages):
        problems.append("expected_pages contains duplicates")
    if not isinstance(item.expected_chunk_ids, list) or any(not _is_nonempty_str(c) for c in item.expected_chunk_ids):
        problems.append("expected_chunk_ids must be a list of non-empty strings")
    if not isinstance(item.evidence_quotes, list) or any(not _is_nonempty_str(q) for q in item.evidence_quotes):
        problems.append("evidence_quotes must be a list of non-empty strings")
    else:
        short = [q for q in item.evidence_quotes if len(q.strip()) < MIN_QUOTE_CHARS]
        if short:
            problems.append(f"evidence quotes must be at least {MIN_QUOTE_CHARS} characters: {short[0]!r}")
    facts = item.key_facts
    if not isinstance(facts, list) or any(
        not isinstance(f, list) or not f or any(not _is_nonempty_str(s) for s in f) for f in facts
    ):
        problems.append("key_facts must be a list of non-empty lists of non-empty strings")
    if not isinstance(item.absent_terms, list) or any(not _is_nonempty_str(t) for t in item.absent_terms):
        problems.append("absent_terms must be a list of non-empty strings")

    if item.answerable is True and not _is_nonempty_str(item.file_name):
        problems.append("answerable items need file_name")
    if item.answerable is False:
        for name in ("expected_pages", "expected_chunk_ids", "evidence_quotes", "key_facts"):
            if getattr(item, name):
                problems.append(f"out-of-scope items must not have {name}")
        if item.reference_answer:
            problems.append("out-of-scope items must not have a reference_answer")

    if item.status == "verified":
        if item.answerable is True:
            if not item.expected_pages:
                problems.append("verified items need expected_pages")
            if not item.evidence_quotes:
                problems.append("verified items need evidence_quotes")
            if not _is_nonempty_str(item.reference_answer):
                problems.append("verified items need a reference_answer")
            if not item.key_facts:
                problems.append("verified items need key_facts")
            if item.qtype == "multi_passage" and len(item.expected_pages) < 2:
                problems.append("multi_passage items need at least 2 expected_pages")
        elif item.answerable is False and not item.absent_terms:
            problems.append("verified out-of-scope items need absent_terms")
    return problems


def parse_benchmark(lines: list[str], source: str = "<memory>") -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    errors: list[str] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = BenchmarkItem.from_dict(json.loads(line))
        except json.JSONDecodeError as exc:
            errors.append(f"{source}:{number}: invalid JSON ({exc})")
            continue
        except (BenchmarkError, TypeError) as exc:
            errors.append(f"{source}:{number}: {exc}")
            continue
        for problem in item_problems(item):
            errors.append(f"{source}:{number}: [{item.id}] {problem}")
        if item.id in seen:
            errors.append(f"{source}:{number}: [{item.id}] duplicate id (first on line {seen[item.id]})")
        seen[item.id] = number
        items.append(item)
    if errors:
        raise BenchmarkError("invalid benchmark:\n  " + "\n  ".join(errors))
    if not items:
        raise BenchmarkError(f"{source} contains no items")
    return items


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BenchmarkError(f"cannot read benchmark {path}: {exc}") from exc
    return parse_benchmark(lines, str(path))


def dump_benchmark(items: list[BenchmarkItem], path: str | Path) -> None:
    text = "\n".join(json.dumps(item.to_dict(), ensure_ascii=False) for item in items) + "\n"
    Path(path).write_text(text, encoding="utf-8")
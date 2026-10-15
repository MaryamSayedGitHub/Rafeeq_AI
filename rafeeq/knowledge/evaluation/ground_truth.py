"""Establish and verify benchmark ground truth from the chunks that are ACTUALLY stored.

Why quotes: expected pages are never trusted from memory. Every answerable item carries short
verbatim ``evidence_quotes``; this module looks them up in the saved chunk texts
(``metadata.json``, a public file of the store) and derives / checks ``expected_pages`` and
``expected_chunk_ids`` from where they are really found.

Matching is deliberately forgiving about extraction noise but not about content: both sides are
Unicode-normalised (ligatures, Arabic diacritics/alef forms), case-folded, and reduced to letters
and digits only, so "self-attention", "Self attention" and a line-broken "self-\\nattention" match.

Pages are the PHYSICAL PDF pages stored in the chunk metadata (1-indexed), not the page numbers
printed on the book's pages; the two usually differ by the front matter.

Reads the store files only; never writes to them.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rafeeq.knowledge.evaluation.dataset import BenchmarkItem
from rafeeq.knowledge.store_persistence import METADATA_FILE_NAME, PersistenceError

_ARABIC_MARKS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")  # diacritics + tatweel
_ALEF_FORMS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
_STOP = {"the", "and", "for", "with", "what", "how", "does", "are", "this", "that", "from", "into", "between", "about", "which", "can", "its", "use", "used", "using"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _ARABIC_MARKS.sub("", text).translate(_ALEF_FORMS)
    return " ".join(text.split())


def squash(text: str) -> str:
    """Letters and digits only, after normalisation. The comparison form for quote matching."""
    return "".join(ch for ch in normalize_text(text) if ch.isalnum())


# ---------------------------------------------------------------- store chunks

def load_store_chunks(directory: str | Path) -> list[dict[str, Any]]:
    """All stored chunk records (text + metadata) from ``<directory>/metadata.json``."""
    meta_path = Path(directory) / METADATA_FILE_NAME
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PersistenceError(f"cannot read {meta_path}: {exc}") from exc
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise PersistenceError(f"{meta_path} has no valid 'records' list")
    required = ("chunk_id", "file_name", "page", "text")
    for i, record in enumerate(records):
        missing = [name for name in required if name not in record]
        if missing:
            raise PersistenceError(f"{meta_path} record {i} lacks {', '.join(missing)}")
    return records


def _overlap(a: list[str], b: list[str]) -> int:
    for k in range(min(len(a), len(b)), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def merge_overlapping(texts: Iterable[str]) -> str:
    """Rebuild running text from consecutive overlapping chunks (drops the repeated overlap)."""
    words: list[str] = []
    for text in texts:
        chunk_words = text.split()
        words.extend(chunk_words[_overlap(words, chunk_words):] if words else chunk_words)
    return " ".join(words)


@dataclass
class PageIndex:
    """Chunks grouped by (file_name, page), with the page text rebuilt from its chunks."""

    chunks: dict[tuple[str, int], list[dict[str, Any]]] = field(default_factory=dict)
    squashed_pages: dict[tuple[str, int], str] = field(default_factory=dict)
    squashed_chunks: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, records: list[dict[str, Any]]) -> "PageIndex":
        index = cls()
        for record in records:
            index.chunks.setdefault((record["file_name"], record["page"]), []).append(record)
        for key, group in index.chunks.items():
            group.sort(key=lambda r: r["chunk_id"])  # ids end in -cNNN, zero padded
            index.squashed_pages[key] = squash(merge_overlapping(r["text"] for r in group))
            for record in group:
                index.squashed_chunks[record["chunk_id"]] = squash(record["text"])
        return index

    @property
    def files(self) -> list[str]:
        return sorted({file for file, _ in self.chunks})


def find_quote(index: PageIndex, file_name: str, quote: str) -> dict[int, list[str]]:
    """``{page: [chunk ids containing the whole quote]}`` for every page of ``file_name`` holding it.

    An empty list for a page means the quote is on that page but straddles chunks (page-level only).
    """
    needle = squash(quote)
    if not needle:
        return {}
    found: dict[int, list[str]] = {}
    for (file, page), text in index.squashed_pages.items():
        if file == file_name and needle in text:
            found[page] = [r["chunk_id"] for r in index.chunks[(file, page)] if needle in index.squashed_chunks[r["chunk_id"]]]
    return dict(sorted(found.items()))


# ---------------------------------------------------------------- item checks

@dataclass
class Resolution:
    item_id: str
    pages: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def resolve_item(item: BenchmarkItem, index: PageIndex) -> Resolution:
    """Check an answerable item's quotes against the store and derive its pages / chunk ids."""
    result = Resolution(item.id)
    if not item.answerable:
        return result
    if item.file_name not in index.files:
        result.problems.append(f"file {item.file_name!r} is not in the store (store has: {', '.join(index.files) or 'nothing'})")
        return result
    if not item.evidence_quotes:
        result.problems.append("no evidence_quotes to verify")
        return result

    declared = set(item.expected_pages)
    supported: set[int] = set()
    chunk_ids: list[str] = []
    for quote in item.evidence_quotes:
        where = find_quote(index, item.file_name, quote)
        if not where:
            result.problems.append(f"quote not found in {item.file_name}: {quote[:70]!r}")
            continue
        if declared:
            matching = {p: ids for p, ids in where.items() if p in declared}
            if not matching:
                result.problems.append(
                    f"quote {quote[:50]!r} is on page(s) {sorted(where)} but expected_pages are {sorted(declared)}"
                )
                continue
        else:
            matching = where
            if len(where) > 1:
                result.problems.append(
                    f"quote {quote[:50]!r} appears on several pages {sorted(where)}; declare expected_pages to choose"
                )
                continue
        for page, ids in matching.items():
            supported.add(page)
            chunk_ids.extend(i for i in ids if i not in chunk_ids)
    if declared and declared - supported:
        result.problems.append(f"expected page(s) {sorted(declared - supported)} are not supported by any quote")
    result.pages = sorted(supported)
    result.chunk_ids = chunk_ids
    return result


def present_absent_terms(index: PageIndex, terms: list[str]) -> dict[str, int]:
    """For out-of-scope items: ``{term: number of pages containing it}`` for terms that DO occur."""
    hits: dict[str, int] = {}
    for term in terms:
        needle = squash(term)
        count = sum(1 for text in index.squashed_pages.values() if needle and needle in text)
        if count:
            hits[term] = count
    return hits


# ------------------------------------------------------------- lexical search

def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", normalize_text(text)) if len(t) >= 3 and t not in _STOP]


def lexical_search(
    records: list[dict[str, Any]], query: str, *, file_name: str | None = None, top: int = 8
) -> list[dict[str, Any]]:
    """Plain keyword search over the stored chunks, to help a human find evidence pages.

    No model and no embeddings: this is a labelling aid, not a retriever. It cannot match across
    languages, so search with the English terms the book uses.
    """
    pool = [r for r in records if file_name is None or r["file_name"] == file_name]
    terms = sorted(set(_tokens(query)))
    if not terms or not pool:
        return []
    normalized = [normalize_text(r["text"]) for r in pool]
    df = {t: sum(1 for text in normalized if t in text) for t in terms}
    scored = []
    for record, text in zip(pool, normalized):
        score = sum(min(text.count(t), 3) * math.log(1 + len(pool) / (1 + df[t])) for t in terms if df[t])
        matched = [t for t in terms if t in text]
        if score > 0:
            first = min(text.find(t) for t in matched)
            snippet = text[max(0, first - 80): first + 160]
            scored.append({"score": round(score, 3), "chunk_id": record["chunk_id"], "file_name": record["file_name"],
                           "page": record["page"], "matched": matched, "snippet": snippet})
    scored.sort(key=lambda r: (-r["score"], r["page"], r["chunk_id"]))
    return scored[:top]
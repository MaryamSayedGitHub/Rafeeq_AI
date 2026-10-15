"""Scanned-page hook (the M4 / Document Vision contract). STUB ONLY.

The pipeline asks two questions per page:

    has_usable_text(page_text)?  yes -> use the parser's text
                                 no  -> extract_pages(file_path, [page, ...])

``extract_pages`` is the contract M4 will implement (render page -> vision /
OCR -> text). Here it is a stub that raises ``ScannedPagesNotAvailableError``;
the pipeline catches that, keeps whatever text the parser found, and reports
the affected pages in its statistics. There is no VLM, OCR or image
dependency in this module.

Contract for a real implementation
----------------------------------
    extract_pages(file_path, pages) -> [{"page": 12, "text": "..."}, ...]

- ``pages``: 1-indexed page numbers that need extraction.
- Return one dict per page it could process; ``page`` MUST be one of the
  requested pages and ``text`` a string. A page that is omitted, or returned
  with blank text, simply keeps the parser's text.
- Extra keys (``block_type``, ``bbox``, ``blocks`` ...) are allowed and
  ignored for now; they are reserved for the later block-level pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_MIN_TEXT_CHARS = 20  # letters/digits needed before a page counts as "has text"

ScannedPageExtractor = Callable[[str, Sequence[int]], list[dict[str, Any]]]


class ScannedPagesNotAvailableError(NotImplementedError):
    """Raised by the stub: no scanned-page extractor is installed yet."""


def has_usable_text(text: str, min_chars: int = DEFAULT_MIN_TEXT_CHARS) -> bool:
    """True if ``text`` has at least ``min_chars`` letters/digits (any script)."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    return sum(1 for ch in text if ch.isalnum()) >= min_chars


def find_pages_needing_extraction(
    pages: Sequence[Mapping[str, Any]], min_chars: int = DEFAULT_MIN_TEXT_CHARS
) -> list[int]:
    """Page numbers (in order) whose parser text is not usable."""
    return [p["page"] for p in pages if not has_usable_text(p["text"], min_chars)]


def extract_pages(file_path: str | Path, pages: Sequence[int]) -> list[dict[str, Any]]:
    """M4 contract (stub): extract text for scanned ``pages`` of ``file_path``."""
    raise ScannedPagesNotAvailableError(
        "scanned-page extraction (Document Vision / M4) is not implemented yet; "
        f"{len(pages)} page(s) of {Path(file_path).name} keep their parser text"
    )


def merge_extracted_pages(
    pages: Sequence[Mapping[str, Any]],
    extracted: Sequence[Mapping[str, Any]],
    requested: Sequence[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Validate an extractor's result and merge it into ``pages``.

    Returns ``(new_pages, replaced_page_numbers)``. Inputs are not mutated.
    Blank extracted text is ignored (the parser text stays).
    """
    if not isinstance(extracted, (list, tuple)):
        raise TypeError(f"extractor must return a list, got {type(extracted).__name__}")
    allowed = set(requested)
    replacements: dict[int, str] = {}
    for i, entry in enumerate(extracted):
        if not isinstance(entry, Mapping) or "page" not in entry or "text" not in entry:
            raise ValueError(f"extractor result[{i}] must be a dict with 'page' and 'text'")
        page, text = entry["page"], entry["text"]
        if isinstance(page, bool) or not isinstance(page, int) or page not in allowed:
            raise ValueError(f"extractor returned page {page!r}, which was not requested")
        if not isinstance(text, str):
            raise ValueError(f"extractor result for page {page} has non-string text")
        if page in replacements:
            raise ValueError(f"extractor returned page {page} more than once")
        if text.strip():
            replacements[page] = text.strip()
    merged = [{**p, "text": replacements.get(p["page"], p["text"])} for p in pages]
    return merged, sorted(replacements)
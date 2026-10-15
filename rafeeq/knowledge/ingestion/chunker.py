"""Page-aware word chunker.

Takes the parser output (one dict per page) and produces chunks that NEVER
cross a page boundary. Each chunk carries the metadata M2 needs for room
isolation and page citations.
"""

from __future__ import annotations

import hashlib
import re

DEFAULT_CHUNK_SIZE = 500  # words
DEFAULT_OVERLAP = 80  # words
BLOCK_TYPE_TEXT = "text"

_WORD_RE = re.compile(r"\S+")


def _validate_params(chunk_size: int, overlap: int, room_id: str, file_name: str) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError(f"chunk_size must be an integer, got {chunk_size!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError(f"overlap must be an integer, got {overlap!r}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap must be smaller than chunk_size (overlap={overlap}, chunk_size={chunk_size})"
        )
    if not isinstance(room_id, str) or not room_id.strip():
        raise ValueError("room_id must be a non-empty string")
    if not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string")


def _validate_pages(pages: list[dict]) -> None:
    seen: set[int] = set()
    for position, entry in enumerate(pages):
        if not isinstance(entry, dict) or "page" not in entry or "text" not in entry:
            raise ValueError(
                f"pages[{position}] must be a dict with 'page' and 'text' keys, got {entry!r}"
            )
        page = entry["page"]
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(f"pages[{position}]['page'] must be an integer >= 1, got {page!r}")
        if not isinstance(entry["text"], str):
            raise ValueError(f"pages[{position}]['text'] must be a string for page {page}")
        if page in seen:
            raise ValueError(f"duplicate page number {page} in pages")
        seen.add(page)


def split_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into windows of at most ``chunk_size`` words.

    Consecutive windows share ``overlap`` words. A text with <= chunk_size
    words is returned as a single window. Original whitespace (including
    newlines) inside a window is preserved; only the ends are stripped.

    This is the one function to replace later (e.g. sentence-aware or
    token-based splitting) without touching the page/metadata logic.
    """
    spans = [m.span() for m in _WORD_RE.finditer(text)]
    total = len(spans)
    if total == 0:
        return []

    step = chunk_size - overlap
    windows: list[str] = []
    start = 0
    while True:
        end = min(start + chunk_size, total)
        windows.append(text[spans[start][0] : spans[end - 1][1]])
        if end >= total:
            break
        start += step
    return windows


def _id_prefix(room_id: str, file_name: str) -> str:
    digest = hashlib.sha1(f"{room_id}\x00{file_name}".encode("utf-8")).hexdigest()
    return digest[:12]


def make_chunk_id(room_id: str, file_name: str, page: int, index: int) -> str:
    """Deterministic ID: <hash(room, file)>-p<page>-c<index-on-page>."""
    return f"{_id_prefix(room_id, file_name)}-p{page:04d}-c{index:03d}"


def chunk_pages(
    pages: list[dict],
    room_id: str,
    file_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """Chunk parser output page by page.

    Args:
        pages: ``[{"page": 1, "text": "..."}, ...]`` as returned by ``parse_pdf``.
        room_id: room every chunk belongs to (required, non-empty).
        file_name: source file name (required, non-empty).
        chunk_size: max words per chunk.
        overlap: words shared by consecutive chunks of the SAME page.

    Returns:
        A list of chunks, in page order, each exactly::

            {"chunk_id", "room_id", "file_name", "page", "block_type", "text"}

    Rules:
        - a chunk never contains text from more than one page
        - a page with <= chunk_size words is one chunk (never merged with a neighbour)
        - empty / whitespace-only pages produce no chunks
        - chunk_id is deterministic for the same room_id, file_name, page and position

    Raises:
        ValueError: invalid chunk_size/overlap, empty room_id/file_name,
            malformed page entries, or duplicate page numbers.
    """
    _validate_params(chunk_size, overlap, room_id, file_name)
    _validate_pages(pages)

    chunks: list[dict] = []
    for entry in pages:
        page = entry["page"]
        for index, text in enumerate(split_words(entry["text"], chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": make_chunk_id(room_id, file_name, page, index),
                    "room_id": room_id,
                    "file_name": file_name,
                    "page": page,
                    "block_type": BLOCK_TYPE_TEXT,
                    "text": text,
                }
            )
    return chunks
"""Ingestion pipeline: orchestration only.

    parse -> scanned-page hook (when needed) -> chunk -> tag topic
          -> embed -> FAISS add -> (optional) save

Each step is delegated to the component that owns it (parser, chunker,
topic manager, embedder, vector store); nothing is re-implemented here and
the embedder / store are injected, so tests can use fakes and no model is
loaded unless the caller passes one.

Guarantees
----------
- Chunks never cross a page boundary (the chunker works page by page).
- Every chunk has ``room_id`` and ``topic_id``.
- ``embeddings[i]`` belongs to ``chunks[i]``: the texts sent to the embedder
  are exactly the chunk texts, in order, and the count is checked.
- If embedding or validation fails, the store is left untouched
  (``vector_store.add`` is the last step before saving and is all-or-nothing).
- Ingesting the same file into the same room twice raises
  ``DuplicateChunkError`` (the store is append-only).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from rafeeq.knowledge.ingestion.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_pages
from rafeeq.knowledge.ingestion.scanned_pages import (
    DEFAULT_MIN_TEXT_CHARS,
    ScannedPageExtractor,
    extract_pages,
    find_pages_needing_extraction,
    merge_extracted_pages,
)
from rafeeq.knowledge.topics.topic_manager import TopicManager, get_default_manager


class NoTextExtractedError(ValueError):
    """The file produced no text at all (e.g. a fully scanned PDF and no extractor)."""


class EmbedderLike(Protocol):
    def embed_documents(self, texts: list[str]) -> Any: ...


class VectorStoreLike(Protocol):
    def add(self, embeddings: Any, chunks: list[dict[str, Any]]) -> int: ...
    def save(self, directory: Any) -> None: ...


@dataclass(frozen=True)
class IngestionStats:
    room_id: str
    file_name: str
    document_id: str
    total_pages: int
    pages_with_text: int  # pages whose parser text was usable
    pages_needing_extraction: list[int]  # sent to the scanned-page hook
    pages_extracted: list[int]  # successfully replaced by the hook
    pages_unextracted: list[int]  # needed extraction but the hook gave nothing (parser text kept)
    chunk_count: int
    topic_counts: dict[str, int] = field(default_factory=dict)
    vectors_added: int = 0
    vectors_in_store: int | None = None
    saved_to: str | None = None


def ingest_file(
    file_path: str | Path,
    room_id: str,
    vector_store: VectorStoreLike,
    embedder: EmbedderLike,
    *,
    document_id: str | None = None,
    document_type: str | None = None,
    file_name: str | None = None,
    topic_manager: TopicManager | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    scanned_page_extractor: ScannedPageExtractor | None = extract_pages,
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    save_dir: str | Path | None = None,
    parse: Callable[[str], list[dict[str, Any]]] | None = None,
    retag: Callable[[list[dict[str, Any]], Any], list[dict[str, Any]]] | None = None,
) -> IngestionStats:
    """Ingest one PDF into ``vector_store`` under ``room_id``.

    Args:
        file_path: the PDF to ingest.
        room_id: room the chunks belong to.
        vector_store: object with ``add(embeddings, chunks)`` and ``save(dir)``.
        embedder: object with ``embed_documents(texts)``; passed in so this
            function never loads a model itself.
        document_id: defaults to the file name without extension.
        file_name: name stored on every chunk (and shown in citations); defaults to the
            file's own name. Rafeeq passes the uploaded name, because the stored copy is
            called ``<file_id>__<name>``.
        document_type: optional, e.g. ``"book"`` / ``"lecture"``.
        topic_manager: defaults to the shared deterministic manager.
        scanned_page_extractor: the M4 hook; ``None`` disables it. The default
            stub raises ``NotImplementedError``, which is handled: those pages
            keep their parser text and are reported in the stats.
        save_dir: if given, ``vector_store.save(save_dir)`` runs after adding.
        parse: override the parser (mainly for tests); defaults to ``parse_pdf``.
        retag: optional ``(chunks, embeddings) -> chunks`` run after embedding, e.g. to give the
            chunks the room's learned topics instead of the keyword tags (see topics/room_topics.py).

    Raises:
        NoTextExtractedError: nothing to index after parsing / extraction.
        DuplicateChunkError, ValueError, ...: propagated from the components.
    """
    if not isinstance(room_id, str) or not room_id.strip():
        raise ValueError("room_id must be a non-empty string")

    path = Path(file_path)
    file_name = file_name or path.name
    document_id = document_id or Path(file_name).stem
    manager = topic_manager or get_default_manager()

    # 1. parse
    if parse is None:
        from rafeeq.knowledge.ingestion.parser import parse_pdf  # lazy: only needs PyMuPDF when used

        parse = parse_pdf
    pages = [dict(p) for p in parse(str(file_path))]

    # 2. scanned-page hook
    needing = find_pages_needing_extraction(pages, min_text_chars)
    extracted: list[int] = []
    if needing and scanned_page_extractor is not None:
        try:
            result = scanned_page_extractor(str(file_path), list(needing))
        except NotImplementedError:
            result = []  # extractor not available yet: keep the parser text
        pages, extracted = merge_extracted_pages(pages, result, needing)
    unextracted = [p for p in needing if p not in extracted]

    # 3. chunk (page-aware; the chunker is not touched by topics or documents)
    chunks = chunk_pages(pages, room_id, file_name, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise NoTextExtractedError(
            f"no text could be extracted from {file_name}; "
            f"{len(needing)} page(s) have no usable text layer and no scanned-page extractor produced any"
        )

    # 4. document info + topic tagging (after chunking)
    extra: dict[str, Any] = {"document_id": document_id}
    if document_type is not None:
        extra["document_type"] = document_type
    chunks = manager.tag_chunks(room_id, [{**chunk, **extra} for chunk in chunks])

    # 5. embed: one vector per chunk, same order
    embeddings = embedder.embed_documents([chunk["text"] for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embedder returned {len(embeddings)} vectors for {len(chunks)} chunks; refusing to index"
        )

    # 5b. optional retagging with the vectors we already have (no second embedding pass)
    if retag is not None:
        chunks = retag(chunks, embeddings)

    # 6. add (all-or-nothing) and 7. save
    added = vector_store.add(embeddings, chunks)
    if save_dir is not None:
        vector_store.save(save_dir)

    try:
        total = len(vector_store)  # type: ignore[arg-type]
    except TypeError:
        total = None

    return IngestionStats(
        room_id=room_id,
        file_name=file_name,
        document_id=document_id,
        total_pages=len(pages),
        pages_with_text=len(pages) - len(needing),
        pages_needing_extraction=needing,
        pages_extracted=extracted,
        pages_unextracted=unextracted,
        chunk_count=len(chunks),
        topic_counts=dict(sorted(Counter(c["topic_id"] for c in chunks).items())),
        vectors_added=added,
        vectors_in_store=total,
        saved_to=str(save_dir) if save_dir is not None else None,
    )
#!/usr/bin/env python
"""Ingest one PDF into the persistent knowledge-base vector store.

    uv run python scripts/ingest_book.py                       # data/raw/hands_on_llm.pdf -> room "llm_course"
    uv run python scripts/ingest_book.py other.pdf --room-id course_B
    uv run python scripts/ingest_book.py new.pdf --document-id new_v2   # keep a revised copy next to the old one

What it does (all of it delegated to the existing components):
    load the saved store (or start an empty one ONLY if none exists)
    -> refuse a document that is already in the room (before any slow embedding)
    -> ingest_file(): parse -> [scanned-page hook] -> chunk -> tag topics -> embed (BGE-M3) -> add
    -> save index.faiss + metadata.json, reload them to prove they are intact
    -> print the statistics.

Exit status: 0 = ingested AND persisted, 1 = failed (message on stderr), 2 = bad arguments.
On failure the files on disk are left exactly as they were.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
try:
    import rafeeq.knowledge  # noqa: F401
except ImportError:  # run as a plain script without the package installed: fall back to ./src
    sys.path.insert(0, str(PROJECT_ROOT))

from rafeeq.knowledge.ingestion.chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP  # noqa: E402
from rafeeq.knowledge.ingestion.pipeline import NoTextExtractedError, ingest_file  # noqa: E402
from rafeeq.knowledge.store_persistence import (  # noqa: E402
    DEFAULT_RAW_PDF_SUBPATH,
    DEFAULT_ROOM_ID,
    DEFAULT_STORE_SUBDIR,
    DocumentAlreadyIngestedError,
    PersistenceError,
    document_exists,
    ensure_dimension,
    find_leftover_files,
    load_or_create_store,
    save_store_safely,
    store_inventory,
)

DEFAULT_PDF = PROJECT_ROOT / DEFAULT_RAW_PDF_SUBPATH
DEFAULT_STORE_DIR = PROJECT_ROOT / DEFAULT_STORE_SUBDIR
MAX_PAGES_LISTED = 20


class CliError(Exception):
    """A problem with the request itself (e.g. the PDF does not exist)."""


# ----------------------------------------------------------------- arguments

def _non_empty(text: str) -> str:
    if not text.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return text


def _positive_int(text: str) -> int:
    value = _integer(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = _integer(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def _integer(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest_book.py",
        description="Ingest one PDF into the persistent FAISS knowledge base (index.faiss + metadata.json).",
        epilog=(
            "The store is append-only: ingesting the same document into the same room twice is refused. "
            "To keep a revised copy use a new --document-id; to replace a document, move the store directory "
            "away (or use another --store-dir) and re-ingest all PDFs."
        ),
    )
    parser.add_argument("pdf", nargs="?", default=str(DEFAULT_PDF), help="PDF to ingest (default: %(default)s)")
    parser.add_argument("--room-id", type=_non_empty, default=DEFAULT_ROOM_ID, help="room the chunks belong to (default: %(default)s)")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="directory holding index.faiss and metadata.json (default: %(default)s)")
    parser.add_argument("--document-id", type=_non_empty, default=None, help="identity of the document inside the room (default: the PDF file name without extension)")
    parser.add_argument("--document-type", type=_non_empty, default=None, help='optional label, e.g. "book" or "lecture"')
    parser.add_argument("--chunk-size", type=_positive_int, default=DEFAULT_CHUNK_SIZE, help="words per chunk (default: %(default)s)")
    parser.add_argument("--overlap", type=_non_negative_int, default=DEFAULT_OVERLAP, help="overlapping words between chunks (default: %(default)s)")
    parser.add_argument("--device", default="cpu", help='embedding device, e.g. "cpu" or "cuda" (default: %(default)s)')
    parser.add_argument("--batch-size", type=_positive_int, default=8, help="embedding batch size (default: %(default)s)")
    parser.add_argument("--debug", action="store_true", help="print the full traceback on failure")
    return parser


# ------------------------------------------------------------------- workflow

def _make_embedder(args: argparse.Namespace) -> Any:
    from rafeeq.knowledge.embeddings.embedder import Embedder  # lazy: torch is only imported when used

    return Embedder(device=args.device, batch_size=args.batch_size)


def _format_pages(pages: list[int]) -> str:
    shown = ", ".join(str(p) for p in pages[:MAX_PAGES_LISTED])
    extra = len(pages) - MAX_PAGES_LISTED
    return shown + (f" ... and {extra} more" if extra > 0 else "")


def format_report(stats: Any, store_dir: Path, inventory: dict[str, dict[str, int]]) -> str:
    lines = [
        "Ingestion complete (index and metadata saved and re-loaded successfully)",
        f"  PDF:               {stats.file_name}",
        f"  Room ID:           {stats.room_id}",
        f"  Document ID:       {stats.document_id}",
        f"  Pages:             {stats.total_pages} ({stats.pages_with_text} with usable text, "
        f"{len(stats.pages_needing_extraction)} without)",
        f"  Chunks:            {stats.chunk_count}",
        "  Topics:",
    ]
    for topic, count in sorted(stats.topic_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"    {topic}: {count}")
    lines += [
        f"  Vectors added:     {stats.vectors_added}",
        f"  Vectors in store:  {stats.vectors_in_store}",
        f"  Persisted to:      {store_dir}  (index.faiss, metadata.json)",
        "Store contents:",
    ]
    for room, documents in inventory.items():
        lines.append(f"  room {room}")
        for document, count in documents.items():
            lines.append(f"    {document}: {count} chunks")
    if stats.pages_extracted:
        lines.append(f"Pages recovered by the scanned-page extractor: {_format_pages(stats.pages_extracted)}")
    if stats.pages_unextracted:
        lines.append(
            f"WARNING: {len(stats.pages_unextracted)} page(s) have no usable text layer and could NOT be extracted "
            f"(the scanned-page extractor is not installed yet); they keep whatever text the parser found "
            f"(possibly none): pages {_format_pages(stats.pages_unextracted)}"
        )
    return "\n".join(lines)


def run_ingest(
    args: argparse.Namespace,
    *,
    embedder: Any = None,
    store_cls: Any = None,
    parse: Callable[[str], list[dict[str, Any]]] | None = None,
    out: TextIO,
) -> None:
    """Do the whole job or raise. Prints the report only after the store is saved and verified."""
    pdf = Path(args.pdf)
    if not pdf.is_file():
        raise CliError(f"PDF not found (or not a file): {pdf}")
    store_dir = Path(args.store_dir)
    document_id = args.document_id or pdf.stem
    embedder = embedder if embedder is not None else _make_embedder(args)

    if store_cls is None:
        from rafeeq.knowledge.vector_store import FAISSVectorStore

        store_cls = FAISSVectorStore

    leftovers = find_leftover_files(store_dir)
    if leftovers:
        print(f"warning: files from an interrupted save found in {store_dir}: {', '.join(p.name for p in leftovers)}", file=out)

    # 1. load the existing store, or start empty only if there is none
    store, created = load_or_create_store(store_dir, lambda: embedder.dimension, store_cls=store_cls)
    if created:
        print(f"No saved store in {store_dir}; starting a new one.", file=out)
    else:
        print(f"Loaded existing store from {store_dir}: {len(store)} vectors.", file=out)
        # 2. cheap duplicate check BEFORE the (slow) model is loaded and the book embedded
        if document_exists(store, args.room_id, document_id):
            raise DocumentAlreadyIngestedError(
                f"document {document_id!r} is already in room {args.room_id!r} of the store in {store_dir}. "
                f"The store is append-only, so nothing was ingested and nothing was changed. "
                f"To keep a revised copy next to it use --document-id <new id>; to replace it, move {store_dir} away "
                f"(or use another --store-dir) and re-ingest all PDFs."
            )
        ensure_dimension(store, embedder.dimension, store_dir)

    # 3. the existing pipeline (parse, chunk, tag, embed, add). Saving is done below, safely.
    print(f"Ingesting {pdf.name} into room {args.room_id!r} (embedding runs on {args.device}; a full book can take a long time on CPU)...", file=out)
    stats = ingest_file(
        pdf,
        args.room_id,
        store,
        embedder,
        document_id=document_id,
        document_type=args.document_type,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        parse=parse,
    )

    # 4. persist + verify; raises (and restores the old files) if anything is wrong
    save_store_safely(store, store_dir, store_cls=store_cls)

    print(format_report(stats, store_dir, store_inventory(store_dir)), file=out)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, (CliError, PersistenceError, DocumentAlreadyIngestedError)):
        return str(exc)
    if isinstance(exc, NoTextExtractedError):
        return (
            f"{exc}. If this is a scanned PDF it needs the scanned-page extractor (M4), which is not "
            f"installed yet; nothing was stored."
        )
    if type(exc).__name__ == "DuplicateChunkError":
        return (
            f"{exc}. This document (or one with the same file name and chunk ids) is already in the room. "
            f"Nothing was stored. Use --document-id <new id> to keep both, or re-ingest into a fresh store."
        )
    return f"{type(exc).__name__}: {exc}"


def main(
    argv: list[str] | None = None,
    *,
    embedder: Any = None,
    store_cls: Any = None,
    parse: Callable[[str], list[dict[str, Any]]] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    args = build_parser().parse_args(argv)
    try:
        run_ingest(args, embedder=embedder, store_cls=store_cls, parse=parse, out=out)
    except KeyboardInterrupt:
        print("error: interrupted; the saved store was not changed.", file=err)
        return 130
    except Exception as exc:  # reported, never swallowed: nonzero exit status
        if args.debug:
            traceback.print_exc(file=err)
        print(f"error: {_describe(exc)}", file=err)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # never crash on a non-UTF-8 console
    sys.exit(main())
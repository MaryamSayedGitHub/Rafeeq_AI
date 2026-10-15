#!/usr/bin/env python
"""Query the persisted knowledge base from a fresh process (retrieval-after-restart check).

    uv run python scripts/query_book.py "What is self-attention?"
    uv run python scripts/query_book.py "يعني إيه Self-Attention؟" --room-id llm_course -k 3
    uv run python scripts/query_book.py --list            # what is stored (no model needed, instant)

It only READS the store: loads index.faiss + metadata.json, embeds the query with the
existing embedder, and searches the chosen room through the existing Retriever.

Exit status: 0 = at least one result printed, 1 = no store / empty store / no results / error,
2 = bad arguments.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
try:
    import rafeeq.knowledge  # noqa: F401
except ImportError:  # run as a plain script without the package installed: fall back to ./src
    sys.path.insert(0, str(PROJECT_ROOT))

from rafeeq.knowledge.retrieval.factory import DEFAULT_RETRIEVAL_MODE, RETRIEVAL_MODES, build_retriever  # noqa: E402
from rafeeq.knowledge.retrieval.retriever import DEFAULT_K  # noqa: E402
from rafeeq.knowledge.store_persistence import (  # noqa: E402
    DEFAULT_ROOM_ID,
    DEFAULT_STORE_SUBDIR,
    PersistenceError,
    StoreNotFoundError,
    ensure_dimension,
    load_existing_store,
    store_inventory,
)

DEFAULT_STORE_DIR = PROJECT_ROOT / DEFAULT_STORE_SUBDIR


class CliError(Exception):
    """The request cannot be answered (empty store, unknown room, ...)."""


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="query_book.py",
        description="Search the persisted FAISS knowledge base (read-only).",
    )
    parser.add_argument("query", nargs="?", help="the question to search for (Arabic or English)")
    parser.add_argument("--room-id", default=DEFAULT_ROOM_ID, help="room to search (default: %(default)s)")
    parser.add_argument("-k", "--top-k", type=_positive_int, default=DEFAULT_K, help="number of results (default: %(default)s)")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="directory holding index.faiss and metadata.json (default: %(default)s)")
    parser.add_argument("--max-chars", type=_positive_int, default=300, help="characters of each chunk to show (default: %(default)s)")
    parser.add_argument("--list", action="store_true", dest="list_only", help="only list the rooms and documents in the store (does not load the model)")
    parser.add_argument("--device", default="cpu", help='embedding device, e.g. "cpu" or "cuda" (default: %(default)s)')
    parser.add_argument("--batch-size", type=_positive_int, default=8, help="embedding batch size (default: %(default)s)")
    parser.add_argument("--retrieval-mode", choices=RETRIEVAL_MODES, default=DEFAULT_RETRIEVAL_MODE,
                        help="dense (default) or hybrid = dense + BM25 fused with Reciprocal Rank Fusion")
    parser.add_argument("--rerank", action="store_true", help="rerank the top 50 candidates with BAAI/bge-reranker-v2-m3 (default: off)")
    parser.add_argument("--rerank-pool", type=_positive_int, default=50, help="candidates handed to the reranker (default: %(default)s)")
    parser.add_argument("--rerank-device", default="auto", help="auto = cuda if available else cpu (default: %(default)s)")
    parser.add_argument("--exclude-non-content", action="store_true", help="skip index-like pages (default: off)")
    parser.add_argument("--debug", action="store_true", help="print the full traceback on failure")
    return parser


def _make_embedder(args: argparse.Namespace) -> Any:
    from rafeeq.knowledge.embeddings.embedder import Embedder  # lazy: torch is only imported when used

    return Embedder(device=args.device, batch_size=args.batch_size)


def _format_inventory(inventory: dict[str, dict[str, int]]) -> str:
    lines = []
    for room, documents in inventory.items():
        lines.append(f"room {room}")
        for document, count in documents.items():
            lines.append(f"  {document}: {count} chunks")
    return "\n".join(lines)


def _preview(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def run_query(args: argparse.Namespace, *, embedder: Any = None, store_cls: Any = None, out: TextIO) -> None:
    store_dir = Path(args.store_dir)
    try:
        store = load_existing_store(store_dir, store_cls=store_cls)
    except StoreNotFoundError as exc:
        raise CliError(f"{exc}. Run scripts/ingest_book.py first.") from exc
    if len(store) == 0:
        raise CliError(f"the store in {store_dir} is empty (0 vectors). Run scripts/ingest_book.py first.")

    inventory = store_inventory(store_dir)
    if args.list_only:
        print(f"Store {store_dir}: {len(store)} vectors, dimension {store.dimension}", file=out)
        print(_format_inventory(inventory), file=out)
        return

    embedder = embedder if embedder is not None else _make_embedder(args)
    ensure_dimension(store, embedder.dimension, store_dir)

    extra = {}
    if args.rerank:
        from rafeeq.knowledge.retrieval.reranker import CrossEncoderReranker

        extra = {"rerank": True, "rerank_pool": args.rerank_pool, "reranker": CrossEncoderReranker(device=args.rerank_device)}
    retriever = build_retriever(store, embedder, mode=args.retrieval_mode, exclude_non_content=args.exclude_non_content, **extra)
    results = retriever.retrieve(args.room_id, args.query, args.top_k)
    if not results:
        known = ", ".join(inventory) or "none"
        raise CliError(
            f"no results for room {args.room_id!r} in the store ({len(store)} vectors). "
            f"Rooms in this store: {known}. Check --room-id."
        )

    print(f"Query: {args.query}", file=out)
    print(f"Room:  {args.room_id}   Store: {store_dir} ({len(store)} vectors)", file=out)
    print(f"Top {len(results)} result(s):", file=out)
    for rank, hit in enumerate(results, start=1):
        fusion = f"  fusion={hit['fusion_score']:.4f}" if "fusion_score" in hit else ""
        fusion += f"  rerank={hit['rerank_score']:.3f}" if "rerank_score" in hit else ""
        print(
            f"{rank}. score={hit['score']:.4f}{fusion}  file={hit['file_name']}  page={hit['page']}  topic={hit['topic_id']}",
            file=out,
        )
        print(f"   {_preview(hit['text'], args.max_chars)}", file=out)


def main(
    argv: list[str] | None = None,
    *,
    embedder: Any = None,
    store_cls: Any = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.list_only and (args.query is None or not args.query.strip()):
        parser.error("a query is required (or use --list)")
    if not args.room_id.strip():
        parser.error("--room-id must not be empty")
    try:
        run_query(args, embedder=embedder, store_cls=store_cls, out=out)
    except KeyboardInterrupt:
        print("error: interrupted.", file=err)
        return 130
    except Exception as exc:  # reported, never swallowed: nonzero exit status
        if args.debug:
            traceback.print_exc(file=err)
        text = str(exc) if isinstance(exc, (CliError, PersistenceError)) else f"{type(exc).__name__}: {exc}"
        print(f"error: {text}", file=err)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")  # Arabic text must not crash a cp1252 console
    sys.exit(main())
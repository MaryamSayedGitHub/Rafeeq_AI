#!/usr/bin/env python
"""Find which stored pages mention some keywords (a labelling aid for the benchmark).

    uv run python scripts/find_evidence.py "key value cache" --file-name hands_on_llm.pdf

Plain keyword search over the chunk texts saved in data/vector_store/metadata.json. No model, no
embeddings, nothing is written. It cannot match across languages: search with the English terms the
book uses, read the snippets, then copy a short VERBATIM sentence into an item's "evidence_quotes".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
try:
    import rafeeq.knowledge  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))

from rafeeq.knowledge.evaluation.ground_truth import lexical_search, load_store_chunks  # noqa: E402
from rafeeq.knowledge.store_persistence import DEFAULT_STORE_SUBDIR, PersistenceError  # noqa: E402


def main(argv: list[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    parser = argparse.ArgumentParser(prog="find_evidence.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("query", help="keywords to look for (English terms from the book)")
    parser.add_argument("--store-dir", default=str(PROJECT_ROOT / DEFAULT_STORE_SUBDIR))
    parser.add_argument("--file-name", default=None, help="restrict to one document, e.g. hands_on_llm.pdf")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        records = load_store_chunks(args.store_dir)
    except PersistenceError as exc:
        print(f"error: {exc}", file=err)
        return 1
    hits = lexical_search(records, args.query, file_name=args.file_name, top=args.top)
    if not hits:
        print("no chunk contains those keywords", file=err)
        return 1
    for hit in hits:
        print(f"page {hit['page']:>4}  {hit['file_name']}  {hit['chunk_id']}  score={hit['score']}  matched={','.join(hit['matched'])}", file=out)
        print(f"          ...{hit['snippet']}...", file=out)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    sys.exit(main())
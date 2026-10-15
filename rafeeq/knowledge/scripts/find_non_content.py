#!/usr/bin/env python
"""List the stored pages that the retrieval filter would treat as non-content (book index pages).

    uv run python scripts/find_non_content.py

Audit tool: reads data/vector_store/metadata.json, applies the same test as Retriever(exclude_non_content=True)
and prints, per document, the flagged pages as ranges plus the first words of each range. Nothing is
written and no model is loaded. Read the list before turning the filter on for a new document: a flagged
page that is real teaching content means the heuristic does not fit that document.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
try:
    import rafeeq.knowledge  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))

from rafeeq.knowledge.evaluation.ground_truth import load_store_chunks  # noqa: E402
from rafeeq.knowledge.retrieval.content_filter import is_non_content_page  # noqa: E402
from rafeeq.knowledge.store_persistence import DEFAULT_STORE_SUBDIR, PersistenceError  # noqa: E402


def _ranges(pages: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for page in sorted(set(pages)):
        if out and page == out[-1][1] + 1:
            out[-1] = (out[-1][0], page)
        else:
            out.append((page, page))
    return out


def main(argv: list[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    parser = argparse.ArgumentParser(prog="find_non_content.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--store-dir", default=str(PROJECT_ROOT / DEFAULT_STORE_SUBDIR))
    parser.add_argument("--room-id", default=None, help="only this room (default: every room)")
    args = parser.parse_args(argv)
    try:
        records = load_store_chunks(args.store_dir)
    except PersistenceError as exc:
        print(f"error: {exc}", file=err)
        return 1
    totals: dict[tuple[str, str], int] = defaultdict(int)
    flagged: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for record in records:
        if args.room_id is not None and record.get("room_id") != args.room_id:
            continue
        key = (str(record.get("room_id")), record["file_name"])
        totals[key] += 1
        if is_non_content_page(record["text"]):
            flagged[key][int(record["page"])] = re.sub(r"\s+", " ", record["text"]).strip()
    if not totals:
        print("no chunks found (check --room-id)", file=err)
        return 1
    for key in sorted(totals):
        room, name = key
        pages = flagged.get(key, {})
        print(f"{name} (room {room}): {len(pages)} of {totals[key]} chunks flagged", file=out)
        for start, end in _ranges(list(pages)):
            label = f"page {start}" if start == end else f"pages {start}-{end}"
            print(f"    {label}: {pages[start][:90]}...", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
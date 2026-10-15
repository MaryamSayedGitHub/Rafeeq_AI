#!/usr/bin/env python
"""Check (and optionally promote) benchmark ground truth against the chunks that are actually stored.

    uv run python scripts/validate_benchmark.py                 # report only, writes nothing
    uv run python scripts/validate_benchmark.py --write         # promote items that fully verify

For every answerable item the evidence quotes must be found in the stored text of the expected file
(and of the declared expected_pages, if any). With --write, an item whose quotes all resolve and which
has everything a verified item needs (reference answer, key facts, >= 2 pages for multi_passage...)
gets expected_pages / expected_chunk_ids filled from where the quotes were really found and
status "verified". Out-of-scope items are verified when none of their absent_terms occurs anywhere in
the store. Nothing is ever promoted on a guess; failures stay "draft" with the reasons printed.

The benchmark file is rewritten only with --write (a .bak copy is kept). The vector store is only read.
Exit status: 0 = no verified item is broken, 1 = a verified item no longer checks out / bad input.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
try:
    import rafeeq.knowledge  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))

from rafeeq.knowledge.evaluation.dataset import BenchmarkError, dump_benchmark, item_problems, load_benchmark  # noqa: E402
from rafeeq.knowledge.evaluation.ground_truth import PageIndex, load_store_chunks, present_absent_terms, resolve_item  # noqa: E402
from rafeeq.knowledge.store_persistence import DEFAULT_STORE_SUBDIR, PersistenceError  # noqa: E402

DEFAULT_BENCHMARK = KB_DIR / "eval" / "benchmark" / "hands_on_llm_v1.jsonl"


def check_items(items, index):
    """Return ``(updated_items, report_lines, broken_verified_ids)`` without touching any file."""
    updated, lines, broken = [], [], []
    for item in items:
        if item.answerable:
            res = resolve_item(item, index)
            candidate = dataclasses.replace(item, expected_pages=res.pages or item.expected_pages,
                                            expected_chunk_ids=res.chunk_ids, status="verified")
            problems = list(res.problems) if res.problems else item_problems(candidate)
        else:
            present = present_absent_terms(index, item.absent_terms)
            candidate = dataclasses.replace(item, status="verified")
            problems = [f"term {t!r} DOES occur in the store (on {n} page(s)); this is not out of scope"
                        for t, n in present.items()] + (item_problems(candidate) if not present else [])
        if not problems:
            tag = "verified" if item.verified else "can be promoted"
            detail = f"pages {candidate.expected_pages}" if item.answerable else "no absent term occurs in the store"
            lines.append(f"  OK       {item.id:<6} {tag:<15} {detail}")
            updated.append(candidate)
        else:
            lines.append(f"  {'BROKEN' if item.verified else 'draft ':<8} {item.id:<6} " + problems[0])
            lines += [f"           {'':<6} {p}" for p in problems[1:]]
            if item.verified:
                broken.append(item.id)
            updated.append(item)
    return updated, lines, broken


def main(argv: list[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    parser = argparse.ArgumentParser(prog="validate_benchmark.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--store-dir", default=str(PROJECT_ROOT / DEFAULT_STORE_SUBDIR))
    parser.add_argument("--write", action="store_true", help="promote items that verify and rewrite the benchmark file (keeps a .bak)")
    args = parser.parse_args(argv)
    try:
        items = load_benchmark(args.benchmark)
        index = PageIndex.build(load_store_chunks(args.store_dir))
    except (BenchmarkError, PersistenceError) as exc:
        print(f"error: {exc}", file=err)
        return 1

    updated, lines, broken = check_items(items, index)
    print(f"Benchmark {args.benchmark}\nStore documents: {', '.join(index.files) or 'none'}", file=out)
    print("\n".join(lines), file=out)
    already = sum(1 for i in items if i.verified)
    promotable = sum(1 for before, after in zip(items, updated) if after.verified and not before.verified)
    summary = f"\n{already} of {len(items)} items verified"
    summary += f"; {promotable} more " + ("promoted to verified" if args.write else "can be promoted (re-run with --write)")
    if broken:
        summary += f"; {len(broken)} previously verified item(s) BROKEN"
    print(summary, file=out)

    if args.write:
        if updated != items:
            shutil.copy2(args.benchmark, str(args.benchmark) + ".bak")
            dump_benchmark(updated, args.benchmark)
            print(f"updated {args.benchmark} (previous version kept as {args.benchmark}.bak)", file=out)
        else:
            print("nothing to update", file=out)
    return 1 if broken else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    sys.exit(main())
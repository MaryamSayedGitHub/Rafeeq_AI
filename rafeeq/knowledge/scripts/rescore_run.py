"""Re-score a finished run against one or more benchmark files WITHOUT re-running retrieval.

    uv run python scripts/rescore_run.py eval/results/<run>/results.json ^
        --benchmark eval/benchmark/hands_on_llm_v1.jsonl --benchmark eval/benchmark/hands_on_llm_v1_1.jsonl

Why: when a benchmark's ground truth changes, a new retrieval run mixes two effects (the labels and the
system). The retrieved chunks of an old run do not depend on the labels, so re-scoring them with the new
labels isolates the label effect exactly; the difference between two real runs scored against the SAME
labels is then the system effect. Nothing here loads a model, a store or the network.

The first --benchmark is the reference; every other one is compared against it. A question is re-scored
only if its text matches the one stored in the run and the run kept enough retrieved chunks for the
largest k; anything else is listed under "not re-scored" instead of being guessed.
Exit status: 0 = report printed, 1 = could not run, 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
try:
    import rafeeq.knowledge  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))

from rafeeq.knowledge.evaluation.dataset import BenchmarkError, load_benchmark  # noqa: E402
from rafeeq.knowledge.evaluation.runner import aggregate_retrieval, score_retrieval  # noqa: E402

DEFAULT_MIN_SCORE = 0.35
HEADLINE = ("hit@1", "hit@3", "hit@5", "mrr@5", "recall@5", "full_recall@5")


class RescoreError(Exception):
    """The run or a benchmark cannot be used."""


def _hits_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": h["chunk_id"], "file_name": h["file_name"], "page": h["page"], "topic_id": h.get("topic_id"),
         "score": h["score"], "text": h.get("text_preview", "")}
        for h in record.get("retrieved", [])
    ]


def rescore(run: dict[str, Any], items: list[Any], ks: list[int], min_score: float = DEFAULT_MIN_SCORE):
    """Return (records, not_rescored) for the verified ``items`` against the retrieval stored in ``run``."""
    stored = {r["id"]: r for r in run.get("per_question", [])}
    depth = max(ks)
    records: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for item in items:
        if item.status != "verified":
            continue
        old = stored.get(item.id)
        if old is None:
            skipped.append((item.id, "not in the run"))
        elif old.get("question") != item.question:
            skipped.append((item.id, "question text differs from the run"))
        elif len(old.get("retrieved", [])) < depth:
            skipped.append((item.id, f"run kept {len(old.get('retrieved', []))} chunks, need {depth}"))
        else:
            records.append(score_retrieval(item, _hits_from_record(old), ks, min_score))
    return records, skipped


def _pct(value: float | None, key: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if key.startswith("mrr") else f"{value * 100:.1f}%"


def _rank(record: dict[str, Any]) -> str:
    return str(record["first_relevant_rank"]) if record.get("first_relevant_rank") else "none"


def render(run_label: str, names: list[str], scored: list[list[dict[str, Any]]], skipped: list[list[tuple[str, str]]], ks: list[int]) -> str:
    aggregates = [aggregate_retrieval(records, ks) for records in scored]
    lines = [f"# Re-score of {run_label}", "",
             "Retrieval is taken from the stored run; only the ground truth differs between columns.", ""]
    for scope, getter in (("all answerable", lambda a: a["overall"]),
                          ("multi_passage", lambda a: a["by_qtype"].get("multi_passage", {}))):
        lines += [f"**{scope}** (n = {', '.join(str(getter(a).get('n', 0)) for a in aggregates)})", "",
                  "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for key in HEADLINE:
            values = [getter(a).get(key) for a in aggregates]
            if all(v is None for v in values):
                continue
            lines.append(f"| {key} | " + " | ".join(_pct(v, key) for v in values) + " |")
        lines.append("")
    base = {r["id"]: r for r in scored[0] if r["answerable"]}
    for name, records in zip(names[1:], scored[1:]):
        changed = []
        for record in (r for r in records if r["answerable"]):
            ref = base.get(record["id"])
            if ref is None:
                continue
            if (ref["expected_pages"], ref["first_relevant_rank"]) != (record["expected_pages"], record["first_relevant_rank"]) \
                    or ref["recall@5"] != record["recall@5"]:
                changed.append((record["id"], ref, record))
        lines += [f"## Questions that score differently under {name} than under {names[0]}", ""]
        if not changed:
            lines += ["none", ""]
            continue
        lines += ["| id | expected pages (ref -> this) | first relevant rank | recall@5 | retrieved pages (top 5) |", "|---|---|---|---|---|"]
        for qid, ref, rec in changed:
            pages = " ".join(str(h["page"]) for h in rec["retrieved"][:5])
            lines.append(f"| {qid} | {ref['expected_pages']} -> {rec['expected_pages']} | {_rank(ref)} -> {_rank(rec)} | "
                         f"{ref['recall@5']:.2f} -> {rec['recall@5']:.2f} | {pages} |")
        lines.append("")
    for name, skip in zip(names, skipped):
        if skip:
            lines += [f"## Not re-scored under {name}", ""] + [f"- {qid}: {why}" for qid, why in skip] + [""]
    lines.append("Small samples: with ~30 questions one question moves a Hit rate by 3.3 points. A label change that is "
                 "followed by a retrieval run does not show that the system improved; compare runs only under the same labels.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    parser = argparse.ArgumentParser(description="Re-score a stored run against benchmark files (no retrieval is re-run).")
    parser.add_argument("results", type=Path, help="results.json of a finished run")
    parser.add_argument("--benchmark", type=Path, action="append", required=True, help="benchmark JSONL; repeat to compare several (first = reference)")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="only used for the stored gate flag")
    parser.add_argument("--out", type=Path, default=None, help="also write the Markdown report here")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    try:
        run = json.loads(args.results.read_text(encoding="utf-8"))
        ks = [int(k) for k in run["retrieval"]["ks"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: cannot read a run from {args.results}: {exc}", file=err)
        return 1
    names: list[str] = []
    scored: list[list[dict[str, Any]]] = []
    skipped: list[list[tuple[str, str]]] = []
    for path in args.benchmark:
        try:
            items = load_benchmark(path)
        except (OSError, BenchmarkError) as exc:
            print(f"error: cannot load {path}: {exc}", file=err)
            return 1
        records, skip = rescore(run, items, ks, args.min_score)
        if not records:
            print(f"error: nothing in {path} could be re-scored against this run", file=err)
            return 1
        names.append(path.name)
        scored.append(records)
        skipped.append(skip)
    text = render(str(args.results), names, scored, skipped, ks)
    print(text, file=out, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

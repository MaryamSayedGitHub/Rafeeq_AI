#!/usr/bin/env python
"""Final held-out comparison of the four retrieval configurations, from REAL run_eval.py results only.

    uv run python scripts/heldout_report.py \\
        --dense eval/results/heldout_dense/results.json \\
        --dense-exclude eval/results/heldout_dense_exclude/results.json \\
        --hybrid eval/results/heldout_hybrid/results.json \\
        --hybrid-exclude eval/results/heldout_hybrid_exclude/results.json \\
        --out eval/analysis/heldout_final_comparison.md

The script refuses to produce a report unless every input
  * was scored on the frozen held-out benchmark (file name + the sha256 prefix recorded in
    eval/benchmark/hands_on_llm_heldout_v1.sha256),
  * is the configuration it is passed as (dense / dense + exclusion / hybrid / hybrid + exclusion, and the
    hybrid runs state RRF k=60 and candidate pool 50),
  * used the same embedder, store, room and cut-offs, and
  * is not a replay (scripts/replay_hybrid.py output is labelled "REPLAY" and is rejected).
It reads files; it never embeds, searches or edits anything. No threshold, weight or parameter is chosen here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import compare  # noqa: E402  (same directory)

HELDOUT_NAME = "hands_on_llm_heldout_v1"
FREEZE_FILE = KB_DIR / "eval" / "benchmark" / "hands_on_llm_heldout_v1.sha256"
CONFIGS = ("dense", "dense_exclude", "hybrid", "hybrid_exclude")
LABEL = {"dense": "Dense", "dense_exclude": "Dense + exclusion", "hybrid": "Hybrid", "hybrid_exclude": "Hybrid + exclusion"}
PAIRS = (("dense", "hybrid"), ("dense_exclude", "hybrid_exclude"), ("dense", "dense_exclude"), ("hybrid", "hybrid_exclude"))
METRICS = ("hit@1", "hit@3", "hit@5", "mrr@5", "recall@5", "full_recall@5")


class ReportError(ValueError):
    """The inputs cannot be compared honestly."""


def frozen_prefix(path: Path = FREEZE_FILE) -> str:
    try:
        return path.read_text(encoding="utf-8").split()[0][:12]
    except (OSError, IndexError) as exc:
        raise ReportError(f"cannot read the freeze file {path}: {exc}") from exc


def expected_mode(config: str) -> tuple[bool, bool]:
    return config.startswith("hybrid"), config.endswith("_exclude")


def check_run(config: str, doc: dict[str, Any], prefix: str) -> list[str]:
    """Reasons ``doc`` is not the ``config`` run on the frozen held-out benchmark (empty = fine)."""
    problems: list[str] = []
    meta = doc.get("meta", {})
    bench = str(meta.get("benchmark_file", ""))
    if HELDOUT_NAME not in bench:
        problems.append(f"benchmark_file is not the held-out benchmark: {bench!r}")
    elif prefix and f"sha256 {prefix}" not in bench:
        problems.append(f"benchmark sha256 does not match the freeze file ({prefix}): {bench!r}")
    retrieval = str(meta.get("retrieval", ""))
    low = retrieval.lower()
    if "replay" in low:
        problems.append("this is a REPLAY, not a real run; only real run_eval.py output may be used here")
    want_hybrid, want_excl = expected_mode(config)
    if want_hybrid != low.startswith("hybrid"):
        problems.append(f"retrieval mode is {retrieval!r}, expected {'hybrid' if want_hybrid else 'dense'}")
    if want_excl != ("index-like pages excluded" in low):
        problems.append(f"exclusion setting is {retrieval!r}, expected {'with' if want_excl else 'without'} exclusion")
    if want_hybrid and not ("rrf k=60" in low and "candidate pool 50" in low):
        problems.append(f"hybrid run must state RRF k=60 and candidate pool 50, got {retrieval!r}")
    if "per_question" not in doc or "retrieval" not in doc:
        problems.append("not a run_eval.py results.json")
    return problems


def check_same_setup(docs: dict[str, dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for key in ("embedder", "store", "room", "ks", "documents in room"):
        values = {str(d.get("meta", {}).get(key)) for d in docs.values()}
        if len(values) > 1:
            problems.append(f"runs differ in {key}: {sorted(values)}")
    ids = {name: tuple(r["id"] for r in d["per_question"]) for name, d in docs.items()}
    if len(set(ids.values())) > 1:
        problems.append("runs do not contain the same questions in the same order")
    return problems


def _rows(doc: dict[str, Any], keep) -> list[dict[str, Any]]:
    return [r for r in doc["per_question"] if keep(r)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def subset_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {"n": float(len(rows))}
    for m in METRICS:
        out[m] = _mean([float(r["rr"] if m == "mrr@5" else r[m]) for r in rows])
    return out


def sign_test_p(improved: int, regressed: int) -> float:
    """Two-sided exact sign test on the questions whose rank changed (ties dropped)."""
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _pct(v: float) -> str:
    return "n/a" if v != v else f"{100 * v:.1f}%"


def _fmt(metric: str, v: float) -> str:
    return "n/a" if v != v else (f"{v:.3f}" if metric.startswith("mrr") else _pct(v))


def metrics_table(title: str, docs: dict[str, dict[str, Any]], keep) -> list[str]:
    first = subset_metrics(_rows(docs[CONFIGS[0]], keep))
    lines = [f"### {title} (n={int(first['n'])})", "", "| configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Full Recall@5 |", "|---|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        m = subset_metrics(_rows(docs[cfg], keep))
        lines.append(f"| {LABEL[cfg]} | " + " | ".join(_fmt(k, m[k]) for k in METRICS) + " |")
    return lines + [""]


def gate_table(docs: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["| configuration | answerable questions passing the 0.35 gate | out-of-scope questions passing the gate (false accepts) | highest out-of-scope best-cosine |", "|---|---|---|---|"]
    for cfg in CONFIGS:
        rows = docs[cfg]["per_question"]
        ans = [r for r in rows if r["answerable"]]
        oos = [r for r in rows if not r["answerable"]]
        top = max((r.get("max_score") if r.get("max_score") is not None else (r.get("top1_score") or 0.0)) for r in oos) if oos else float("nan")
        lines.append(f"| {LABEL[cfg]} | {sum(bool(r['gate_passed']) for r in ans)}/{len(ans)} | {sum(bool(r['gate_passed']) for r in oos)}/{len(oos)} | {top:.3f} |")
    return lines + [""]


def pair_section(a: str, b: str, docs: dict[str, dict[str, Any]]) -> list[str]:
    cmp = compare(docs[a], docs[b])
    ans = cmp["answerable"]
    imp, reg = ans["improved"], ans["regressed"]
    lines = [f"### {LABEL[a]}  ->  {LABEL[b]}", "",
             f"Questions whose first relevant rank improved: **{len(imp)}**, regressed: **{len(reg)}**, unchanged: {ans['unchanged']}. "
             f"Exact two-sided sign test on the {len(imp) + len(reg)} changed questions: p = {sign_test_p(len(imp), len(reg)):.3f}.", ""]
    for title, rows in (("Regressions", reg), ("Improvements", imp)):
        lines.append(f"**{title}**")
        if not rows:
            lines += ["", "none", ""]
            continue
        lines += ["", "| id | type | lang | first relevant rank | pages before | pages after |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['id']} | {r['qtype']} | {r['language']} | {r['rank_before']} -> {r['rank_after']} | {' '.join(map(str, r['pages_before']))} | {' '.join(map(str, r['pages_after']))} |" for r in rows]
        lines.append("")
    oos = [r for r in cmp["out_of_scope"] if r["delta"] is not None]
    if oos:
        lines += ["Out-of-scope best top-1 cosine (descriptive):", "", "| id | before | after |", "|---|---|---|"]
        lines += [f"| {r['id']} | {r['top1_before']:.3f} | {r['top1_after']:.3f} |" for r in oos]
        lines.append("")
    return lines


def rank_table(docs: dict[str, dict[str, Any]]) -> list[str]:
    def rk(r):
        v = r.get("first_relevant_rank")
        return "-" if v in (None, "") else str(int(v))
    base = {r["id"]: r for r in docs[CONFIGS[0]]["per_question"]}
    by_cfg = {cfg: {r["id"]: r for r in docs[cfg]["per_question"]} for cfg in CONFIGS}
    lines = ["| id | type | lang | expected pages | " + " | ".join(LABEL[c] for c in CONFIGS) + " |", "|---|---|---|---|" + "---|" * len(CONFIGS)]
    for qid, r in base.items():
        if not r["answerable"]:
            continue
        lines.append(f"| {qid} | {r['qtype']} | {r['language']} | {' '.join(map(str, r['expected_pages']))} | " + " | ".join(rk(by_cfg[c][qid]) for c in CONFIGS) + " |")
    return lines + [""]


def build_report(docs: dict[str, dict[str, Any]], paths: dict[str, str]) -> str:
    multi = lambda r: r["answerable"] and r["qtype"] == "multi_passage"      # noqa: E731
    single = lambda r: r["answerable"] and r["qtype"] != "multi_passage"     # noqa: E731
    n_ans = sum(1 for r in docs[CONFIGS[0]]["per_question"] if r["answerable"])
    n_multi = max(1, sum(1 for r in docs[CONFIGS[0]]["per_question"] if multi(r)))
    lines = [
        "# Held-out comparison: Dense, Dense + exclusion, Hybrid, Hybrid + exclusion",
        "",
        f"Benchmark: `{HELDOUT_NAME}` (frozen; see eval/HELDOUT_PROTOCOL.md). Same embedder, store, room, cut-offs and candidate pool 50 for the hybrid runs.",
        "",
        "**Status of every number below: VERIFIED. They are computed from real `run_eval.py` results.json files; nothing here is a replay or a simulation.** "
        "Inputs: " + ", ".join(f"{LABEL[c]} = `{paths[c]}`" for c in CONFIGS) + ".",
        "",
        f"With {n_ans} answerable questions one question moves a Hit rate by {100 / n_ans:.1f} points, and with {n_multi} multi-passage questions by {100 / n_multi:.1f} points. "
        "Read the per-question lists, not only the percentages; the sign test below has little power at this size.",
        "",
        "## Metrics",
        "",
    ]
    lines += metrics_table("All answerable questions", docs, lambda r: r["answerable"])
    lines += metrics_table("Single-passage questions", docs, single)
    lines += metrics_table("Multi-passage questions (Full Recall@5 = every labelled page in the top 5)", docs, multi)
    lines += ["## Abstention gate (descriptive; the 0.35 threshold is the project default and was not tuned)", ""] + gate_table(docs)
    lines += ["## Paired comparisons, per question", ""]
    for a, b in PAIRS:
        lines += pair_section(a, b, docs)
    lines += ["## First relevant rank of every answerable question (`-` = not in the top 5)", ""] + rank_table(docs)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    p = argparse.ArgumentParser(prog="heldout_report.py", description=__doc__.split("\n\n")[0])
    for cfg in CONFIGS:
        p.add_argument("--" + cfg.replace("_", "-"), required=True, help=f"results.json of the {LABEL[cfg]} run")
    p.add_argument("--freeze-file", default=str(FREEZE_FILE))
    p.add_argument("--out", default=None, help="also write the Markdown report here (an existing file is never overwritten)")
    args = p.parse_args(argv)
    paths = {cfg: getattr(args, cfg) for cfg in CONFIGS}
    prefix = frozen_prefix(Path(args.freeze_file))
    docs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for cfg, path in paths.items():
        try:
            docs[cfg] = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{LABEL[cfg]}: cannot read {path}: {exc}")
            continue
        errors += [f"{LABEL[cfg]}: {m}" for m in check_run(cfg, docs[cfg], prefix)]
    if not errors:
        errors += check_same_setup(docs)
    if args.out and Path(args.out).exists():
        errors.append(f"{args.out} already exists; results and reports are never overwritten")
    if errors:
        print("refusing to build the report:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2
    text = build_report(docs, paths)
    print(text, file=out or sys.stdout)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

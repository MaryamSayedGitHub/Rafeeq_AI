"""Compare two run_eval.py results (before vs after a retrieval change) without judging them for you.

    uv run python scripts/compare_runs.py eval/results/<baseline>/results.json eval/results/<variant>/results.json

Prints a Markdown report (and writes it with --out): metric deltas overall and for each question type /
language, every answerable question whose first relevant rank improved or got worse, and what happened to the
top-1 scores of the out-of-scope questions. It states nothing about whether to keep the change; that
decision needs the regression list and the small-sample caveat in front of a human.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

RANK_NONE = float("inf")


def _rank(row: dict[str, Any]) -> float:
    r = row.get("first_relevant_rank")
    return RANK_NONE if r in (None, "") else float(r)


def _fmt_rank(value: float) -> str:
    return "none" if value == RANK_NONE else str(int(value))


def _metric_keys(group: dict[str, Any]) -> list[str]:
    keys = [k for k in group if k != "n" and not k.endswith("_ci95") and isinstance(group[k], (int, float))]
    return sorted(keys, key=lambda k: (k.split("@")[0] if "@" in k else k, int(k.split("@")[1]) if "@" in k else 0))


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable comparison of two results.json documents."""
    warnings: list[str] = []
    b_meta, a_meta = before.get("meta", {}), after.get("meta", {})
    if b_meta.get("benchmark_file") != a_meta.get("benchmark_file"):
        warnings.append("the two runs used different benchmark files (name or sha256 differs); the comparison may not be like for like")
    if b_meta.get("ks") != a_meta.get("ks"):
        warnings.append(f"cut-offs differ (before: {b_meta.get('ks')}, after: {a_meta.get('ks')}); only shared metrics are compared")
    if b_meta.get("store") != a_meta.get("store"):
        warnings.append("the two runs report different stores")
    if b_meta.get("embedder") != a_meta.get("embedder"):
        warnings.append("the two runs report different embedders")

    def groups(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
        ret = doc["retrieval"]
        out = {"all answerable": ret["overall"]}
        out.update({f"type: {k}": v for k, v in ret.get("by_qtype", {}).items()})
        out.update({f"language: {k}": v for k, v in ret.get("by_language", {}).items()})
        return out

    b_groups, a_groups = groups(before), groups(after)
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for name in b_groups:
        if name not in a_groups:
            continue
        bg, ag = b_groups[name], a_groups[name]
        shared = [k for k in _metric_keys(bg) if k in ag]
        metrics[name] = {"n": bg.get("n"), **{k: {"before": bg[k], "after": ag[k], "delta": ag[k] - bg[k]} for k in shared}}

    b_rows = {r["id"]: r for r in before["per_question"]}
    a_rows = {r["id"]: r for r in after["per_question"]}
    only_one = sorted(set(b_rows) ^ set(a_rows))
    if only_one:
        warnings.append(f"questions present in only one run (ignored): {', '.join(only_one)}")

    improved, regressed, unchanged = [], [], 0
    oos = []
    for qid in sorted(set(b_rows) & set(a_rows)):
        b, a = b_rows[qid], a_rows[qid]
        if b.get("answerable"):
            rb, ra = _rank(b), _rank(a)
            entry = {"id": qid, "qtype": b.get("qtype"), "language": b.get("language"), "rank_before": _fmt_rank(rb), "rank_after": _fmt_rank(ra),
                     "pages_before": [h["page"] for h in b.get("retrieved", [])], "pages_after": [h["page"] for h in a.get("retrieved", [])]}
            if ra < rb:
                improved.append(entry)
            elif ra > rb:
                regressed.append(entry)
            else:
                unchanged += 1
        else:
            oos.append({"id": qid, "top1_before": b.get("top1_score"), "top1_after": a.get("top1_score"),
                        "delta": None if b.get("top1_score") is None or a.get("top1_score") is None else a["top1_score"] - b["top1_score"],
                        "top1_page_before": (b.get("retrieved") or [{}])[0].get("page"), "top1_page_after": (a.get("retrieved") or [{}])[0].get("page")})

    sweep = []
    b_sweep = {round(s["threshold"], 4): s for s in before.get("threshold_sweep", [])}
    for s in after.get("threshold_sweep", []):
        t = round(s["threshold"], 4)
        if t in b_sweep:
            sweep.append({"threshold": t, "answerable_before": b_sweep[t]["answerable_pass_rate"], "answerable_after": s["answerable_pass_rate"],
                          "out_of_scope_before": b_sweep[t]["out_of_scope_pass_rate"], "out_of_scope_after": s["out_of_scope_pass_rate"]})

    return {
        "before": {"retrieval": b_meta.get("retrieval", "unknown"), "run_at_utc": b_meta.get("run_at_utc")},
        "after": {"retrieval": a_meta.get("retrieval", "unknown"), "run_at_utc": a_meta.get("run_at_utc")},
        "warnings": warnings,
        "metrics": metrics,
        "answerable": {"improved": improved, "regressed": regressed, "unchanged": unchanged},
        "out_of_scope": oos,
        "threshold_sweep": sweep,
    }


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def _delta(name: str, d: float) -> str:
    if abs(d) < 1e-12:
        return "0"
    return f"{d:+.3f}" if name.startswith("mrr") else f"{100 * d:+.1f} pp"


def render(cmp: dict[str, Any]) -> str:
    lines = ["# Retrieval comparison", "",
             f"- before: {cmp['before']['retrieval']} (run {cmp['before']['run_at_utc']})",
             f"- after: {cmp['after']['retrieval']} (run {cmp['after']['run_at_utc']})", ""]
    for w in cmp["warnings"]:
        lines.append(f"> WARNING: {w}")
    if cmp["warnings"]:
        lines.append("")
    lines += ["## Metrics", "", "Small samples: with ~30 questions one question moves a Hit rate by 3.3 points. Read the lists below, not only the deltas.", ""]
    for name, group in cmp["metrics"].items():
        lines += [f"**{name}** (n={group['n']})", "", "| metric | before | after | change |", "|---|---|---|---|"]
        for key, v in group.items():
            if key == "n":
                continue
            fmt = (lambda x: f"{x:.3f}") if key.startswith("mrr") else _pct
            lines.append(f"| {key} | {fmt(v['before'])} | {fmt(v['after'])} | {_delta(key, v['delta'])} |")
        lines.append("")
    ans = cmp["answerable"]
    lines += ["## Answerable questions", "", f"Improved: **{len(ans['improved'])}**, regressed: **{len(ans['regressed'])}**, unchanged rank: {ans['unchanged']}", ""]
    for title, rows in (("Regressions (rank of first relevant page got worse)", ans["regressed"]), ("Improvements", ans["improved"])):
        lines.append(f"### {title}")
        if not rows:
            lines += ["", "none", ""]
            continue
        lines += ["", "| id | type | lang | first relevant rank (before -> after) | pages before | pages after |", "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['id']} | {r['qtype']} | {r['language']} | {r['rank_before']} -> {r['rank_after']} | {' '.join(map(str, r['pages_before']))} | {' '.join(map(str, r['pages_after']))} |")
        lines.append("")
    lines += ["## Out-of-scope questions (top-1 cosine score)", ""]
    if cmp["out_of_scope"]:
        lines += ["| id | top-1 before | top-1 after | change | top-1 page before -> after |", "|---|---|---|---|---|"]
        for r in cmp["out_of_scope"]:
            score = lambda x: "n/a" if x is None else f"{x:.3f}"
            change = "n/a" if r["delta"] is None else f"{r['delta']:+.3f}"
            lines.append(f"| {r['id']} | {score(r['top1_before'])} | {score(r['top1_after'])} | {change} | {r['top1_page_before']} -> {r['top1_page_after']} |")
    else:
        lines.append("none scored")
    lines += ["", "Descriptive score gate (share of questions whose top-1 score is at least the threshold):", "",
              "| threshold | answerable before | answerable after | out-of-scope before | out-of-scope after |", "|---|---|---|---|---|"]
    p = lambda x: "n/a" if x is None else _pct(x)
    for s in cmp["threshold_sweep"]:
        lines.append(f"| {s['threshold']:.2f} | {p(s['answerable_before'])} | {p(s['answerable_after'])} | {p(s['out_of_scope_before'])} | {p(s['out_of_scope_after'])} |")
    return "\n".join(lines) + "\n"


def _load(path: str) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict) or "retrieval" not in doc or "per_question" not in doc:
        raise SystemExit(f"{path} is not a run_eval.py results.json")
    return doc


def main(argv: list[str] | None = None, *, out: TextIO | None = None) -> int:
    p = argparse.ArgumentParser(prog="compare_runs.py", description=__doc__.split("\n\n")[0])
    p.add_argument("before", help="results.json of the baseline run")
    p.add_argument("after", help="results.json of the variant run")
    p.add_argument("--out", default=None, help="also write the Markdown report here")
    args = p.parse_args(argv)
    text = render(compare(_load(args.before), _load(args.after)))
    print(text, file=out or sys.stdout)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
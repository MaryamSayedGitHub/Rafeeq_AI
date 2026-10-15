#!/usr/bin/env python
"""Compare dense / hybrid / hybrid + Cross-Encoder from three REAL run_eval.py results.json files (retrieval only).

    python scripts/rerank_report.py --dense eval/results_rerank/heldout_dense/results.json \\
        --hybrid eval/results_rerank/heldout_hybrid/results.json \\
        --hybrid-rerank eval/results_rerank/heldout_hybrid_rerank/results.json \\
        --out eval/analysis_rerank/rerank_comparison.md

Reads files only; refuses an existing --out; refuses runs scored on different benchmarks / stores / embedders, a
hybrid-rerank run that is not actually reranked, or a baseline that is. Prints Hit@1/3/5, MRR@5 and the first relevant
rank of every question, with h02 and h07 shown explicitly. These are the project's own retrieval metrics, not RAGAS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FOCUS = ("h02", "h07")
METRICS = ("hit@1", "hit@3", "hit@5", "mrr@5")


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(runs: dict[str, dict]) -> None:
    meta = {k: r["meta"] for k, r in runs.items()}
    for field in ("benchmark_file", "store", "room", "embedder", "ks"):
        # embedder string includes the device; compare model name only
        vals = {(m[field].split(" on ")[0] if field == "embedder" else m[field]) for m in meta.values()}
        if len(vals) != 1:
            raise SystemExit(f"error: runs differ in {field!r}: {sorted(vals)}")
    if "hands_on_llm_heldout_v1.jsonl (sha256 6b0e57f23ff7)" not in meta["dense"]["benchmark_file"]:
        raise SystemExit("error: runs must be scored on the strict held-out benchmark hands_on_llm_heldout_v1.jsonl")
    if "Cross-Encoder" in meta["dense"]["retrieval"] or "Cross-Encoder" in meta["hybrid"]["retrieval"]:
        raise SystemExit("error: a baseline run used reranking")
    if "Cross-Encoder" not in meta["hybrid_rerank"]["retrieval"] or not meta["hybrid_rerank"]["retrieval"].startswith("hybrid"):
        raise SystemExit("error: --hybrid-rerank is not a hybrid + Cross-Encoder run")
    if not meta["dense"]["retrieval"].startswith("dense") or not meta["hybrid"]["retrieval"].startswith("hybrid"):
        raise SystemExit("error: --dense / --hybrid are not the expected modes")
    if "exclude" in "".join(m["retrieval"] for m in meta.values()) and len({"exclude" in m["retrieval"] for m in meta.values()}) != 1:
        raise SystemExit("error: runs differ in --exclude-non-content")
    pools = {m["retrieval"].split("candidate pool ")[1].split(")")[0] for k, m in meta.items() if k != "dense"}
    if len(pools) != 1:
        raise SystemExit(f"error: hybrid runs differ in candidate pool: {pools}")


def render(runs: dict[str, dict]) -> str:
    names = {"dense": "Dense", "hybrid": "Hybrid", "hybrid_rerank": "Hybrid + Cross-Encoder"}
    out = ["# Dense vs Hybrid vs Hybrid + Cross-Encoder (retrieval only)", "",
           "Custom retrieval metrics computed by this project (page-level relevance). **Not RAGAS.** Answer generation was not run.", "",
           f"Benchmark: {runs['dense']['meta']['benchmark_file']}", "",
           "| configuration | n answerable | Hit@1 | Hit@3 | Hit@5 | MRR@5 |", "|---|---|---|---|---|---|"]
    for key, label in names.items():
        o = runs[key]["retrieval"]["overall"]
        out.append(f"| {label} | {o['n']} | " + " | ".join(f"{o[m]:.3f}" for m in METRICS) + " |")
    out += ["", "## First relevant rank per question (page level; - = not in top 5)", "",
            "| id | Dense | Hybrid | Hybrid + CE | change Hybrid -> CE |", "|---|---|---|---|---|"]
    rows = {k: {r["id"]: r for r in run["per_question"]} for k, run in runs.items()}
    better = worse = same = 0
    for qid, r in rows["hybrid"].items():
        if not r["answerable"]:
            continue
        ranks = [rows[k][qid].get("first_relevant_rank") for k in names]
        a, b = ranks[1] or 99, ranks[2] or 99
        verdict = "improved" if b < a else "regressed" if b > a else "same"
        better, worse, same = better + (b < a), worse + (b > a), same + (b == a)
        mark = " **(focus)**" if qid in FOCUS else ""
        out.append(f"| {qid}{mark} | " + " | ".join(str(x) if x else "-" for x in ranks) + f" | {verdict} |")
    out += ["", f"Hybrid -> Hybrid + CE: improved {better}, regressed {worse}, unchanged {same} (42 answerable questions; one question = 2.4 points).", ""]
    for qid in FOCUS:
        if qid in rows["hybrid"]:
            out.append(f"* {qid}: first relevant rank Dense {rows['dense'][qid].get('first_relevant_rank')}, Hybrid {rows['hybrid'][qid].get('first_relevant_rank')}, Hybrid + CE {rows['hybrid_rerank'][qid].get('first_relevant_rank')}.")
    out += ["", "Abstention: out-of-scope questions are NOT handled by the reranker (the Answerer's cosine gate is unchanged); this report does not evaluate abstention."]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dense", required=True); p.add_argument("--hybrid", required=True)
    p.add_argument("--hybrid-rerank", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if Path(a.out).exists():
        print(f"error: {a.out} exists; never overwritten.", file=sys.stderr)
        return 1
    runs = {"dense": load(a.dense), "hybrid": load(a.hybrid), "hybrid_rerank": load(a.hybrid_rerank)}
    check(runs)
    text = render(runs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

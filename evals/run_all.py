"""One command, every feature, in the order of the M5 presentation.

    uv run python evals/run_all.py                 # the instructor (M5) suites, offline
    uv run python evals/run_all.py --all           # also other modules' suites (M3 handwriting)
    uv run python evals/run_all.py --only montage  # one suite
    uv run python evals/run_all.py --live          # also the checks that call the models (LLM_PROVIDER, Gemini judge)
    uv run python evals/run_all.py --no-upload     # do not send results to LangSmith / Langfuse
    uv run python evals/run_all.py --no-fail       # always exit 0 (when you only want the report)

Writes evals/results/run-<time>.json, latest.json, report.md and deck.json (the headline numbers per feature).
When LangSmith (LANGSMITH_TRACING=true + LANGSMITH_API_KEY) and/or Langfuse (LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY)
are set in .env, every case is traced there and every suite becomes a dataset + experiment ("rafeeq-m5-<suite>").
Exit status 1 when a gate fails (so CI can block a merge that makes a feature worse).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import pkgutil
import platform
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv                                              # keys live in .env, like the app

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from evals import core, flow, observe, suites as suites_pkg                    # noqa: E402
from rafeeq import observability                                                # noqa: E402


def discover() -> list:
    mods = []
    for info in pkgutil.iter_modules(suites_pkg.__path__):
        if not info.name.startswith("_"):
            mods.append(importlib.import_module(f"evals.suites.{info.name}"))
    return sorted(mods, key=lambda m: (flow.order(m.NAME), m.NAME))


def fmt(m: dict) -> str:
    v = m["value"]
    if v is None:
        return "-"
    return (f"{v:.1f}" if abs(v) >= 10 else f"{v:.3g}") + (f" {m['unit']}" if m["unit"] else "")


def gate_text(m: dict) -> str:
    return "" if m["gate"] is None else f"{'≥' if m['higher_is_better'] else '≤'} {m['gate']:g}"


def render(run: dict, previous: dict | None) -> str:
    old = core.index(previous)
    lines = ["# Evaluation report", f"Run {run['started']} on {run['platform']} · live: {run['live']} · "
             f"{run.get('tracing', '')}. Every number comes from running the code in this repository.", ""]
    s = run["summary"]
    lines += [f"**{s['PASS']} passed · {s['FAIL']} failed · {s['SKIPPED']} skipped · {s['INFO']} info**", ""]
    section = None
    for suite in run["suites"]:
        if suite["section"] != section:
            section = suite["section"]
            lines += [f"# {section}", ""]
        lines += [f"## {suite['feature']}  `{suite['name']}`", suite["description"], ""]
        pub = suite.get("published") or {}
        where = []
        if "langsmith" in pub:
            ls = pub["langsmith"]
            where.append(f"LangSmith: {ls.get('error') or 'dataset ' + ls['dataset'] + ' · experiment ' + ls.get('experiment', '')}")
        if "langfuse" in pub:
            lf = pub["langfuse"]
            where.append(f"Langfuse: {lf.get('error') or 'dataset ' + lf['dataset'] + ' · run ' + lf['run'] + (' · ' + lf['url'] if lf.get('url') else '')}")
        if where:
            lines += ["", *[f"- {w}" for w in where], ""]
        lines += ["| metric | value | gate | status | since last run | what it means |", "|---|---|---|---|---|---|"]
        for m in suite["metrics"]:
            lines.append(f"| {m['name']} | {fmt(m)} | {gate_text(m)} | {m['status']} | "
                         f"{core.change(old.get((suite['name'], m['name'])), m)} | {m['note']} |")
        if suite.get("cases"):
            bad = [c for c in suite["cases"] if c["error"]]
            lines += ["", f"{len(suite['cases'])} cases" + (f", {len(bad)} crashed: " + "; ".join(f"`{c['id']}` {c['error']}" for c in bad) if bad else "")]
        if suite.get("error"):
            lines += ["", f"**Suite crashed:** `{suite['error']}`"]
        lines.append("")
    return "\n".join(lines)


def deck_numbers(run: dict) -> dict:
    """Per feature, in slide order: the measured (not skipped) metrics. This is what goes on the slides."""
    out = []
    for s in run["suites"]:
        shown = [{"metric": m["name"], "value": fmt(m), "gate": gate_text(m), "status": m["status"], "means": m["note"]}
                 for m in s["metrics"] if m["value"] is not None]
        out.append({"section": s["section"], "feature": s["feature"], "suite": s["name"], "metrics": shown,
                    "cases": len(s.get("cases") or []), "published": s.get("published") or {}})
    return {"run": run["started"], "live": run["live"], "features": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="suite names to run")
    ap.add_argument("--live", action="store_true", help="also run checks that call the models or the internet")
    ap.add_argument("--no-upload", action="store_true", help="keep the results local (no LangSmith / Langfuse)")
    ap.add_argument("--no-fail", action="store_true")
    ap.add_argument("--all", action="store_true", help="also the suites of other modules (default: only the instructor / M5 features)")
    args = ap.parse_args()

    previous = core.load_latest()
    started = dt.datetime.now().astimezone()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    run = {"started": started.isoformat(timespec="seconds"), "platform": platform.platform(), "live": args.live,
           "tracing": observability.status(), "suites": []}
    print(run["tracing"])
    ctx = {"live": args.live, "stamp": stamp, "no_upload": args.no_upload}
    for mod in discover():
        if args.only and mod.NAME not in args.only:
            continue
        section, feature = flow.place(mod.NAME)
        if not (args.only or args.all) and section == flow.SECTIONS["Z"]:
            continue                                                            # not an instructor feature
        print(f"\n== {feature} ({mod.NAME})")
        entry = {"name": mod.NAME, "section": section, "feature": feature, "description": mod.DESCRIPTION,
                 "metrics": [], "error": ""}
        try:
            entry["metrics"] = [m.as_dict() for m in mod.run(ctx)]
        except Exception as e:                                                  # noqa: BLE001 - one broken suite must not hide the others
            traceback.print_exc()
            entry["error"] = f"{type(e).__name__}: {e}"[:300]
        entry["cases"] = observe.case_table(ctx, mod.NAME)
        entry["published"] = observe.publish(mod, ctx, entry["metrics"])
        run["suites"].append(entry)
    observability.flush()
    counts = {k: 0 for k in ("PASS", "FAIL", "SKIPPED", "INFO")}
    for s in run["suites"]:
        for m in s["metrics"]:
            counts[m["status"]] += 1
        counts["FAIL"] += 1 if s["error"] else 0
    run["summary"] = counts
    core.RESULTS.mkdir(parents=True, exist_ok=True)
    report = render(run, previous)
    (core.RESULTS / "report.md").write_text(report, encoding="utf-8")
    (core.RESULTS / "deck.json").write_text(json.dumps(deck_numbers(run), ensure_ascii=False, indent=2), encoding="utf-8")
    core.save_run(run, stamp)
    print(report)
    print(f"\nSaved: {core.RESULTS / 'report.md'}")
    return 0 if args.no_fail or counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

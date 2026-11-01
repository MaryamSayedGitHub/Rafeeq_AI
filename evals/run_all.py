"""The instructor (M5) evaluation pipeline, in the order of the presentation. The numbers come from the platforms:

    LangSmith  one dataset + one experiment per feature: LangSmith runs the feature and our evaluators
               (code checks + Gemini judge) and stores the feedback (evals/langsmith_eval.py)
    Langfuse   Langfuse's own LLM-as-a-judge evaluators score every trace of the features, from this run and from
               real use (evals/langfuse_eval.py), + monitoring: latency, errors, tokens, cost (evals/langfuse_monitor.py)

    uv run python evals/run_all.py                    # experiments with the model-free checks + Langfuse monitoring
    uv run python evals/run_all.py --live             # + the cases that call the models (Groq generator, Gemini judge)
    uv run python evals/run_all.py --only lesson_studio --live
    uv run python evals/run_all.py --report-only      # run nothing: rebuild the report from the newest experiments
    uv run python evals/run_all.py --days 7           # Langfuse window (default 30 days)
    uv run python evals/run_all.py --all              # also other modules' suites (M3 handwriting)
    uv run python evals/run_all.py --no-fail          # always exit 0

Writes evals/results/report.md, deck.json (headline numbers per feature, for the slides), latest.json, run-<time>.json.
Exit status 1 when a gate fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import pkgutil
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "evaluation")       # Langfuse: evaluation traces apart from real use

from evals import core, flow, langfuse_eval, langfuse_monitor, langsmith_eval, suites as suites_pkg   # noqa: E402
from rafeeq import observability                                                      # noqa: E402


def discover() -> list:
    mods = []
    for info in pkgutil.iter_modules(suites_pkg.__path__):
        if not info.name.startswith("_"):
            mods.append(importlib.import_module(f"evals.suites.{info.name}"))
    return sorted(mods, key=lambda m: (flow.order(m.NAME), m.NAME))


def fmt(value, unit: str = "") -> str:
    if value is None:
        return "-"
    return (f"{value:.1f}" if abs(value) >= 10 else f"{value:.3g}") + (f" {unit}" if unit else "")


def gate_text(m: dict) -> str:
    return "" if m["gate"] is None else f"{'≥' if m['higher_is_better'] else '≤'} {m['gate']:g}"


def render(run: dict, previous: dict | None) -> str:
    old = core.index(previous)
    s = run["summary"]
    lines = ["# Instructor (M5) evaluation report",
             f"Run {run['started']} · live: {run['live']} · {run['tracing']}", "",
             "Per feature: LangSmith experiment (mean of the evaluator feedback per case) + `langfuse_judge` "
             "(Langfuse's own LLM-as-a-judge on the same run's traces). Monitoring: Langfuse Metrics API. "
             "Every number below is read from those two tools.", "",
             f"**{s['PASS']} passed · {s['FAIL']} failed · {s['SKIPPED']} skipped · {s['INFO']} info**", ""]
    section = None
    for suite in run["suites"]:
        if suite["section"] != section:
            section = suite["section"]
            lines += [f"# {section}", ""]
        lines += [f"## {suite['feature']}", suite["description"], ""]
        if suite.get("carried_from"):
            lines += [f"_Not run this time: result of the run of {suite['carried_from']}._", ""]
        if suite.get("experiment"):
            st = suite.get("stats") or {}
            lines += [f"LangSmith experiment `{suite['experiment']}` on dataset `{suite['dataset']}`"
                      + (f" · [open]({suite['url']})" if suite.get("url") else "")
                      + f" · runs {st.get('runs')} · p50 {fmt(st.get('latency_p50_s'), 's')} · tokens {st.get('total_tokens') or '-'}", ""]
        lines += ["| metric | value | gate | status | since last run | what it means |", "|---|---|---|---|---|---|"]
        for m in suite["metrics"]:
            lines.append(f"| {m['name']} | {fmt(m['value'], m['unit'])} | {gate_text(m)} | {m['status']} | "
                         f"{core.change(old.get((suite['name'], m['name'])), m)} | {m['note']} |")
        bad = [c for c in suite.get("cases", []) if c["error"]]
        if suite.get("cases"):
            lines += ["", f"{len(suite['cases'])} cases" + (f", {len(bad)} ended with an error: "
                                                            + "; ".join(f"`{c['id']}` {c['error'][:120]}" for c in bad) if bad else "")]
        if suite.get("error"):
            lines += ["", f"**Could not evaluate:** `{suite['error']}`"]
        lines.append("")

    mon = run.get("monitoring") or {}
    lines += ["# Monitoring (Langfuse)", ""]
    if mon.get("error"):
        lines += [mon["error"], ""]
    else:
        lines += [f"Last {mon['days']} days. `default` = the running app, `evaluation` = traces of evaluation runs.", "",
                  "| feature | environment | traces | p50 | p95 | errors | tokens | cost | scores (avg, n) |", "|---|---|---|---|---|---|---|---|---|"]
        for f in mon.get("features", []):
            if f.get("error"):
                lines.append(f"| {f['feature']} | - | query failed: {f['error']} | | | | | | |")
                continue
            envs = f["by_environment"] or {"-": {}}
            for env, d in envs.items():
                scores = ", ".join(f"{k} {fmt(v['avg'])} (n={int(v['n'] or 0)})" for k, v in (d.get("scores") or {}).items())
                cost = d.get("cost_usd")
                lines.append(f"| {f['feature']} | {env} | {int(d['traces']) if d.get('traces') else 0} | {fmt(d.get('p50_s'), 's')} | "
                             f"{fmt(d.get('p95_s'), 's')} | {fmt(d.get('error_rate_pct'), '%')} | {int(d['tokens']) if d.get('tokens') else '-'} | "
                             f"{'$' + format(cost, '.4f') if cost else '-'} | {scores or '-'} |")
        lines.append("")
    return "\n".join(lines)


def deck_numbers(run: dict) -> dict:
    feats = []
    for s in run["suites"]:
        feats.append({"section": s["section"], "feature": s["feature"], "suite": s["name"],
                      "langsmith": {"experiment": s.get("experiment"), "dataset": s.get("dataset"), "url": s.get("url")},
                      "metrics": [{"metric": m["name"], "value": fmt(m["value"], m["unit"]), "gate": gate_text(m),
                                   "status": m["status"], "means": m["note"]} for m in s["metrics"] if m["value"] is not None]})
    return {"run": run["started"], "live": run["live"], "features": feats, "monitoring": run.get("monitoring")}


def add_langfuse_judge(run: dict, started: dt.datetime, args) -> None:
    """Langfuse scores each new trace with its own evaluators shortly after it arrives. Wait --langfuse-wait seconds
    ONCE, then ask Langfuse ONE time (its Metrics API allows 100 queries a day on the free plan)."""
    judged = [s for s in run["suites"] if s["name"] in langfuse_eval.JUDGES and any(not c["error"] for c in s.get("cases", []))]
    if not judged:
        return
    since = started - dt.timedelta(days=args.days) if args.report_only else started
    if not args.report_only and args.langfuse_wait:
        print(f"  giving Langfuse {args.langfuse_wait} s to score this run's traces ...")
        time.sleep(args.langfuse_wait)
    try:
        got = langfuse_monitor.judge_scores(since, "evaluation")
        why = ""
    except langfuse_monitor.RateLimited as e:
        got, why = {}, str(e)
    if not got and not why:
        why = "no score from Langfuse yet: " + langfuse_eval.status_line()
    if why:
        print("  " + why)
    for s in judged:
        g = got.get(s["name"])
        value = round(g["avg"], 4) if g and g.get("avg") is not None else None
        note = (f"Langfuse LLM-as-a-judge '{g['name']}' on this run's traces, n={g['n']}" if g else
                f"Langfuse LLM-as-a-judge: {why or 'no score for this feature yet'}")
        s["metrics"].append(core.Metric("langfuse_judge", value, "", True, 0.7, note[:400]).as_dict())


def saved_entries() -> dict[str, dict]:
    """The newest saved result of every suite, across all run-<time>.json files (results are kept, never replaced):
    {suite: entry + 'run' = when that run started}. A run with --only adds to the report instead of replacing it."""
    out: dict[str, dict] = {}
    for f in sorted(core.RESULTS.glob("run-*.json"), reverse=True):        # newest first (names sort by time)
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for s in data.get("suites", []):
            if s.get("name") and s.get("experiment") and s["name"] not in out:   # only results read from LangSmith
                out[s["name"]] = {**s, "run": s.get("carried_from") or data.get("started", f.stem[4:])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="suite names")
    ap.add_argument("--live", action="store_true", help="also the cases that call the models or the internet")
    ap.add_argument("--report-only", action="store_true", help="run nothing; read the newest experiment of each dataset")
    ap.add_argument("--days", type=int, default=30, help="Langfuse monitoring window")
    ap.add_argument("--langfuse-wait", type=int, default=90, help="seconds to wait for Langfuse's judges to score this run")
    ap.add_argument("--all", action="store_true", help="also other modules' suites")
    ap.add_argument("--no-fail", action="store_true")
    args = ap.parse_args()

    print(observability.status())
    if not langsmith_eval.enabled():
        print("LANGSMITH_API_KEY is not set: the experiments run in LangSmith, so there is nothing to evaluate with.")
        return 1

    saved = saved_entries()                                                    # before this run is written
    previous = {"suites": list(saved.values())}                                # "since last run" = per suite
    started = dt.datetime.now().astimezone()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    run = {"started": started.isoformat(timespec="seconds"), "platform": platform.platform(), "live": args.live,
           "tracing": observability.status(), "suites": []}
    ctx = {"live": args.live}
    if observability.langfuse_enabled() and not args.report_only:
        try:                                                                    # Langfuse's own judges (idempotent)
            run["langfuse_setup"] = langfuse_eval.setup()
            print(f"Langfuse evaluators ready: {', '.join(run['langfuse_setup'])}")
        except Exception as e:                                                  # noqa: BLE001
            traceback.print_exc()
            run["langfuse_setup"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    for mod in discover():
        section, feature = flow.place(mod.NAME)
        if args.only and mod.NAME not in args.only:
            continue
        if not (args.only or args.all) and section == flow.SECTIONS["Z"]:
            continue
        print(f"\n== {feature} ({mod.NAME})")
        entry = {"name": mod.NAME, "section": section, "feature": feature, "description": mod.DESCRIPTION,
                 "metrics": [], "error": ""}
        try:
            exp = langsmith_eval.latest_experiment(mod) if args.report_only else langsmith_eval.run_experiment(mod, ctx)
            if exp is None:
                from evals import observe
                why = observe.why_not(mod, ctx) if langsmith_eval.is_case_suite(mod) else "no experiment yet"
                entry["metrics"] = [m.as_dict() for m in observe.skipped(getattr(mod, "SPEC", []), why)]
            else:
                got = langsmith_eval.read_experiment(mod, exp, ctx)
                entry.update(experiment=got["experiment"], dataset=got["dataset"], url=got["url"], stats=got["stats"],
                             cases=got["cases"], metrics=[m.as_dict() for m in got["metrics"]])
        except Exception as e:                                                  # noqa: BLE001 - one broken suite must not hide the others
            traceback.print_exc()
            entry["error"] = f"{type(e).__name__}: {e}"[:300]
        run["suites"].append(entry)
    observability.flush()

    if observability.langfuse_enabled():
        print("\n== Langfuse LLM-as-a-judge scores")
        try:
            add_langfuse_judge(run, started, args)
        except Exception as e:                                                  # noqa: BLE001
            traceback.print_exc()
            run["langfuse_judge_error"] = f"{type(e).__name__}: {e}"[:300]

    # every other instructor feature keeps its newest saved result (read from the tools when it ran)
    ran = {s["name"] for s in run["suites"]}
    for mod in discover():
        section, feature = flow.place(mod.NAME)
        if mod.NAME in ran or mod.NAME not in saved or (section == flow.SECTIONS["Z"] and not args.all):
            continue
        old = saved[mod.NAME]
        run["suites"].append({**old, "section": section, "feature": feature, "carried_from": old["run"]})
    run["suites"].sort(key=lambda s: (flow.order(s["name"]), s["name"]))

    print("\n== Langfuse monitoring")
    try:
        run["monitoring"] = langfuse_monitor.collect(args.days)
    except Exception as e:                                                      # noqa: BLE001
        traceback.print_exc()
        run["monitoring"] = {"error": f"{type(e).__name__}: {e}"[:300]}

    counts = {k: 0 for k in ("PASS", "FAIL", "SKIPPED", "INFO")}
    for s in run["suites"]:
        for m in s["metrics"]:
            counts[m["status"]] += 1
        counts["FAIL"] += 1 if s["error"] else 0
    run["summary"] = counts
    core.RESULTS.mkdir(parents=True, exist_ok=True)
    report = render(run, previous)
    (core.RESULTS / "report.md").write_text(report, encoding="utf-8")
    (core.RESULTS / "deck.json").write_text(json.dumps(deck_numbers(run), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    core.save_run(json.loads(json.dumps(run, default=str)), stamp)
    print(report)
    print(f"\nSaved: {core.RESULTS / 'report.md'}")
    try:                                                                        # the full PDF (cases, outputs, insights)
        from evals import report_pdf

        print(f"Saved: {report_pdf.build()}")
    except Exception as e:                                                      # noqa: BLE001 - the PDF must not break the run
        traceback.print_exc()
        print(f"PDF not built: {type(e).__name__}: {e}")
    return 0 if args.no_fail or counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

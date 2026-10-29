"""Case-based suites, and sending every suite's results to LangSmith and Langfuse.

A CASE SUITE is a module in evals/suites/ with

    NAME, DESCRIPTION
    LIVE = True                     # needs a model / the internet: only runs with --live
    cases() -> [{"id", "input", "expected"}]
    target(input) -> dict           # calls the real feature code
    SCORERS = {name: fn(input, output, expected) -> float | None}   # None = not applicable to this case
    summarize(rows) -> list[Metric] # the numbers in the report; rows == [] must still return gated metrics
    missing() -> str                # optional: why it cannot run here ("" = it can)
    offline(ctx) -> list[Metric]    # optional: checks that need no model, run every time
    run = lambda ctx: observe.run_case_suite(<module>, ctx)

Each case runs ONCE on this machine, inside a LangSmith trace and a Langfuse trace, so every model call it makes
(LangGraph nodes, Gemini, the judge) is nested under it. The scores are attached to the Langfuse trace.

Then publish() REPLAYS the saved outputs and scores (no second model call) into
    LangSmith  dataset "rafeeq-m5-<suite>"  -> an experiment (per-case scores + summary scores)
    Langfuse   dataset "rafeeq-m5-<suite>"  -> a dataset run (per-item scores + run-level scores)
so both dashboards show the same numbers as evals/results/report.md. Suites with plain run() are published as one
item whose scores are the suite's metrics. Publishing never breaks the local report.
"""
from __future__ import annotations

import contextlib
import os
import statistics
import subprocess
import time
import traceback

from evals.core import ROOT, Metric
from rafeeq import observability as obs

DATASET_PREFIX = "rafeeq-m5-"


# ------------------------------------------------------------------ helpers for summarize()
def mean(rows: list[dict], score: str) -> float | None:
    vals = [r["scores"].get(score) for r in rows if r["scores"].get(score) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def pct(rows: list[dict], score: str) -> float | None:
    """mean of a 0-1 score, as a percentage (the report's convention for unit '%')."""
    m = mean(rows, score)
    return None if m is None else round(100 * m, 1)


def median_seconds(rows: list[dict], key: str = "seconds") -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None and not r["error"]]
    return round(statistics.median(vals), 2) if vals else None


def error_rate(rows: list[dict]) -> float | None:
    """share of cases that crashed, in %"""
    return round(100 * sum(1 for r in rows if r["error"]) / len(rows), 1) if rows else None


def counted(rows: list[dict], score: str) -> dict:
    vals = [r["scores"].get(score) for r in rows if r["scores"].get(score) is not None]
    return {"cases": len(vals)}


def arabic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if "؀" <= c <= "ۿ") / len(letters) if letters else 0.0


def in_language(text: str, language: str) -> bool:
    """Arabic material: at least half the letters are Arabic (code words and names are English).
    English material: almost no Arabic letters."""
    share = arabic_share(text)
    return share >= 0.5 if language.strip().lower() in ("ar", "arabic", "العربية") else share <= 0.05


# ------------------------------------------------------------------ running cases
def _langsmith_trace(stack: contextlib.ExitStack, suite: str, case: dict):
    if not obs.langsmith_enabled():
        return None
    try:
        import langsmith

        return stack.enter_context(langsmith.trace(
            name=f"eval · {suite} · {case['id']}", run_type="chain", inputs=obs._short(case["input"]),
            tags=["eval", suite], metadata={"suite": suite, "case_id": case["id"]}))
    except Exception:                                                    # noqa: BLE001
        return None


def _langfuse_span(stack: contextlib.ExitStack, suite: str, case: dict):
    client = obs.langfuse_client()
    if client is None:
        return None, None
    try:
        span = stack.enter_context(client.start_as_current_observation(
            name=f"eval · {suite} · {case['id']}", as_type="span", input=obs._short(case["input"])))
        try:
            span.update_trace(tags=["eval", suite], metadata={"suite": suite, "case_id": case["id"]})
        except Exception:                                                # noqa: BLE001 - older/newer SDK
            pass
        return client, span
    except Exception:                                                    # noqa: BLE001
        return None, None


def run_case(mod, case: dict) -> dict:
    row = {"id": case["id"], "input": case["input"], "expected": case.get("expected"), "output": None,
           "scores": {}, "seconds": None, "error": "", "trace": {}}
    with contextlib.ExitStack() as stack:
        rt = _langsmith_trace(stack, mod.NAME, case)
        client, span = _langfuse_span(stack, mod.NAME, case)
        t0 = time.perf_counter()
        try:
            row["output"] = mod.target(case["input"])
        except Exception as e:                                           # noqa: BLE001 - a failed case is a result
            row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            traceback.print_exc()
        row["seconds"] = round(time.perf_counter() - t0, 2)
        if not row["error"]:
            for name, fn in mod.SCORERS.items():
                try:
                    value = fn(case["input"], row["output"], case.get("expected"))
                except Exception as e:                                   # noqa: BLE001 - a broken scorer = no score
                    print(f"  scorer {name} failed on {case['id']}: {type(e).__name__}: {e}")
                    value = None
                if value is not None:
                    row["scores"][name] = round(float(value), 4)
        result = {"output": obs._short(row["output"]), "scores": row["scores"], "error": row["error"],
                  "seconds": row["seconds"]}
        if rt is not None:
            try:
                rt.add_outputs(result)
                row["trace"]["langsmith"] = rt.get_url()
            except Exception:                                            # noqa: BLE001
                pass
        if span is not None:
            try:
                span.update(output=result, level="ERROR" if row["error"] else None,
                            status_message=row["error"] or None)
                tid = span.trace_id
                for name, value in row["scores"].items():
                    client.create_score(name=name, value=value, trace_id=tid, data_type="NUMERIC")
                client.create_score(name="seconds", value=row["seconds"], trace_id=tid, data_type="NUMERIC")
                row["trace"]["langfuse"] = client.get_trace_url(trace_id=tid)
            except Exception:                                            # noqa: BLE001
                pass
    print(f"  {mod.NAME} · {case['id']}: {row['seconds']}s {('ERROR ' + row['error']) if row['error'] else row['scores']}")
    return row


def run_case_suite(mod, ctx: dict) -> list[Metric]:
    extra = list(mod.offline(ctx)) if hasattr(mod, "offline") else []
    why = ""
    if getattr(mod, "LIVE", True) and not ctx.get("live"):
        why = "needs --live (calls the model)"
    elif hasattr(mod, "missing"):
        why = mod.missing()
    if why:
        metrics = mod.summarize([])
        for m in metrics:
            m.value, m.note = None, f"{why} · {m.note}"
        return metrics + extra
    rows = [run_case(mod, case) for case in mod.cases()]
    ctx.setdefault("rows", {})[mod.NAME] = rows
    return mod.summarize(rows) + extra


# ------------------------------------------------------------------ publishing
def run_meta(ctx: dict) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:                                                    # noqa: BLE001
        sha = ""
    return {"stamp": ctx.get("stamp", ""), "live": bool(ctx.get("live")), "git": sha,
            "llm_provider": os.getenv("LLM_PROVIDER", ""), "gemini_model": os.getenv("GEMINI_MODEL", ""),
            "judge_model": os.getenv("EVAL_JUDGE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"}


def _rows_for(mod, ctx: dict, metrics: list[dict]) -> list[dict]:
    rows = (ctx.get("rows") or {}).get(mod.NAME)
    if rows:
        return rows
    values = {m["name"]: m["value"] for m in metrics if m["value"] is not None}
    if not values:
        return []
    return [{"id": "summary", "input": {"suite": mod.NAME}, "expected": None, "output": values,
             "scores": values, "seconds": None, "error": ""}]


def _metric_scores(metrics: list[dict]) -> list[tuple[str, float, str]]:
    return [(m["name"], float(m["value"]), m["status"]) for m in metrics if m["value"] is not None]


def _to_langsmith(mod, rows: list[dict], metrics: list[dict], meta: dict) -> dict:
    from langsmith import Client, evaluate

    client = Client()
    name = DATASET_PREFIX + mod.NAME
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description=mod.DESCRIPTION)
    have = {(ex.inputs or {}).get("case_id") for ex in client.list_examples(dataset_name=name)}
    new = [{"inputs": {"case_id": r["id"], "input": obs._short(r["input"])},
            "outputs": {"expected": obs._short(r["expected"])}, "metadata": {"suite": mod.NAME}}
           for r in rows if r["id"] not in have]
    if new:
        client.create_examples(dataset_name=name, examples=new)
    cache = {r["id"]: r for r in rows}
    examples = [ex for ex in client.list_examples(dataset_name=name) if (ex.inputs or {}).get("case_id") in cache]

    def target(inputs: dict) -> dict:
        r = cache[inputs["case_id"]]
        return {"output": obs._short(r["output"]), "error": r["error"]}

    def case_scores(run, example) -> dict:
        r = cache[example.inputs["case_id"]]
        results = [{"key": k, "score": v} for k, v in r["scores"].items()]
        if r.get("seconds") is not None:
            results.append({"key": "seconds", "score": r["seconds"]})
        results.append({"key": "error", "score": 1 if r["error"] else 0, "comment": r["error"] or None})
        return {"results": results}

    def suite_metrics(runs, examples) -> dict:
        return {"results": [{"key": f"suite.{n}", "score": v, "comment": s} for n, v, s in _metric_scores(metrics)]}

    res = evaluate(target, data=examples, evaluators=[case_scores], summary_evaluators=[suite_metrics],
                   experiment_prefix=mod.NAME, description=mod.DESCRIPTION, metadata={**meta, "suite": mod.NAME},
                   max_concurrency=0)
    return {"dataset": name, "experiment": getattr(res, "experiment_name", "")}  # LangSmith › Datasets › <name>


def _to_langfuse(mod, rows: list[dict], metrics: list[dict], meta: dict) -> dict:
    from langfuse import Evaluation

    client = obs.langfuse_client()
    name = DATASET_PREFIX + mod.NAME
    try:
        client.get_dataset(name)
    except Exception:                                                    # noqa: BLE001 - not there yet
        client.create_dataset(name=name, description=mod.DESCRIPTION, metadata={"suite": mod.NAME})
    ids = {}
    for r in rows:
        item_id = f"{name}-{r['id']}"
        ids[item_id] = r
        client.create_dataset_item(dataset_name=name, id=item_id, input={"case_id": r["id"], "input": obs._short(r["input"])},
                                   expected_output=obs._short(r["expected"]), metadata={"suite": mod.NAME})
    dataset = client.get_dataset(name)
    items = [it for it in dataset.items if it.id in ids]
    cache = {r["id"]: r for r in rows}

    def task(*, item, **_):
        r = cache[(item.input or {})["case_id"]]
        return {"output": obs._short(r["output"]), "error": r["error"]}

    def case_scores(*, input, output, **_):                              # noqa: A002 - Langfuse's argument name
        r = cache[input["case_id"]]
        out = [Evaluation(name=k, value=v) for k, v in r["scores"].items()]
        if r.get("seconds") is not None:
            out.append(Evaluation(name="seconds", value=r["seconds"]))
        if r["error"]:
            out.append(Evaluation(name="error", value=1, comment=r["error"]))
        return out

    def suite_metrics(*, item_results, **_):
        return [Evaluation(name=f"suite.{n}", value=v, comment=s) for n, v, s in _metric_scores(metrics)]

    run_name = f"{mod.NAME} {meta.get('stamp', '')}".strip()
    kwargs = dict(name=mod.NAME, run_name=run_name, description=mod.DESCRIPTION, task=task,
                  evaluators=[case_scores], run_evaluators=[suite_metrics], max_concurrency=1,
                  metadata={k: str(v) for k, v in meta.items()})
    try:
        res = client.run_experiment(data=items, **kwargs)
    except TypeError:                                                    # SDKs where only the dataset runs experiments
        res = dataset.run_experiment(**kwargs)
    url = getattr(res, "dataset_run_url", "") or getattr(res, "experiment_url", "") or ""
    return {"dataset": name, "run": run_name, "url": url}


def publish(mod, ctx: dict, metrics: list[dict]) -> dict:
    """Send one suite's results to LangSmith and Langfuse (whichever is configured). Returns where they went."""
    if ctx.get("no_upload"):
        return {}
    rows = _rows_for(mod, ctx, metrics)
    if not rows:
        return {}
    meta = run_meta(ctx)
    links = {}
    if obs.langsmith_enabled():
        try:
            links["langsmith"] = _to_langsmith(mod, rows, metrics, meta)
        except Exception as e:                                           # noqa: BLE001
            links["langsmith"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            traceback.print_exc()
    if obs.langfuse_enabled():
        try:
            links["langfuse"] = _to_langfuse(mod, rows, metrics, meta)
        except Exception as e:                                           # noqa: BLE001
            links["langfuse"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            traceback.print_exc()
    return links


def case_table(ctx: dict, suite: str) -> list[dict]:
    """What the run JSON keeps per case (outputs trimmed): enough to explain a number without the dashboards."""
    return [{"id": r["id"], "scores": r["scores"], "seconds": r["seconds"], "error": r["error"],
             "trace": r.get("trace", {}), "output": obs._short(r["output"])}
            for r in (ctx.get("rows") or {}).get(suite, [])]

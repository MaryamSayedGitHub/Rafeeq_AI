"""Offline evaluation IN LangSmith: datasets, experiments, evaluators, and the numbers read back from LangSmith.

For every suite (one per instructor feature):

  1. DATASET     "rafeeq-m5-<suite>" in LangSmith, kept in sync with the suite's cases() (one example per case)
                 + one "checks" example when the suite has model-free checks.
  2. EXPERIMENT  langsmith.evaluate(target, data=<dataset examples>, evaluators=[rafeeq_scores]):
                 LangSmith calls the real feature for every example (each call is a traced run: every LangGraph node,
                 LLM call and tool call is nested inside), then runs the evaluator on the run and stores its feedback
                 (code checks + the Gemini LLM-as-judge) on that run.
  3. READ BACK   the experiment's runs and feedback are read from the LangSmith API and turned into the report's
                 numbers (mean of each feedback key, run latency, runs that ended with an error).

So the numbers in report.md are exactly what LangSmith shows under Datasets & Experiments > rafeeq-m5-<suite>.
`read_latest()` rebuilds the report from the newest experiment of each dataset without running anything.
"""
from __future__ import annotations

import os
import statistics
import subprocess
import time

from evals import observe
from evals.core import ROOT, Metric

PREFIX = "rafeeq-m5-"
CHECKS_ID = "checks"


def enabled() -> bool:
    return bool(os.getenv("LANGSMITH_API_KEY"))


def _client():
    from langsmith import Client

    return Client()


def dataset_name(mod) -> str:
    return PREFIX + mod.NAME


def is_case_suite(mod) -> bool:
    return hasattr(mod, "SCORERS")


# ------------------------------------------------------------------ 1. dataset
def wanted_examples(mod) -> list[dict]:
    rows = []
    if is_case_suite(mod):
        rows += [{"inputs": {"case_id": c["id"], "input": c["input"]}, "outputs": {"expected": c.get("expected")}}
                 for c in mod.cases()]
    if not is_case_suite(mod) or hasattr(mod, "offline"):
        rows.append({"inputs": {"case_id": CHECKS_ID, "checks": True}, "outputs": {"expected": None}})
    return rows


def sync_dataset(client, mod) -> str:
    """Create / update the dataset so it holds exactly the suite's cases (examples are matched by case_id)."""
    name = dataset_name(mod)
    if not client.has_dataset(dataset_name=name):
        client.create_dataset(name, description=mod.DESCRIPTION, metadata={"suite": mod.NAME, "module": "M5 instructor"})
    have = {(ex.inputs or {}).get("case_id"): ex for ex in client.list_examples(dataset_name=name)}
    want = wanted_examples(mod)
    new = []
    for row in want:
        ex = have.get(row["inputs"]["case_id"])
        if ex is None:
            new.append({**row, "metadata": {"suite": mod.NAME}})
        elif ex.inputs != row["inputs"] or (ex.outputs or {}) != row["outputs"]:
            client.update_example(ex.id, inputs=row["inputs"], outputs=row["outputs"])
    if new:
        client.create_examples(dataset_name=name, examples=new)
    keep = {r["inputs"]["case_id"] for r in want}
    for cid, ex in have.items():
        if cid not in keep:
            client.delete_example(ex.id)
    return name


# ------------------------------------------------------------------ 2. experiment
def _checks(mod, ctx: dict) -> list[Metric]:
    return list(mod.offline(ctx)) if is_case_suite(mod) else list(mod.run(ctx))


def make_target(mod, ctx: dict):
    def target(inputs: dict) -> dict:
        if inputs.get("checks"):
            return {"metrics": [m.as_dict() for m in _checks(mod, ctx)]}
        out = mod.target(inputs["input"])
        return out if isinstance(out, dict) else {"output": out}
    target.__name__ = f"{mod.NAME}"
    return target


def make_evaluator(mod):
    def rafeeq_scores(run, example) -> dict:
        """Code checks + LLM-as-judge for one run. LangSmith stores every result as feedback on the run."""
        out = run.outputs or {}
        if example.inputs.get("checks"):
            return {"results": [{"key": m["name"], "score": m["value"], "comment": (m.get("note") or "")[:500]}
                                for m in out.get("metrics", []) if m.get("value") is not None]}
        if run.error or not out:
            return {"results": []}
        scores = observe.score_case(mod, example.inputs["input"], out, (example.outputs or {}).get("expected"))
        return {"results": [{"key": k, "score": v} for k, v in scores.items()]}
    return rafeeq_scores


def run_meta(ctx: dict) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:                                                    # noqa: BLE001
        sha = ""
    return {"live": bool(ctx.get("live")), "git": sha, "llm_provider": os.getenv("LLM_PROVIDER", ""),
            "generator_model": os.getenv("GROQ_MODEL", "") if os.getenv("LLM_PROVIDER", "groq") == "groq" else os.getenv("GEMINI_MODEL", ""),
            "judge_model": os.getenv("EVAL_JUDGE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"}


def run_experiment(mod, ctx: dict) -> str | None:
    """Sync the dataset and run the experiment in LangSmith. Returns the experiment name (None = nothing to run)."""
    from langsmith import evaluate

    client = _client()
    name = sync_dataset(client, mod)
    why = observe.why_not(mod, ctx) if is_case_suite(mod) else ""
    examples = [ex for ex in client.list_examples(dataset_name=name)
                if not why or (ex.inputs or {}).get("checks")]              # no --live: only the model-free checks
    if not examples:
        return None
    res = evaluate(make_target(mod, ctx), data=examples, evaluators=[make_evaluator(mod)],
                   experiment_prefix=mod.NAME, description=mod.DESCRIPTION,
                   metadata={**run_meta(ctx), "suite": mod.NAME, "skipped_cases": why}, max_concurrency=0)
    return res.experiment_name


# ------------------------------------------------------------------ 3. read the numbers back from LangSmith
def _feedback(client, run_ids: list, expected: int, wait: float = 90.0) -> list:
    """Feedback is written asynchronously; wait until every run has some (or the wait is over)."""
    deadline = time.time() + wait
    while True:
        fb = list(client.list_feedback(run_ids=run_ids)) if run_ids else []
        if len({f.run_id for f in fb}) >= expected or time.time() > deadline:
            return fb
        time.sleep(5)


def read_experiment(mod, experiment: str, ctx: dict) -> dict:
    """Everything the report needs about one experiment, from the LangSmith API."""
    client = _client()
    project = client.read_project(project_name=experiment, include_stats=True)
    runs = list(client.list_runs(project_name=experiment, is_root=True))
    case_runs = [r for r in runs if not (r.inputs or {}).get("checks")]
    check_runs = [r for r in runs if (r.inputs or {}).get("checks")]
    scored = [r for r in runs if not r.error]
    fb = _feedback(client, [r.id for r in runs], expected=len(scored))
    by_run: dict = {}
    for f in fb:
        if f.score is not None:
            by_run.setdefault(f.run_id, {})[f.key] = float(f.score)

    metrics: list[Metric] = []
    if is_case_suite(mod):
        why = (_safe(lambda: project.metadata) or {}).get("skipped_cases") or ""
        if case_runs:
            scores: dict[str, list[float]] = {}
            for r in case_runs:
                for k, v in by_run.get(r.id, {}).items():
                    scores.setdefault(k, []).append(v)
            seconds = [(r.end_time - r.start_time).total_seconds() for r in case_runs if r.end_time and not r.error]
            metrics += observe.metrics_from(mod.SPEC, scores, seconds, sum(1 for r in case_runs if r.error), len(case_runs))
        else:
            metrics += observe.skipped(mod.SPEC, why or observe.why_not(mod, ctx) or "no case ran")
    for r in check_runs:                                             # model-free checks: meta from the run, value from feedback
        values = by_run.get(r.id, {})
        for m in (r.outputs or {}).get("metrics", []):
            v = values.get(m["name"], None)
            metrics.append(Metric(m["name"], v if v is not None else m["value"], m["unit"], m["higher_is_better"],
                                  m["gate"], m["note"], m.get("detail") or {}))

    cases = []
    for r in case_runs:
        cases.append({"id": (r.inputs or {}).get("case_id"), "scores": by_run.get(r.id, {}), "error": (r.error or "")[:300],
                      "seconds": round((r.end_time - r.start_time).total_seconds(), 2) if r.end_time else None,
                      "url": _safe(lambda: client.get_run_url(run=r, project_name=experiment))})
    stats = {"runs": project.run_count, "latency_p50_s": _secs(project.latency_p50), "latency_p99_s": _secs(project.latency_p99),
             "total_tokens": project.total_tokens, "total_cost": float(project.total_cost) if project.total_cost is not None else None,
             "error_rate": project.error_rate}
    return {"experiment": experiment, "dataset": dataset_name(mod), "url": _safe(lambda: project.url) or "",
            "metrics": metrics, "cases": cases, "stats": stats}


def latest_experiment(mod) -> str | None:
    client = _client()
    name = dataset_name(mod)
    if not client.has_dataset(dataset_name=name):
        return None
    projects = list(client.list_projects(reference_dataset_name=name))
    projects.sort(key=lambda p: p.start_time, reverse=True)
    return projects[0].name if projects else None


def _secs(td) -> float | None:
    return round(td.total_seconds(), 2) if td is not None else None


def _safe(fn):
    try:
        return fn()
    except Exception:                                                    # noqa: BLE001
        return None


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None

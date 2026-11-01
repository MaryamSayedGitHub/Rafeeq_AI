"""Monitoring IN Langfuse: how each instructor feature behaves on its real traces, read from the Langfuse Metrics API.

The app sends every feature's traces to Langfuse (rafeeq/observability.py: LangChain callback for the LangGraph parts,
@traced for the direct model calls). The evaluation run marks its own traces with the environment "evaluation"
(LANGFUSE_TRACING_ENVIRONMENT), so the report shows real use ("default" = the running app) and the evaluation run apart.

Per feature and environment, Langfuse computes: number of traces, latency p50 / p95, share of traces that ended with an
error, tokens and cost, and the average of every score on those traces (its own LLM-as-a-judge evaluators).

The Metrics API is rate-limited per day on the free plan (100 queries), so the whole monitoring is FOUR grouped queries
(by environment and trace name, never one per feature), the judge scores ONE, and the first "rate limited" answer
stops every further query of the run.
"""
from __future__ import annotations

import datetime as dt
import json

from rafeeq import observability

# feature (as on the slides) -> the trace names of its root observations in Langfuse
FEATURES = [
    ("Lesson studio", "lesson_studio", ["LangGraph", "lesson_studio.draft"]),
    ("Video discovery", "video_discovery", ["video_discovery.search"]),
    ("Animated video · planner", "story_video", ["story_video.plan"]),
    ("Teacher inbox · drafts", "inbox", ["inbox.draft", "inbox.assistant"]),
    ("Smart montage · agent team", "montage", ["montage-team"]),
    ("Smart montage · captions", "montage", ["montage.transcribe"]),
    ("Voice to Visual (future) · planner", "voice_visual", ["voice_visual.plan"]),
]

ROOT = {"column": "isRootObservation", "operator": "=", "value": True, "type": "boolean"}
ERROR = {"column": "level", "operator": "any of", "value": ["ERROR"], "type": "stringOptions"}


class RateLimited(RuntimeError):
    """Langfuse refused: the Metrics API quota of the plan is used up (it says when it resets)."""


_LIMITED: dict = {}


def _query(client, view: str, metrics: list[tuple[str, str]], filters: list[dict], frm: str, to: str,
           dims: list[str] | None = None) -> list[dict]:
    if _LIMITED:
        raise RateLimited(_LIMITED["message"])
    body = {"view": view, "dimensions": [{"field": d} for d in (dims or [])],
            "metrics": [{"measure": m, "aggregation": a} for m, a in metrics],
            "filters": filters, "fromTimestamp": frm, "toTimestamp": to, "config": {"row_limit": 1000}}
    try:
        return list(client.api.metrics.metrics(query=json.dumps(body)).data)
    except Exception as e:                                                       # noqa: BLE001
        if getattr(e, "status_code", None) == 429:
            details = (getattr(e, "body", None) or {}).get("details", {}) if isinstance(getattr(e, "body", None), dict) else {}
            _LIMITED["message"] = (f"Langfuse Metrics API quota used up ({details.get('limit', '?')} queries); it resets at "
                                   f"{details.get('resetAt', '?')}. The traces and scores are still in Langfuse; "
                                   "rebuild later with: uv run python evals/run_all.py --report-only")
            raise RateLimited(_LIMITED["message"]) from None
        raise


def _pick(row: dict, measure: str, agg: str):
    """Langfuse names result columns after the aggregation and the measure; accept its spellings."""
    for key in (f"{agg}_{measure}", f"{measure}_{agg}", f"{agg}{measure[0].upper()}{measure[1:]}", measure):
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def _ms(v):
    return round(v / 1000, 2) if v is not None else None


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def judge_scores(since: dt.datetime, environment: str = "evaluation") -> dict[str, dict]:
    """{suite: {name, avg, n}}: the scores Langfuse's own LLM-as-a-judge evaluators (evals/langfuse_eval.py) gave to the
    traces of one environment since `since`. ONE Metrics API query."""
    from evals.langfuse_eval import JUDGES, PREFIX

    client = observability.langfuse_client()
    if client is None:
        return {}
    to = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=1)
    rows = _query(client, "scores-numeric", [("value", "avg"), ("count", "count")],
                  [{"column": "environment", "operator": "any of", "value": [environment], "type": "stringOptions"}],
                  _iso(since), _iso(to), ["name"])
    by_name = {r.get("name"): r for r in rows}
    out = {}
    for suite, (title, _, _) in JUDGES.items():
        r = by_name.get(PREFIX + title)
        if r:
            out[suite] = {"name": PREFIX + title, "avg": _pick(r, "value", "avg"), "n": int(_pick(r, "count", "count") or 0)}
    return out


def collect(days: int = 30) -> dict:
    """Every feature's numbers for the last `days` days, straight from Langfuse (4 queries in all)."""
    client = observability.langfuse_client()
    if client is None:
        return {"error": "Langfuse is not configured (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY, `uv add langfuse`)"}
    to = dt.datetime.now(dt.timezone.utc)
    frm, to_s = _iso(to - dt.timedelta(days=days)), _iso(to)
    dims = ["environment", "traceName"]
    stats: dict[tuple, dict] = {}

    def at(row) -> dict:
        return stats.setdefault((row.get("environment") or "default", row.get("traceName") or ""), {})

    try:
        for row in _query(client, "observations", [("count", "count"), ("latency", "p50"), ("latency", "p95")], [ROOT], frm, to_s, dims):
            at(row).update(traces=_pick(row, "count", "count"), p50_s=_ms(_pick(row, "latency", "p50")),
                           p95_s=_ms(_pick(row, "latency", "p95")))
        for row in _query(client, "observations", [("count", "count")], [ROOT, ERROR], frm, to_s, dims):
            at(row)["errors"] = _pick(row, "count", "count")
        for row in _query(client, "observations", [("totalTokens", "sum"), ("totalCost", "sum")], [], frm, to_s, dims):
            at(row).update(tokens=_pick(row, "totalTokens", "sum"), cost_usd=_pick(row, "totalCost", "sum"))
        for row in _query(client, "scores-numeric", [("value", "avg"), ("count", "count")], [], frm, to_s, dims + ["name"]):
            at(row).setdefault("scores", {})[row.get("name")] = {"avg": _pick(row, "value", "avg"),
                                                                  "n": _pick(row, "count", "count")}
    except RateLimited as e:
        return {"error": str(e), "days": days}

    known = {n for _, _, names in FEATURES for n in names}
    rows = []
    for title, suite, names in FEATURES:
        by_env: dict[str, dict] = {}
        for (env, name), d in stats.items():
            if name in names and d.get("traces"):
                e = by_env.setdefault(env, {"traces": 0, "errors": 0, "tokens": 0, "cost_usd": 0.0, "scores": {}})
                e["traces"] += d.get("traces") or 0
                e["errors"] += d.get("errors") or 0
                e["tokens"] += d.get("tokens") or 0
                e["cost_usd"] += d.get("cost_usd") or 0
                e["p50_s"], e["p95_s"] = d.get("p50_s"), d.get("p95_s")          # one trace name per feature in practice
                e["scores"].update(d.get("scores") or {})
        for e in by_env.values():
            e["error_rate_pct"] = round(100 * e["errors"] / e["traces"], 1) if e["traces"] else None
        rows.append({"feature": title, "suite": suite, "filter": "trace name = " + ", ".join(names), "by_environment": by_env})
    others = sorted({name for (_, name), d in stats.items() if d.get("traces") and name not in known})
    return {"from": frm, "to": to_s, "days": days, "features": rows, "other_trace_names": others}

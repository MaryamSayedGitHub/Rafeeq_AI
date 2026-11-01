"""What the case-based suites share.

A CASE SUITE (one per LLM feature) is a module in evals/suites/ with

    NAME, DESCRIPTION
    LIVE = True                     # calls a model / the internet: its cases only run with --live
    cases() -> [{"id", "input", "expected"}]           -> the LangSmith dataset "rafeeq-m5-<NAME>"
    target(input) -> dict           # calls the real feature code   -> what LangSmith's evaluate() runs per example
    SCORERS = {key: fn(input, output, expected) -> float | None}     -> the evaluator LangSmith runs on every run
    SPEC = [Spec(...)]              # which feedback key becomes which number in the report (mean of the key)
    missing() -> str                # optional: why it cannot run on this machine ("" = it can)
    offline(ctx) -> list[Metric]    # optional: checks with no model, run every time (one extra dataset example)
    run(ctx)                        # local, for pytest only: same cases + scorers, no platform involved

The numbers in the report are NOT computed here: evals/langsmith_eval.py runs the experiment IN LangSmith and reads
the feedback back from LangSmith. This file only has the shared helpers and the local runner used by the tests.
"""
from __future__ import annotations

import statistics
import time
import traceback

from evals.core import Metric, Spec


# ------------------------------------------------------------------ text helpers used by scorers
def arabic_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if "؀" <= c <= "ۿ") / len(letters) if letters else 0.0


def in_language(text: str, language: str) -> bool:
    """Arabic material: at least half the letters are Arabic (code words and names are English).
    English material: almost no Arabic letters."""
    share = arabic_share(text)
    return share >= 0.5 if language.strip().lower() in ("ar", "arabic", "العربية") else share <= 0.05


# ------------------------------------------------------------------ one place that turns per-case scores into metrics
def metrics_from(spec: list[Spec], scores: dict[str, list[float]], seconds: list[float], crashed: int, total: int) -> list[Metric]:
    """spec + per-key score lists (from LangSmith feedback, or from the local runner) -> report metrics."""
    out = []
    for s in spec:
        if s.key == "@median_seconds":
            value = round(statistics.median(seconds), 2) if seconds else None
        elif s.key == "@crashed":
            value = round(100 * crashed / total, 1) if total else None
        else:
            vals = scores.get(s.key) or []
            value = round(s.scale * sum(vals) / len(vals), 4 if s.scale == 1 else 1) if vals else None
        out.append(Metric(s.name, value, s.unit, s.higher_is_better, s.gate, s.note,
                          {"cases": len(scores.get(s.key) or [])} if not s.key.startswith("@") else {}))
    return out


def skipped(spec: list[Spec], why: str) -> list[Metric]:
    return [Metric(s.name, None, s.unit, s.higher_is_better, s.gate, f"{why} · {s.note}") for s in spec]


def score_case(mod, inp: dict, out, expected) -> dict[str, float]:
    """Every scorer of a suite on one output (the LangSmith evaluator calls this). A broken scorer = no score."""
    res = {}
    for name, fn in mod.SCORERS.items():
        try:
            value = fn(inp, out, expected)
        except Exception as e:                                           # noqa: BLE001
            print(f"  scorer {name} failed: {type(e).__name__}: {e}")
            value = None
        if value is not None:
            res[name] = round(float(value), 4)
    return res


def why_not(mod, ctx: dict) -> str:
    if getattr(mod, "LIVE", True) and not ctx.get("live"):
        return "needs --live (calls the model)"
    return mod.missing() if hasattr(mod, "missing") else ""


# ------------------------------------------------------------------ local runner (pytest; no platform)
def run_case_suite(mod, ctx: dict) -> list[Metric]:
    extra = list(mod.offline(ctx)) if hasattr(mod, "offline") else []
    why = why_not(mod, ctx)
    if why:
        return skipped(mod.SPEC, why) + extra
    scores: dict[str, list[float]] = {}
    seconds, crashed, rows = [], 0, []
    for case in mod.cases():
        t0 = time.perf_counter()
        try:
            out = mod.target(case["input"])
        except Exception as e:                                           # noqa: BLE001 - a failed case is a result
            traceback.print_exc()
            crashed += 1
            rows.append({"id": case["id"], "error": f"{type(e).__name__}: {e}"[:300], "scores": {}})
            continue
        seconds.append(time.perf_counter() - t0)
        s = score_case(mod, case["input"], out, case.get("expected"))
        rows.append({"id": case["id"], "error": "", "scores": s})
        for k, v in s.items():
            scores.setdefault(k, []).append(v)
    ctx.setdefault("rows", {})[mod.NAME] = rows
    return metrics_from(mod.SPEC, scores, seconds, crashed, len(rows)) + extra

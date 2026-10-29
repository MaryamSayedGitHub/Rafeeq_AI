"""What every suite shares: a Metric, the metric helpers (WER, overlap), the gate check and the run-to-run comparison.

A suite is a module in evals/suites/ with `NAME` (str), `DESCRIPTION` (str) and `run(ctx) -> list[Metric]`.
Nothing here needs a model or the internet; a suite that needs one declares it and says SKIPPED when it is missing.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"


@dataclass
class Metric:
    name: str
    value: float | None                 # None = SKIPPED (why: `note`)
    unit: str = ""                      # "", "%", "s", ...
    higher_is_better: bool = True
    gate: float | None = None           # pass if value >= gate (or <= gate when lower is better)
    note: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.value is None:
            return "SKIPPED"
        if self.gate is None:
            return "INFO"
        return "PASS" if (self.value >= self.gate if self.higher_is_better else self.value <= self.gate) else "FAIL"

    def as_dict(self) -> dict:
        return {**asdict(self), "status": self.status}


# ------------------------------------------------------------------ helpers
def overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total seconds covered by BOTH lists of (start, end) ranges."""
    total = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def length(ranges: list[tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in ranges)


def complement(ranges: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    out, cur = [], 0.0
    for s, e in sorted(ranges):
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < total:
        out.append((cur, total))
    return out


def _words(text: str) -> list[str]:
    text = re.sub(r"[ً-ْـ]", "", text.lower())            # Arabic diacritics and tatweel do not count
    text = re.sub(r"[أإآ]", "ا", text).replace("ة", "ه").replace("ى", "ي")
    return re.findall(r"[\w']+", text)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate (edit distance over words / reference words), light Arabic normalisation. 0 = perfect."""
    r, h = _words(reference), _words(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


def mean_abs_offset(truth: list[float], found: list[float]) -> float | None:
    """For each true time, the distance to the nearest found time; the mean of those. None when there is nothing to compare."""
    if not truth or not found:
        return None
    return sum(min(abs(t - f) for f in found) for t in truth) / len(truth)


# ------------------------------------------------------------------ results on disk
def save_run(payload: dict, stamp: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"run-{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / "latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_latest() -> dict | None:
    f = RESULTS / "latest.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except (OSError, ValueError):
        return None


def index(run: dict | None) -> dict[tuple[str, str], dict]:
    return {(s["name"], m["name"]): m for s in (run or {}).get("suites", []) for m in s["metrics"]}


def change(old: dict | None, new: dict) -> str:
    """'+0.02' / '-3.1' / '' : how a metric moved since the previous run (only when both ran)."""
    if not old or old.get("value") is None or new.get("value") is None:
        return ""
    d = new["value"] - old["value"]
    return "" if abs(d) < 1e-9 else f"{d:+.3g}"

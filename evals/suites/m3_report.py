"""Handwriting and tutor reliability (M3). Reads the report M3's own tool already writes (results/report.json) and
shows its headline numbers next to everybody else's. To refresh them run M3's command (README, "M3 reliability"):
    uv run python -m rafeeq.snap.report --run-cer ...
This suite does not re-run it (it needs the OCR / vision models), so the numbers carry the date they were made."""
from __future__ import annotations

import json

from evals.core import ROOT, Metric

NAME = "m3_handwriting"
DESCRIPTION = "CER, step-status accuracy, first-error localisation, hint leaks, citation validity (from M3's report)"


def run(ctx: dict) -> list[Metric]:
    f = ROOT / "results" / "report.json"
    if not f.exists():
        return [Metric("m3_report", None, note="results/report.json not found; run M3's report first")]
    d = json.loads(f.read_text(encoding="utf-8"))
    stamp = str(d.get("generated", "?"))[:10]
    out: list[Metric] = []
    cer = (d.get("cer") or {}).get("best_per_reader") or {}
    if (d.get("cer") or {}).get("status") == "RAN" and cer:
        for reader, row in cer.items():
            val = row.get("mean_cer") if isinstance(row, dict) else None
            if isinstance(val, (int, float)):
                out.append(Metric(f"cer_best_{reader}", float(val), "", higher_is_better=False,
                                  note=f"report of {stamp}; data: {(d.get('cer') or {}).get('data', '?')}"))
    w = d.get("work") or {}
    for key, label, gate in (("step_status_accuracy", "step_status_accuracy", 0.85), ("first_error_localization", "first_error_localization", 0.8),
                             ("hint_relevance_keyword_proxy", "hint_relevance", None), ("uncertainty_handling", "uncertainty_handling", 0.9)):
        r = (w.get(key) or {}).get("rate")
        out.append(Metric(label, r, "", gate=gate, note=f"report of {stamp}; {w.get('cases', '?')} developer-written cases") if r is not None
                   else Metric(label, None, note="not in the report"))
    leaks = w.get("hint_leaks") or {}
    if leaks:
        out.append(Metric("hint_leaks", float(sum(v for v in leaks.values() if isinstance(v, (int, float)))), "", higher_is_better=False, gate=0.0,
                          note="hints that give away the solution; must be 0"))
    c = (d.get("citations") or {}).get("citation_validity") or {}
    if c.get("rate") is not None:
        out.append(Metric("citation_validity", c["rate"], "", gate=0.95, note=f"report of {stamp}; material source: {(d.get('citations') or {}).get('material_source', '?')}"))
    return out or [Metric("m3_report", None, note="the report has no numbers yet")]

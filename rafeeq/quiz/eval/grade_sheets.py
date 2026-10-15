"""Measure bubble accuracy on real photographed sheets.

Put the photos in one folder and write what you really marked in truth.json:
    {"sheet1.jpg": [0, 2, 1, null, 3], "sheet2.jpg": [1, 1, 0, 2, 2]}
(0..3 = A..D, null = left blank). Then:

    uv run python -m rafeeq.quiz.eval.grade_sheets --dir data/sheets --truth data/sheets/truth.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rafeeq.quiz.sheet import grader


def evaluate(folder: Path, truth: dict[str, list]) -> dict:
    rows = []
    for name, expected in truth.items():
        path = folder / name
        if not path.exists():
            rows.append({"sheet": name, "error": "photo file not found", "total": len(expected), "correct": 0, "flagged": 0})
            continue
        try:
            marks = grader.detect_marks(path.read_bytes(), len(expected))
        except grader.SheetError as exc:
            rows.append({"sheet": name, "error": str(exc), "total": len(expected), "correct": 0, "flagged": 0})
            continue
        correct = sum(m.choice == t for m, t in zip(marks, expected))
        flagged = sum(m.choice is None or m.confidence < 0.5 for m in marks)
        wrong = [i + 1 for i, (m, t) in enumerate(zip(marks, expected)) if m.choice != t]
        rows.append({"sheet": name, "total": len(expected), "correct": correct, "flagged": flagged, "wrong_questions": wrong})
    total = sum(r["total"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "sheets": len(rows),
        "sheets_unreadable": sum("error" in r for r in rows),
        "bubble_accuracy": correct / total if total else 0.0,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/sheets")
    ap.add_argument("--truth", default="data/sheets/truth.json")
    ap.add_argument("--report", default="data/sheets/report.json")
    a = ap.parse_args()

    result = evaluate(Path(a.dir), json.loads(Path(a.truth).read_text(encoding="utf-8")))
    print(f"{'sheet':<24}{'correct':>10}{'flagged':>9}  notes")
    for r in result["rows"]:
        note = r.get("error") or (f"wrong on Q{r['wrong_questions']}" if r["wrong_questions"] else "")
        print(f"{r['sheet']:<24}{r['correct']:>4}/{r['total']:<5}{r['flagged']:>9}  {note}")
    print(f"\nbubble accuracy: {result['bubble_accuracy']:.1%} over {result['sheets']} sheets "
          f"({result['sheets_unreadable']} unreadable)")
    Path(a.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", a.report)


if __name__ == "__main__":
    main()
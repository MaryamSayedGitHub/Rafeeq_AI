"""Evaluation of the extra question types: true/false, short answer, maths, problem, essay.

The unit tests say the code does what we wrote. This says how GOOD it is. Five checks, each answers one question:

  math        Does the maths checker accept equal answers, reject different ones, and refuse unreadable text safely?
              No model, no key, runs in seconds.
  grader      Does the answer grader (LLM + rubric) agree with a person? Pass/fail agreement, kappa, average error.
  injection   Can a student talk the grader into marks? (answers that are only instructions)
  generate    Per question type: how often does the model write a valid question (first try, after one retry)?
              Saves the questions for a person to read.
  review      Counts the verdicts a person wrote in the review file made by `generate`.

Run (from the repo root, with the .env of the app so LLM_API_KEY, LLM_BASE_URL and LLM_MODEL are found):

  uv run python -m rafeeq.quiz.eval.eval_kinds math
  uv run python -m rafeeq.quiz.eval.eval_kinds grader
  uv run python -m rafeeq.quiz.eval.eval_kinds injection
  uv run python -m rafeeq.quiz.eval.eval_kinds generate --n 12
  uv run python -m rafeeq.quiz.eval.eval_kinds review data/eval_kinds_review.csv

The reference cases live in rafeeq/quiz/eval/cases/*.json. They are a starter set written by hand: have a person check
the labels before quoting the numbers, and add your own cases.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Callable, Optional

from rafeeq.quiz.contracts import ALL_KINDS, QuizItem
from rafeeq.quiz.quiz import grading
from rafeeq.quiz.quiz.grading import PASS_MARK, grade_typed_answer

CASES = Path(__file__).parent / "cases"
LLM = Callable[[str], str]


def load_cases(name: str) -> dict:
    return json.loads((CASES / f"{name}.json").read_text(encoding="utf-8"))


def make_item(item_id: str, spec: dict) -> QuizItem:
    base = dict(item_id=item_id, topic_id="eval", explanation="reference case", file_name="eval.pdf", page=1)
    return QuizItem(**{**base, **spec})


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------- 1. maths (no model)
def run_math(cases: dict, equal: Callable[[str, str], bool] = grading.math_equal, parse: Callable = grading.parse_math) -> dict:
    """Every pair is judged by the checker; `equal` and `parse` can be swapped in tests."""
    scored = [(s, e, True) for s, e in cases["same"]] + [(s, e, False) for s, e in cases["different"]]
    wrong_accepts, wrong_rejects, unreadable, slowest = [], [], [], 0.0
    for student, expected, want in scored:
        start = time.time()
        try:
            got = equal(student, expected)
        except grading.MathParseError:
            unreadable.append({"student": student, "expected": expected, "want": want})
            continue
        finally:
            slowest = max(slowest, time.time() - start)
        if got and not want:
            wrong_accepts.append({"student": student, "expected": expected})
        elif want and not got:
            wrong_rejects.append({"student": student, "expected": expected})
    n = len(scored)
    right = n - len(wrong_accepts) - len(wrong_rejects) - len(unreadable)

    unsafe_failures = []
    for text in cases["unreadable"]:
        start = time.time()
        try:
            parse(text)
            unsafe_failures.append({"text": text[:60], "problem": "was parsed"})
        except grading.MathParseError:
            pass
        except ImportError:
            raise
        except Exception as exc:  # anything else means the checker did not refuse cleanly
            unsafe_failures.append({"text": text[:60], "problem": f"{type(exc).__name__}"})
        slowest = max(slowest, time.time() - start)

    gaps = []
    for g in cases.get("known_gaps", []):
        try:
            got = equal(g["student"], g["expected"])
        except grading.MathParseError:
            got = None
        gaps.append({**g, "now": "accepted" if got else "still not accepted"})
    return {"scored": n, "accuracy": right / n if n else 0.0, "wrong_accepts": wrong_accepts,
            "wrong_rejects": wrong_rejects, "unreadable_but_should_be_judged": unreadable,
            "unsafe_total": len(cases["unreadable"]), "unsafe_failures": unsafe_failures,
            "slowest_seconds": round(slowest, 3), "known_gaps": gaps}


# ---------------------------------------------------------------- 2. grader vs a person
def kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two yes/no judges: 1 = same verdicts, 0 = no better than chance."""
    n = len(a)
    if not n:
        return 0.0
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


class _Spy:
    """Wraps the model: counts the calls and keeps the last raw reply (to read the per-point verdicts)."""

    def __init__(self, llm: Optional[LLM]):
        self.llm, self.calls, self.last = llm, 0, ""

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.last = ""
        self.last = self.llm(prompt)
        return self.last


def run_grader(cases: dict, llm: Optional[LLM], pace: float = 0.0) -> dict:
    rows = []
    for i, c in enumerate(cases["answers"]):
        item = make_item(c["item"], cases["items"][c["item"]])
        spy = _Spy(llm) if llm else None
        if pace and spy:
            time.sleep(pace)
        try:
            g = grade_typed_answer(item, c["answer"], spy)
        except Exception as exc:  # the model call itself failed
            rows.append({"n": i + 1, "item": c["item"], "kind": item.kind, "answer": c["answer"], "error": str(exc)[:120]})
            continue
        row = {"n": i + 1, "item": c["item"], "kind": item.kind, "answer": c["answer"], "model": g.score,
               "needs_review": g.needs_review, "feedback": g.feedback, "model_calls": spy.calls if spy else 0}
        if "human_met" in c:
            row["human"] = round(sum(c["human_met"]) / len(c["human_met"]), 2)
            raw = grading._parse_json(spy.last) if spy and spy.last else None
            met = raw.get("met") if raw else None
            if isinstance(met, list) and len(met) == len(c["human_met"]):
                row["points_agree"] = sum((m is True) == h for m, h in zip(met, c["human_met"]))
                row["points_total"] = len(met)
        else:
            row["human"] = 1.0 if c["human_same"] else 0.0
        rows.append(row)

    ok = [r for r in rows if "error" not in r and not r["needs_review"]]
    rubric = [r for r in ok if r["kind"] in ("essay", "problem")]
    short = [r for r in ok if r["kind"] == "short"]
    pts = [(r["points_agree"], r["points_total"]) for r in rubric if "points_agree" in r]

    def passes(r, who):
        return r[who] >= PASS_MARK

    return {
        "cases": len(rows), "graded": len(ok), "needs_review": sum(1 for r in rows if r.get("needs_review")),
        "errors": [r for r in rows if "error" in r],
        "model_calls": sum(r.get("model_calls", 0) for r in rows),
        "short_without_model": sum(1 for r in short if r["model_calls"] == 0),
        "pass_agreement": sum(passes(r, "model") == passes(r, "human") for r in ok) / len(ok) if ok else 0.0,
        "kappa": round(kappa([passes(r, "model") for r in ok], [passes(r, "human") for r in ok]), 2),
        "rubric_mean_abs_error": round(sum(abs(r["model"] - r["human"]) for r in rubric) / len(rubric), 3) if rubric else None,
        "rubric_points_agreement": sum(a for a, _ in pts) / sum(t for _, t in pts) if pts else None,
        "short_accuracy": sum(passes(r, "model") == passes(r, "human") for r in short) / len(short) if short else None,
        "disagreements": [r for r in ok if passes(r, "model") != passes(r, "human")
                          or (r["kind"] in ("essay", "problem") and abs(r["model"] - r["human"]) > 0.34)],
    }


# ---------------------------------------------------------------- 3. can a student talk the grader into marks?
def run_injection(cases: dict, llm: Optional[LLM], pace: float = 0.0) -> dict:
    rows = []
    for c in cases["attacks"]:
        item = make_item(c["item"], cases["items"][c["item"]])
        if pace:
            time.sleep(pace)
        try:
            g = grade_typed_answer(item, c["answer"], llm)
        except Exception as exc:
            rows.append({"item": c["item"], "answer": c["answer"][:80], "error": str(exc)[:120]})
            continue
        rows.append({"item": c["item"], "answer": c["answer"][:80], "score": g.score, "needs_review": g.needs_review})
    done = [r for r in rows if "error" not in r and not r["needs_review"]]
    return {"attacks": len(rows), "judged": len(done),
            "fooled": [r for r in done if r["score"] >= PASS_MARK],
            "some_marks": [r for r in done if 0 < r["score"] < PASS_MARK],
            "needs_review": sum(1 for r in rows if r.get("needs_review")),
            "errors": [r for r in rows if "error" in r]}


# ---------------------------------------------------------------- 4. does the model write good questions of each type?
def run_generation(llm: LLM, chunks: list[dict], kinds: list[str], n: int, seed: int = 0, pace: float = 0.0) -> dict:
    """The same pages for every type, so the types are comparable."""
    from rafeeq.quiz.quiz.generator import generate_item, select_chunks

    pages = select_chunks(chunks, n, rng=random.Random(seed))
    out: dict[str, dict] = {}
    samples: list[dict] = []
    for kind in kinds:
        first = produced = errors = streak = 0
        seconds: list[float] = []
        for i, chunk in enumerate(pages, 1):
            if pace:
                time.sleep(pace)
            start = time.time()
            try:
                item, first_try = generate_item(llm, chunk, f"{kind}-{i}", random.Random(seed + i), kind=kind)
                streak = 0
            except Exception as exc:  # rate limit, timeout, bad key
                errors += 1
                streak += 1
                print(f"  [{kind}] page {chunk['page']}: {type(exc).__name__}: {str(exc)[:120]}")
                if streak >= 3:
                    print(f"  [{kind}] 3 errors in a row, skipping this type")
                    break
                continue
            seconds.append(time.time() - start)
            first += int(first_try)
            if item is not None:
                produced += 1
                samples.append({"kind": kind, "item_id": item.item_id, "page": item.page, "item": item.model_dump()})
        out[kind] = {"tried": len(pages), "first_try": first / len(pages) if pages else 0.0,
                     "produced": produced / len(pages) if pages else 0.0, "errors": errors,
                     "seconds_per_item": round(sum(seconds) / len(seconds), 1) if seconds else 0.0}
    return {"kinds": out, "samples": samples}


def write_review_csv(samples: list[dict], path: str) -> None:
    """One row per question for a person to fill: human_ok = y or n (is it a good question with a right key?)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig so Excel shows Arabic
        w = csv.writer(f)
        w.writerow(["item_id", "kind", "page", "question", "options", "answer", "rubric", "human_ok", "note"])
        for s in samples:
            it = s["item"]
            options = it.get("options") or []
            answer = it.get("answer", "")
            if options and not answer:        # choice questions keep the key as an index: show it in the answer column
                i = it.get("right_choice", 0)
                answer = options[i] if 0 <= i < len(options) else ""
            w.writerow([s["item_id"], s["kind"], s["page"], it["stem"], " | ".join(options),
                        answer, " | ".join(it.get("rubric") or []), "", ""])


def write_reading_text(samples: list[dict], path: str) -> None:
    """The same questions as plain text, easy to read in any editor (the right option is marked with >>)."""
    lines: list[str] = []
    for s in samples:
        it = s["item"]
        lines += ["=" * 70, f"{s['item_id']}   [{s['kind']}]   page {s['page']}", "", it["stem"], ""]
        options = it.get("options") or []
        for i, o in enumerate(options):
            lines.append(f"  {'>>' if i == it.get('right_choice') and it.get('kind') in ('mcq', 'true_false') else '  '} {o}")
        if it.get("answer"):
            lines += ["", "ANSWER KEY:", it["answer"]]
        for i, r in enumerate(it.get("rubric") or [], 1):
            lines.append(f"  rubric {i}. {r}")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def summarize_bad(samples: list[dict], bad: list[str]) -> dict:
    """Every question counts as good except the ids a person listed as bad."""
    known = {s["item_id"] for s in samples}
    unknown = [b for b in bad if b not in known]
    if unknown:
        raise SystemExit(f"these ids are not in the report: {unknown}")
    per: dict[str, list[bool]] = {}
    for s in samples:
        per.setdefault(s["kind"], []).append(s["item_id"] not in bad)
    return {k: {"reviewed": len(v), "good": sum(v), "good_rate": sum(v) / len(v)} for k, v in per.items()}


def summarize_review(path: str) -> dict:
    per: dict[str, list[bool]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            v = (row.get("human_ok") or "").strip().lower()
            if v in ("y", "yes", "1", "ok", "✓"):
                per.setdefault(row["kind"], []).append(True)
            elif v in ("n", "no", "0", "x", "✗"):
                per.setdefault(row["kind"], []).append(False)
    return {k: {"reviewed": len(v), "good": sum(v), "good_rate": sum(v) / len(v)} for k, v in per.items()}


# ---------------------------------------------------------------- printing and the command line
def _show(title: str, lines: list[str]) -> None:
    print(f"\n=== {title} ===")
    print("\n".join(lines))


def print_math(r: dict) -> None:
    _show("maths checker", [
        f"scored pairs:               {r['scored']}",
        f"accuracy:                   {pct(r['accuracy'])}",
        f"accepted a WRONG answer:    {len(r['wrong_accepts'])}",
        f"rejected a RIGHT answer:    {len(r['wrong_rejects'])}",
        f"could not read (scored):    {len(r['unreadable_but_should_be_judged'])}",
        f"unsafe/unreadable refused:  {r['unsafe_total'] - len(r['unsafe_failures'])}/{r['unsafe_total']}",
        f"slowest single check:       {r['slowest_seconds']} s",
    ])
    for key, label in (("wrong_accepts", "WRONG ACCEPT"), ("wrong_rejects", "WRONG REJECT"), ("unreadable_but_should_be_judged", "UNREADABLE"),
                       ("unsafe_failures", "NOT REFUSED")):
        for x in r[key]:
            print(f"  {label}: {x}")
    for g in r["known_gaps"]:
        print(f"  known gap ({g['now']}): {g['student']!r} vs {g['expected']!r}: {g['why']}")


def print_grader(r: dict) -> None:
    _show("answer grader vs a person", [
        f"cases graded:               {r['graded']}/{r['cases']}  (needs_review: {r['needs_review']}, errors: {len(r['errors'])})",
        f"model calls:                {r['model_calls']}  (short answers matched without a model: {r['short_without_model']})",
        f"pass/fail agreement:        {pct(r['pass_agreement'])}   kappa {r['kappa']}",
        f"short-answer accuracy:      {pct(r['short_accuracy']) if r['short_accuracy'] is not None else 'n/a'}",
        f"rubric points agreement:    {pct(r['rubric_points_agreement']) if r['rubric_points_agreement'] is not None else 'n/a'}",
        f"rubric mean abs. error:     {r['rubric_mean_abs_error']}",
    ])
    for d in r["disagreements"]:
        print(f"  #{d['n']} {d['item']} person {d['human']} model {d['model']}: {d['answer'][:70]!r}\n      {d['feedback'][:110]}")
    for e in r["errors"]:
        print(f"  ERROR #{e['n']}: {e['error']}")


def print_injection(r: dict) -> None:
    _show("prompt injection", [
        f"attacks judged:             {r['judged']}/{r['attacks']}  (needs_review: {r['needs_review']}, errors: {len(r['errors'])})",
        f"FOOLED the grader (>= {PASS_MARK}): {len(r['fooled'])}",
        f"got some marks (0 to {PASS_MARK}): {len(r['some_marks'])}",
    ])
    for x in r["fooled"] + r["some_marks"]:
        print(f"  score {x['score']}: {x['item']} {x['answer']!r}")


def print_generation(r: dict) -> None:
    lines = [f"{'type':<12}{'tried':>6}{'first try':>11}{'produced':>10}{'errors':>8}{'sec/item':>10}"]
    for k, s in r["kinds"].items():
        lines.append(f"{k:<12}{s['tried']:>6}{pct(s['first_try']):>11}{pct(s['produced']):>10}{s['errors']:>8}{s['seconds_per_item']:>10}")
    _show("question generation by type (same pages for every type)", lines)


def _llm(a) -> LLM:
    from rafeeq.quiz.eval.judge_quiz import paced
    from rafeeq.quiz.quiz.llm import make_llm

    return paced(make_llm(a.provider, a.model), a.sleep)


def _save(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nfull report saved to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["math", "grader", "injection", "generate", "review", "sheet"])
    ap.add_argument("path", nargs="?", help="review: the CSV a person filled in; sheet: a saved generate report (.json)")
    ap.add_argument("--provider", default="compatible", help="compatible = the app's own model (LLM_* in .env)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between model calls (free quotas are small)")
    ap.add_argument("--book", default="data/Hands-On_Large_Language_Models.pdf")
    ap.add_argument("--n", type=int, default=12, help="generate: pages per question type")
    ap.add_argument("--kinds", nargs="+", default=[k for k in ALL_KINDS], choices=list(ALL_KINDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bad", nargs="*", default=None, help="review: ids of the BAD questions (all others count as good); path = the generate .json report")
    ap.add_argument("--report", default=None, help="where to save the JSON report (default data/eval_kinds_<check>.json)")
    a = ap.parse_args()
    report = a.report or f"data/eval_kinds_{a.check}.json"

    if a.check == "math":
        try:
            r = run_math(load_cases("math_cases"))
        except ImportError:
            raise SystemExit("sympy is not installed: run `uv sync` first")
        print_math(r)
        _save(report, r)
    elif a.check == "grader":
        r = run_grader(load_cases("grader_cases"), _llm(a))
        print_grader(r)
        _save(report, r)
    elif a.check == "injection":
        r = run_injection(load_cases("injection_cases"), _llm(a))
        print_injection(r)
        _save(report, r)
    elif a.check == "generate":
        from rafeeq.quiz.dev.book_chunks import load_chunks

        r = run_generation(_llm(a), load_chunks(a.book), a.kinds, a.n, a.seed)
        print_generation(r)
        csv_path = report.replace(".json", "_review.csv")
        write_review_csv(r["samples"], csv_path)
        _save(report, r)
        print(f"questions to read: {csv_path}  (write y or n in the human_ok column, then run: eval_kinds review {csv_path})")
    elif a.check == "sheet":
        if not a.path:
            raise SystemExit("give a saved report: eval_kinds sheet data/eval_kinds_generate.json")
        data = json.loads(Path(a.path).read_text(encoding="utf-8"))
        out = a.report or a.path.replace(".json", "_review.csv")
        write_review_csv(data["samples"], out)
        txt = out.replace(".csv", ".txt")
        write_reading_text(data["samples"], txt)
        print(f"review sheet written to {out}\nthe same questions as plain text: {txt}")
    else:
        if not a.path:
            raise SystemExit("give the review CSV: eval_kinds review data/eval_kinds_generate_review.csv")
        if a.bad is not None:
            data = json.loads(Path(a.path).read_text(encoding="utf-8"))
            s = summarize_bad(data["samples"], a.bad)
        else:
            s = summarize_review(a.path)
        _show("human review of generated questions", [f"{k:<12}{v['good']}/{v['reviewed']} good  ({pct(v['good_rate'])})" for k, v in s.items()]
              or ["no verdicts found: fill the human_ok column with y or n"])


if __name__ == "__main__":
    main()

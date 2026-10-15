"""Check question quality with a blind judge.

1. generate N questions from the book (same agent the API uses),
2. show each question + its source page to a judge model WITHOUT the answer key,
3. compare the judge's answer with ours, and ask if the page really supports it.

    uv run python -m rafeeq.quiz.eval.judge_quiz --n 20 --sleep 3
    uv run python -m rafeeq.quiz.eval.judge_quiz --reuse          # judge the saved questions again

A disagreement means either our question is wrong or the judge is: read those
by hand (they are printed and saved) and report the human-checked number.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Callable

LETTERS = "ABCD"
PAGE_CHARS = 6000
STOP_AFTER_ERRORS = 3

JUDGE_PROMPT = """You are checking a multiple-choice question against its source page.

SOURCE PAGE:
\"\"\"
{page}
\"\"\"

QUESTION: {stem}
A) {a}
B) {b}
C) {c}
D) {d}

Answer using ONLY the source page above.
- "answer": the letter (A, B, C or D) of the option the page supports best.
- "supported": true only if the page clearly contains enough information to answer
  and exactly one option is correct; false if the page does not say, or two options
  could be right, or the question is unclear.
- "reason": one short sentence.

Return ONLY JSON: {{"answer": "A", "supported": true, "reason": "..."}}"""


def build_judge_prompt(page_text: str, stem: str, options: list[str]) -> str:
    a, b, c, d = options
    return JUDGE_PROMPT.format(page=page_text[:PAGE_CHARS], stem=stem, a=a, b=b, c=c, d=d)


def parse_judgement(text: str) -> dict | None:
    """-> {"answer": 0..3, "supported": bool, "reason": str} or None if unreadable."""
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return None
    ans = data.get("answer")
    if isinstance(ans, str) and ans.strip().upper()[:1] in LETTERS:
        ans = LETTERS.index(ans.strip().upper()[:1])
    if not isinstance(ans, int) or isinstance(ans, bool) or not 0 <= ans <= 3:
        return None
    supported = data.get("supported")
    if not isinstance(supported, bool):
        return None
    return {"answer": ans, "supported": supported, "reason": str(data.get("reason", ""))[:300]}


def judge_item(llm: Callable[[str], str], page_text: str, item: dict) -> dict:
    """Row with the judge's verdict on one question (a plain dict item)."""
    row = {"item_id": item["item_id"], "topic_id": item["topic_id"], "page": item["page"],
           "stem": item["stem"], "options": item["options"], "key": item["right_choice"]}
    judged = parse_judgement(llm(build_judge_prompt(page_text, item["stem"], item["options"])))
    if judged is None:
        return {**row, "error": "judge answer could not be read"}
    return {**row, "judge": judged["answer"], "supported": judged["supported"],
            "reason": judged["reason"], "agree": judged["answer"] == item["right_choice"]}


def summarize(rows: list[dict], first_try_rate: float | None = None) -> dict:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)

    def pct(k: int) -> float:
        return k / n if n else 0.0

    return {
        "asked": len(rows),
        "judged": n,
        "first_try_validation": first_try_rate,
        "agree": pct(sum(r["agree"] for r in ok)),
        "supported": pct(sum(r["supported"] for r in ok)),
        "solid": pct(sum(r["agree"] and r["supported"] for r in ok)),
        "to_review": [r for r in ok if not (r["agree"] and r["supported"])],
    }


def paced(llm: Callable[[str], str], seconds: float) -> Callable[[str], str]:
    """Wait between calls so a free API quota is not burned in a burst."""
    last = [0.0]

    def call(prompt: str) -> str:
        wait = seconds - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            return llm(prompt)
        finally:
            last[0] = time.time()

    return call


def main() -> None:
    from rafeeq.quiz.dev.book_chunks import load_chunks
    from rafeeq.quiz.quiz.generator import generate_quiz
    from rafeeq.quiz.quiz.llm import make_llm

    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="data/Hands-On_Large_Language_Models.pdf")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gen-provider", default="gemini")
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--judge-provider", default="gemini")
    ap.add_argument("--judge-model", default="gemini-3.8-flash")
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between API calls")
    ap.add_argument("--reuse", action="store_true", help="judge the saved questions, do not generate")
    ap.add_argument("--items-file", default="data/judge_items.json")
    ap.add_argument("--report", default="data/judge_report.json")
    a = ap.parse_args()

    chunks = load_chunks(a.book)
    pages = {(c["file_name"], c["page"]): c["text"] for c in chunks}
    items_file = Path(a.items_file)

    first_try = None
    if a.reuse and items_file.exists():
        saved = json.loads(items_file.read_text(encoding="utf-8"))
        items, first_try = saved["items"], saved.get("first_try_rate")
        print(f"reusing {len(items)} saved questions")
    else:
        gen = paced(make_llm(a.gen_provider, a.gen_model), a.sleep)
        print(f"generating {a.n} questions ...")
        result = generate_quiz(gen, chunks, a.n, rng=random.Random(a.seed))
        items, first_try = [it.model_dump() for it in result.items], result.first_try_rate
        items_file.parent.mkdir(parents=True, exist_ok=True)
        items_file.write_text(json.dumps({"items": items, "first_try_rate": first_try}, indent=2), encoding="utf-8")
        print(f"got {len(items)} questions (first-try validation {first_try:.0%})")

    judge = paced(make_llm(a.judge_provider, a.judge_model), a.sleep)

    # with --reuse, keep the verdicts already obtained and only retry the failed ones
    done: dict[str, dict] = {}
    report_file = Path(a.report)
    if a.reuse and report_file.exists():
        for r in json.loads(report_file.read_text(encoding="utf-8")).get("rows", []):
            if "error" not in r:
                done[r["item_id"]] = r
        print(f"keeping {len(done)} verdicts from the previous run")

    rows: list[dict] = []
    errors_in_a_row = 0
    for i, item in enumerate(items, 1):
        if item["item_id"] in done:
            rows.append(done[item["item_id"]])
            print(f"  judged {i}/{len(items)}  kept")
            continue
        page_text = pages.get((item["file_name"], item["page"]), "")
        try:
            row = judge_item(judge, page_text, item)
            errors_in_a_row = 0
        except Exception as exc:  # quota, network ...
            row = {"item_id": item["item_id"], "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            errors_in_a_row += 1
        rows.append(row)
        status = row["error"] if "error" in row else ("agree" if row["agree"] else "DISAGREE")
        print(f"  judged {i}/{len(items)}  {status}")
        if errors_in_a_row >= STOP_AFTER_ERRORS:
            print("stopping: the judge keeps failing (see the error above). Wait a few minutes, "
                  "or use --judge-model gemini-3.5-flash-lite, then run again with --reuse")
            break

    s = summarize(rows, first_try)
    print("\n=== question quality ===")
    print(f"questions judged:                {s['judged']}/{s['asked']}")
    if s["first_try_validation"] is not None:
        print(f"passed validator first try:      {s['first_try_validation']:.0%}")
    print(f"judge picks our answer key:      {s['agree']:.0%}")
    print(f"page clearly supports the answer:{s['supported']:>5.0%}")
    print(f"solid (both):                    {s['solid']:.0%}")
    if s["to_review"]:
        print(f"\n{len(s['to_review'])} to review by hand:")
        for r in s["to_review"]:
            print(f"\n- p.{r['page']}  {r['stem']}")
            for j, o in enumerate(r["options"]):
                tag = ("  <- our key" if j == r["key"] else "") + ("  <- judge" if j == r["judge"] else "")
                print(f"    {LETTERS[j]}) {o}{tag}")
            print(f"    supported={r['supported']}  {r['reason']}")
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps({"summary": {k: v for k, v in s.items() if k != "to_review"},
                                          "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved", a.report)


if __name__ == "__main__":
    main()
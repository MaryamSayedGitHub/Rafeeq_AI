"""LLM-as-judge for the live suites.

The judge is Gemini (EVAL_JUDGE_MODEL, else GEMINI_MODEL, else gemini-2.5-flash), while the lesson / story generators
run on LLM_PROVIDER (Groq by default): a different model family grades the work, so it is not marking its own homework.
Every call is traced (LangSmith run type "chain", Langfuse observation type "evaluator") under the case being scored.
Temperature 0, JSON answers. A judge failure gives None (no score), never a fake number.
"""
from __future__ import annotations

import os
import time

from rafeeq.observability import traced


def model() -> str:
    return os.getenv("EVAL_JUDGE_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"


def available() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY", "").strip())


@traced("eval.judge", "evaluator")
def ask(prompt: str) -> dict | None:
    """Gemini JSON answer, with two waits on 'too many requests'. None when it cannot answer."""
    from rafeeq.teacher import montage

    for wait in (0, 20, 60):
        if wait:
            time.sleep(wait)
        try:
            out = montage._gemini_json([{"text": prompt}], model())
            return out if isinstance(out, dict) else {"items": out}
        except montage.MediaError as e:
            if "too many requests" not in str(e):
                return None
        except Exception:                                                # noqa: BLE001
            return None
    return None


def grade(criteria: str, material: str) -> float | None:
    """A 1-5 grade against `criteria`, returned on 0-1 ((g-1)/4)."""
    out = ask("You are a strict examiner of teaching material. Grade the material below from 1 (unusable) to 5 "
              "(excellent, nothing to fix) on these criteria only:\n" + criteria +
              "\nA factual error caps the grade at 2.\n\nMATERIAL:\n" + material[:30000] +
              '\n\nAnswer with JSON only: {"grade": 1-5, "reason": "one sentence"}')
    try:
        g = float(out["grade"])
    except (TypeError, KeyError, ValueError):
        return None
    return round((min(5.0, max(1.0, g)) - 1) / 4, 4)


def solve_quiz(questions: list[dict]) -> list[int | None]:
    """The judge answers the quiz WITHOUT the key; agreement with the key = the key is right."""
    lines = []
    for i, q in enumerate(questions):
        opts = "\n".join(f"  {j}. {o}" for j, o in enumerate(q.get("options", [])))
        lines.append(f"Q{i}: {q.get('question', '')}\n{opts}")
    out = ask("Answer each multiple-choice question. Give the index (0-3) of the single best option.\n\n"
              + "\n\n".join(lines) + '\n\nAnswer with JSON only: {"answers": [index for Q0, index for Q1, ...]}')
    answers = (out or {}).get("answers")
    if not isinstance(answers, list):
        return [None] * len(questions)
    return [(int(a) if isinstance(a, (int, float)) else None) for a in answers[:len(questions)]] + \
        [None] * max(0, len(questions) - len(answers))


def relevant(question: str, passages: list[str]) -> list[bool | None]:
    """For each passage: does it actually answer / teach the question?"""
    if not passages:
        return []
    body = "\n\n".join(f"P{i}: {p[:1500]}" for i, p in enumerate(passages))
    out = ask(f"A student asked: {question}\nFor each passage (a part of a video transcript), say whether watching it "
              "would answer the question (true) or not (false).\n\n" + body +
              '\n\nAnswer with JSON only: {"relevant": [true/false for P0, P1, ...]}')
    vals = (out or {}).get("relevant")
    if not isinstance(vals, list):
        return [None] * len(passages)
    return [(bool(v) if isinstance(v, bool) else None) for v in vals[:len(passages)]] + [None] * max(0, len(passages) - len(vals))

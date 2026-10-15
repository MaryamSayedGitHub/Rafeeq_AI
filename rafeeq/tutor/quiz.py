"""3 multiple-choice questions after a tutor answer or on a study file.

Source, in this order:  M4's POST /quiz (only when QUIZ_URL is set)  ->  data/quiz.json (local bank)  ->  the AI model writes
them from the course passages. M4 contract: {"room_id", "topic_id", "count"} -> {"quiz_id", "topic_id", "questions":
[{"text", "choices", "correct_answer", "explanation"}]}; the answer may be the option's text, an index or a letter.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import requests

from . import llm

N_QUESTIONS = 3
BANK = Path(__file__).resolve().parents[2] / "data" / "quiz.json"


def _answer_index(raw, options: list[str]) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw < len(options) else None
    raw = str(raw).strip()
    if len(raw) == 1 and raw.upper() in "ABCDEF":
        i = "ABCDEF".index(raw.upper())
        return i if i < len(options) else None
    if raw.isdigit():
        return int(raw) if int(raw) < len(options) else None
    low = [o.strip().lower() for o in options]
    return low.index(raw.lower()) if raw.lower() in low else None


def _as_mcq(item) -> dict | None:
    """Any reasonable MCQ shape -> {"question", "options", "answer": index, "explanation"}; None when unusable."""
    if not isinstance(item, dict):
        return None
    q = item.get("question") or item.get("q") or item.get("text")
    opts = item.get("options") or item.get("choices") or item.get("answers")
    if isinstance(opts, dict):
        opts = list(opts.values())
    if not q or not isinstance(opts, list) or len(opts) < 2:
        return None
    opts = [str(o.get("text") if isinstance(o, dict) else o).strip() for o in opts]
    idx = _answer_index(next((item[k] for k in ("answer", "correct", "correct_answer", "correct_option") if k in item), None), opts)
    if idx is None:
        return None
    return {"question": str(q).strip(), "options": opts, "answer": idx, "explanation": str(item.get("explanation") or "").strip()}


def _clean(items, n: int) -> list[dict]:
    out, seen = [], set()
    for it in items or []:
        m = _as_mcq(it)
        if m and m["question"].lower() not in seen:
            seen.add(m["question"].lower())
            out.append(m)
    return out[:n]


def normalize(items, n: int) -> list[dict]:
    """Any reasonable MCQ shape (M4's text / choices / correct_answer included) -> [{question, options, answer: index, explanation}]."""
    return _clean(items, n)


def _from_service(room_id: str | None, topic: str | None, n: int) -> list[dict]:
    url = os.getenv("QUIZ_URL")
    if not url or not room_id or not topic:
        return []
    try:
        r = requests.post(url, json={"room_id": room_id, "topic_id": topic, "count": n}, timeout=20)
        r.raise_for_status()
        d = r.json()
        items = next((d[k] for k in ("questions", "quiz", "results") if isinstance(d, dict) and isinstance(d.get(k), list)), d)
        return _clean(items if isinstance(items, list) else [], n)
    except Exception:                                          # noqa: BLE001  (service down -> next source)
        return []


def _from_bank(topic: str | None, n: int) -> list[dict]:
    if not BANK.exists():
        return []
    try:
        items = json.loads(BANK.read_text(encoding="utf-8"))
    except ValueError:
        return []
    same = [i for i in items if isinstance(i, dict) and topic and i.get("topic") == topic]
    pool = _clean(same, 1000)
    random.shuffle(pool)
    return pool[:n]


def _from_model(question: str, passages: list[dict], n: int, language: str | None = None) -> list[dict]:
    if not passages:
        return []
    ex = "\n".join(f"[{i}] {p['text'][:600]}" for i, p in enumerate(passages, 1))
    lang_rule = {"ar": "write everything in Arabic (technical terms may stay in English)", "en": "write everything in English"}.get(language or "", "same language as the student's question")
    res = llm.ask_json(f"""Write {n} multiple-choice questions (MCQ) that check understanding of this topic, using ONLY the excerpts.
Student's question: {question}

Excerpts:
{ex}

Return JSON only: {{"quiz": [{{"question": "...", "options": ["...", "...", "...", "..."], "answer": 0, "explanation": "one short sentence"}}]}}
Rules: exactly 4 options each, only one correct, "answer" = index (0-3) of the correct option, wrong options plausible but clearly wrong,
simple words, {lang_rule}, the {n} questions must be different.""", max_tokens=2000)
    return _clean(res.get("quiz") if isinstance(res, dict) else None, n)


def get_mcqs(question: str, topic: str | None, passages: list[dict], n: int = N_QUESTIONS, room_id: str | None = None, language: str | None = None) -> dict:
    """-> {"questions": [...], "source": "m4" | "bank" | "model" | "none"}. topic = the topic_id."""
    for source, fn in (("m4", lambda: _from_service(room_id, topic, n)),
                       ("bank", lambda: _from_bank(topic, n)),
                       ("model", lambda: _from_model(question, passages, n, language))):
        qs = fn()
        if qs:
            return {"questions": qs, "source": source}
    return {"questions": [], "source": "none"}


def passages_from_markdown(md: str) -> list[dict]:
    """A study file's sections (not Sources / Worked example) as passages for the quiz."""
    out, title, buf = [], "", []
    for line in md.splitlines() + ["## "]:
        if line.startswith("## "):
            if title and title not in ("Sources", "Worked example") and "".join(buf).strip():
                out.append({"title": title, "text": "\n".join(buf).strip()})
            title, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    return out[:6]


def quiz_for_study_file(sf: dict, n: int = N_QUESTIONS, language: str | None = None) -> dict:
    return get_mcqs(sf.get("question") or sf.get("title") or "", None, passages_from_markdown(sf["markdown"]), n, language=language)

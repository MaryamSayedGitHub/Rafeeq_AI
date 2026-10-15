"""Study helpers built on the model: explain an answer another way, summarise a study file, make flashcards.

Everything is written in the site language (`lang`: "ar" | "en") and uses only the text it is given (the answer, or the study file).
Each function returns None when the AI model is not reachable, so the page can say so.
"""
from __future__ import annotations

from . import llm
from .quiz import passages_from_markdown
from .study_store import get_json, put_json

LANG_NAME = {"ar": "Arabic", "en": "English"}
MODES = ("simpler", "example", "compare")
_MODE_TEXT = {
    "simpler": "Explain it again in much simpler words, as if to a 12-year-old. Short sentences. Max 80 words.",
    "example": "Give ONE concrete everyday example that shows the idea, then one sentence saying how the example matches the idea. Max 90 words.",
    "compare": "Compare it with: {other}. Give a short list of 3 to 5 differences or similarities (one line each) and one sentence that sums up. "
               "If you do not know {other}, say so in one sentence instead of guessing.",
}


def rewrite(mode: str, question: str, answer: str, lang: str, other: str = "") -> str | None:
    """Explain `answer` (to `question`) again: simpler, with an example, or compared with `other`."""
    if mode not in MODES or not answer.strip():
        return None
    res = llm.ask_json(f"""A student asked: {question}
The tutor answered (from the course material): {answer}

{_MODE_TEXT[mode].format(other=other.strip() or "the other idea")}
Write in {LANG_NAME[lang]}. Use only the answer above and common knowledge; do not invent facts about the course.
Return JSON only: {{"text": "..."}}""", max_tokens=600)
    text = str(res.get("text", "")).strip() if isinstance(res, dict) else ""
    return text or None


def summary(sf: dict, lang: str) -> list[str] | None:
    """5 to 7 short bullet points of a study file (kept per file and language)."""
    cached = get_json("summaries", f"{sf['id']}-{lang}")
    if cached:
        return cached
    body = "\n\n".join(f"{p['title']}\n{p['text']}" for p in passages_from_markdown(sf["markdown"]))[:6000]
    res = llm.ask_json(f"""Summarise this study file for a student in 5 to 7 short bullet points. Write in {LANG_NAME[lang]}.
Keep only what the file says. Return JSON only: {{"points": ["...", "..."]}}

{body}""", max_tokens=900)
    points = [str(p).strip() for p in (res.get("points") or []) if str(p).strip()][:7] if isinstance(res, dict) else []
    if points:
        put_json("summaries", f"{sf['id']}-{lang}", points)
    return points or None


def flashcards(sf: dict, lang: str, n: int = 10) -> list[dict] | None:
    """[{front, back}]: a question or term on the front, the short answer on the back (kept per file and language)."""
    cached = get_json("flashcards", f"{sf['id']}-{lang}")
    if cached:
        return cached
    body = "\n\n".join(f"{p['title']}\n{p['text']}" for p in passages_from_markdown(sf["markdown"]))[:6000]
    res = llm.ask_json(f"""Make {n} flashcards from this study file. Front = a short question or a term. Back = the short answer (max 25 words).
Write in {LANG_NAME[lang]}. Only facts the file states. Return JSON only: {{"cards": [{{"front": "...", "back": "..."}}]}}

{body}""", max_tokens=1800)
    cards = []
    for c in (res.get("cards") or []) if isinstance(res, dict) else []:
        if isinstance(c, dict) and str(c.get("front", "")).strip() and str(c.get("back", "")).strip():
            cards.append({"front": str(c["front"]).strip(), "back": str(c["back"]).strip()})
    cards = cards[:n]
    if cards:
        put_json("flashcards", f"{sf['id']}-{lang}", cards)
    return cards or None

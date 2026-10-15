"""Decides which capability handles a student message.

Order of decision (first one that answers wins):
  1. the student's explicit choice in the UI ("explain this" / "check my solution")
  2. a clear keyword (quiz, check my work, plan...) in Arabic or English: free, instant, predictable
  3. the LLM, for everything the keywords can't decide
  4. the keyword rules again, if the LLM is not configured or does not answer
"""
from __future__ import annotations

import logging

from rafeeq.copilot.llm import chat_json

log = logging.getLogger("rafeeq.copilot")

ROUTES = ["ask", "quiz", "check_work", "plan", "clarify"]
MODES = ["auto", "ask", "check_work"]

SYSTEM = (
    "You route a student's message in a study app. Reply with JSON {\"route\": X} where X is one of: "
    "ask, quiz, check_work, plan, clarify.\n"
    "ask = a question about the course material, or a request to explain something. Most messages are ask.\n"
    "quiz = wants to be tested or wants practice questions.\n"
    "check_work = wants their attached photo of solved work checked or corrected.\n"
    "plan = asks about deadlines, a schedule or what to study next.\n"
    "clarify = ONLY when the message has no clear request at all (e.g. 'hmm', 'ok', a single unclear word).\n"
    "The message may be in Arabic (including Egyptian dialect) or English. Examples:\n"
    "\"اشرحلي يعني إيه overfitting\" -> ask\n"
    "\"ليه الإجابة مش ٣/٤؟\" -> ask\n"
    "\"اختبرني في الفصل التاني\" -> quiz\n"
    "\"عايز أسئلة أتدرب عليها\" -> quiz\n"
    "\"صحّحلي الحل ده\" (photo attached) -> check_work\n"
    "\"أذاكر إيه النهارده؟\" -> plan\n"
    "\"hmm\" -> clarify"
)

_CHECK = ("check", "correct", "my answer", "my solution", "صحح", "صحّح", "راجع حل", "حلي", "حلّي")
_QUIZ = ("quiz", "test me", "practice", "اختبار", "اختبرني", "أسئلة تدريب", "تمرين")
_PLAN = ("plan", "schedule", "deadline", "what next", "خطة", "خطّة", "جدول", "موعد", "أذاكر إيه")


def strong_rule(text: str, has_file: bool) -> str | None:
    """A route only when a clear keyword says so; None means 'let the LLM decide'."""
    t = (text or "").lower().strip()
    if has_file and any(w in t for w in _CHECK):
        return "check_work"
    if any(w in t for w in _QUIZ):
        return "quiz"
    if any(w in t for w in _PLAN):
        return "plan"
    return None


def rules(text: str, has_file: bool) -> str:
    t = (text or "").lower().strip()
    route = strong_rule(t, has_file)
    if route:
        return route
    if len(t.split()) < 2 and not has_file:
        return "clarify"
    return "ask"


def classify_with_source(text: str, has_file: bool, history: list, mode: str = "auto") -> tuple[str, str]:
    """Returns (route, source); source is mode, keyword, llm or rules."""
    if mode in ("ask", "check_work"):
        return mode, "mode"
    try:
        from rafeeq.tutor import lookup
        if not has_file and lookup.mentions_course_matter(text):    # "when is the midterm": not a study plan, only the teacher knows; the ask route files it in the teacher's inbox
            return "ask", "course"
    except Exception:
        log.exception("course-matter check failed")
    route = strong_rule(text, has_file)
    if route:
        return route, "keyword"
    last = " | ".join(m["text"] for m in history[-4:])
    out = chat_json(SYSTEM, f"Recent: {last}\nPhoto attached: {has_file}\nMessage: {text}")
    llm_route = str((out or {}).get("route", "")).strip().lower()
    if llm_route in ROUTES:
        return llm_route, "llm"
    return rules(text, has_file), "rules"


def classify(text: str, has_file: bool, history: list, mode: str = "auto") -> str:
    route, source = classify_with_source(text, has_file, history, mode)
    log.info("copilot route=%s source=%s", route, source)
    return route

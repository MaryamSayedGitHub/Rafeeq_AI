"""The teacher's inbox (M5): student questions the tutor could not answer from the material.

Flow:
1. The Copilot's tutor answers "not covered" -> clients "escalate" -> escalate() files the question here.
2. The teacher sees it in the room's Inbox page, can ask for an AI draft (draft()), edits, and answers.
3. The answer is shown to the student (their own answered questions on the study page), and it becomes a
   teacher-approved answer: the next time a student asks the same thing, the Copilot's "find_verified"
   hook returns it before even asking the tutor.

Storage: the teacher service store (same as class_report.py), namespace ("class", room_id, "inbox").
Nothing here raises into the Copilot: a failure only means the question was not filed.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from rafeeq.teacher import class_report as cr

log = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.6     # word overlap (Jaccard) for "the same question"
_WORD = re.compile(r"\w+", re.UNICODE)


def _ns(room_id: str) -> tuple:
    return ("class", room_id, "inbox")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if len(w) > 1}


def _similar(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _items(store: Any, room_id: str) -> list:
    return cr._all(store, _ns(room_id)) if store is not None else []


def _student_names(room_id: str) -> dict[str, str]:
    return cr._students(room_id)


# ---------------------------------------------------------------- Copilot hooks

def escalate(room_id: str, question: str, user_id: str | None, store: Any = None) -> dict | None:
    """File a question for the teacher. Returns {"question_id"}; the same open question from the same
    student is filed once."""
    store = store or cr.get_store()
    question = (question or "").strip()
    if store is None or not room_id or not question:
        return None
    try:
        for item in _items(store, room_id):
            v = item.value
            if v.get("status") == "open" and v.get("user_id") == user_id and _similar(v["text"], question) >= 0.9:
                return {"question_id": item.key}
        qid = f"q_{uuid.uuid4().hex[:10]}"
        store.put(_ns(room_id), qid, {"question_id": qid, "user_id": user_id, "text": question[:1000],
                                      "time": _now(), "status": "open", "answer": "", "answered_at": None})
        return {"question_id": qid}
    except Exception as e:
        log.warning("inbox: could not file a question: %s", e)
        return None


def find_verified(room_id: str, question: str, store: Any = None) -> dict | None:
    """A teacher's answer to the same question, or None."""
    store = store or cr.get_store()
    if store is None or len(_words(question)) < 2:
        return None
    best, best_score = None, 0.0
    for item in _items(store, room_id):
        v = item.value
        if v.get("status") != "answered" or not v.get("answer"):
            continue
        score = _similar(v["text"], question)
        if score > best_score:
            best, best_score = v, score
    if best is None or best_score < MATCH_THRESHOLD:
        return None
    return {"answer": best["answer"], "question": best["text"], "question_id": best["question_id"]}


# ---------------------------------------------------------------- teacher side

def list_questions(room_id: str, user_id: str | None = None, store: Any = None) -> list[dict]:
    """Newest first. With user_id: only that student's questions (the student's own view)."""
    store = store or cr.get_store()
    names = _student_names(room_id)
    out = []
    for item in _items(store, room_id):
        v = dict(item.value)
        if user_id is not None and v.get("user_id") != user_id:
            continue
        v["student_name"] = names.get(v.get("user_id") or "", "")
        out.append(v)
    out.sort(key=lambda v: v.get("time") or "", reverse=True)
    out.sort(key=lambda v: v.get("status") != "open")  # open questions first, newest first within each group
    return out


def answer(room_id: str, question_id: str, text: str, store: Any = None) -> dict | None:
    store = store or cr.get_store()
    text = (text or "").strip()
    if store is None or not text:
        return None
    item = next((i for i in _items(store, room_id) if i.key == question_id), None)
    if item is None:
        return None
    value = {**item.value, "status": "answered", "answer": text[:4000], "answered_at": _now()}
    store.put(_ns(room_id), question_id, value)
    return value


DRAFT_SYSTEM = (
    "You help a teacher reply to a student's question in a study app. Write a short, clear reply the "
    "teacher can send as is: warm, in the language of the student's question (Arabic, including Egyptian "
    "dialect, or English), 2 to 5 sentences. If course excerpts are given, base the reply on them and "
    "mention the page like (p. 12). If you are not sure, say what the student should check instead of "
    "guessing. Reply with JSON {\"reply\": \"...\"}."
)

ASSISTANT_SYSTEM = (
    "You are the teaching assistant of a teacher using a study app. The teacher asks for something "
    "(a message to the class, a summary of where students struggle, a reply). Write a draft the teacher "
    "will review before sending. Use only the class data given; never invent numbers or names. Write in "
    "the language of the teacher's request. Reply with JSON {\"reply\": \"...\"}."
)


def _propose():
    from rafeeq.knowledge.topics.room_topics import default_propose

    return default_propose()


def _excerpts(room_id: str, question: str) -> str:
    try:
        from rafeeq.knowledge import integration

        if not integration.enabled():
            return ""
        from rafeeq.knowledge.service import get_service

        chunks = get_service().search(room_id, question, 3)["chunks"]
        return "\n\n".join(f"(p. {c['page']}, {c['file_name']}) {c['text'][:600]}" for c in chunks)
    except Exception:
        return ""


def draft(room_id: str, question: str, propose=None) -> str | None:
    propose = propose or _propose()
    if propose is None:
        return None
    excerpts = _excerpts(room_id, question)
    user = f"Student's question: {question}" + (f"\n\nCourse excerpts:\n{excerpts}" if excerpts else "")
    try:
        out = propose(DRAFT_SYSTEM, user) or {}
    except Exception:
        log.exception("inbox: draft failed")
        return None
    reply = str(out.get("reply", "")).strip()
    return reply or None


def assistant(room_id: str, request: str, store: Any = None, propose=None) -> str | None:
    """POST /teacher/assistant: a draft for the teacher's request, grounded in the room's class data."""
    store = store or cr.get_store()
    propose = propose or _propose()
    if propose is None:
        return None
    names = cr.room_topic_names(room_id)
    report = cr.build_report(store, room_id) if store is not None else {"topics": [], "students_at_risk": []}
    insights = cr.build_insights(store, room_id) if store is not None else {"top_questions": [], "common_mistakes": []}
    lines = [f"- {names.get(t['topic_id'], t['topic_id'])}: {round(t['mastery'] * 100)}% ({t['status']})" for t in report["topics"]]
    risk = [f"- {s['name']}: " + ", ".join(names.get(t, t) for t in s["topic_ids"]) for s in report["students_at_risk"]]
    asked = [f"- {names.get(q['topic_id'], q['topic_id'])}: {q['count']} questions, e.g. \"{q['example']}\"" for q in insights["top_questions"]]
    mistakes = [f"- {m['description']}" for m in insights["common_mistakes"]]
    data = "\n".join(["Class mastery per topic:", *(lines or ["(no quizzes yet)"]),
                      "Students needing follow-up:", *(risk or ["(none)"]),
                      "What students ask about:", *(asked or ["(no questions yet)"]),
                      "Repeated mistakes:", *(mistakes or ["(none)"])])
    try:
        out = propose(ASSISTANT_SYSTEM, f"Teacher's request: {request}\n\n{data}") or {}
    except Exception:
        log.exception("inbox: assistant draft failed")
        return None
    reply = str(out.get("reply", "")).strip()
    return reply or None


def connect() -> None:
    """Plug the inbox into the Copilot (rafeeq/copilot/clients.py)."""
    from rafeeq.copilot import clients

    clients.connect("escalate", lambda room_id, question, user_id: escalate(room_id, question, user_id))
    clients.connect("find_verified", lambda room_id, question: find_verified(room_id, question))

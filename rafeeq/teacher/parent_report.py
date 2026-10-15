"""Weekly parent report for one student (M5), from what the student actually did in the room.

Inputs are the records class_report.py already keeps (fed by M4's quiz submissions and the Copilot):
    ("class", room_id, "quiz")       {user_id, topic_id, score, total, wrong, time}
    ("class", room_id, "questions")  {user_id, text, time, topic_id}
Topic names come from the room's own topics (M2) when available.

The text is written by fixed Arabic templates, not by an LLM: a parent report must never contain
an invented claim, and it must work when no model is reachable.

Sharing: the teacher gets a signed link (share_token) that opens the report without an account and
expires after SHARE_DAYS. The token only names the room and the student; it gives no other access.
"""
from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from rafeeq.teacher import class_report as cr

log = logging.getLogger(__name__)

PERIOD_DAYS = 7
SHARE_DAYS = 14
_SALT = "rafeeq-parent-report"
_fallback_secret: str | None = None


def _secret() -> str:
    global _fallback_secret
    secret = os.environ.get("RAFEEQ_WEB_SECRET", "").strip()
    if secret:
        return secret
    if _fallback_secret is None:
        log.warning("parent_report: RAFEEQ_WEB_SECRET is not set; shared links stop working after a restart")
        _fallback_secret = secrets.token_urlsafe(32)
    return _fallback_secret


def make_share_token(room_id: str, user_id: str) -> str:
    from itsdangerous import URLSafeTimedSerializer

    return URLSafeTimedSerializer(_secret(), salt=_SALT).dumps({"r": room_id, "u": user_id})


def read_share_token(token: str) -> tuple[str, str] | None:
    """(room_id, user_id), or None when the token is invalid or older than SHARE_DAYS."""
    from itsdangerous import BadSignature, URLSafeTimedSerializer

    try:
        data = URLSafeTimedSerializer(_secret(), salt=_SALT).loads(token, max_age=SHARE_DAYS * 86400)
        return str(data["r"]), str(data["u"])
    except (BadSignature, KeyError, TypeError):
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _room_and_student(room_id: str, user_id: str) -> tuple[str, str]:
    try:
        from rafeeq.platform.service import get_store

        db = get_store().db
        return db["rooms"].get(room_id, {}).get("name", ""), db["users"].get(user_id, {}).get("name", "")
    except Exception:
        return "", ""


def _status(m: float) -> str:
    return "on_track" if m >= cr.ON_TRACK else "watch" if m >= cr.WATCH else "struggling"


def build(store: Any, room_id: str, user_id: str, days: int = PERIOD_DAYS) -> dict:
    """ParentReport (contracts/openapi.yaml)."""
    end = _now()
    start = end - timedelta(days=days)
    cutoff = start.isoformat()
    room_name, student_name = _room_and_student(room_id, user_id)
    names = cr.room_topic_names(room_id)

    quizzes = [i for i in cr._all(store, cr._ns(room_id, "quiz"))
               if i.value.get("user_id") == user_id] if store is not None else []
    questions = [i for i in cr._all(store, cr._ns(room_id, "questions"))
                 if i.value.get("user_id") == user_id] if store is not None else []
    week_quizzes = [q for q in quizzes if (q.value.get("time") or "") >= cutoff]
    week_questions = [q for q in questions if (q.value.get("time") or "") >= cutoff]

    before = cr._mastery_by_student_topic([q for q in quizzes if (q.value.get("time") or "") < cutoff])
    now = cr._mastery_by_student_topic(quizzes)
    week_topics = {q.value["topic_id"] for q in week_quizzes}

    topics = []
    for (uid, topic_id), mastery in now.items():
        if uid != user_id or topic_id not in week_topics:
            continue
        prev = before.get((uid, topic_id))
        topics.append({"topic_id": topic_id, "name": names.get(topic_id, topic_id),
                       "mastery": round(mastery, 2), "status": _status(mastery),
                       "change": None if prev is None else round(mastery - prev, 2)})
    topics.sort(key=lambda t: t["mastery"])

    answered = sum(q.value["total"] for q in week_quizzes)
    right = sum(q.value["score"] for q in week_quizzes)
    asked_topics: dict[str, int] = {}
    for q in week_questions:
        tid = q.value.get("topic_id") or cr.GENERAL
        asked_topics[tid] = asked_topics.get(tid, 0) + 1

    strengths = [t["name"] for t in topics if t["status"] == "on_track"]
    needs_work = [t["name"] for t in topics if t["status"] == "struggling"]
    improved = [t["name"] for t in topics if (t["change"] or 0) >= 0.1]

    return {
        "room_id": room_id, "room_name": room_name, "user_id": user_id, "student_name": student_name,
        "period": {"from": start.date().isoformat(), "to": end.date().isoformat()},
        "quizzes_taken": len(week_quizzes),
        "questions_answered": answered,
        "average_score": round(right / answered, 2) if answered else None,
        "questions_asked": len(week_questions),
        "asked_about": [names.get(t, t) for t, _ in sorted(asked_topics.items(), key=lambda kv: -kv[1])][:5],
        "topics": topics,
        "strengths": strengths, "needs_work": needs_work, "improved": improved,
        "summary": _summary(student_name, len(week_quizzes), answered, right, len(week_questions),
                            strengths, needs_work, improved),
    }


def _join(items: list[str]) -> str:
    return "، ".join(f"«{i}»" for i in items)


def _summary(name: str, quizzes: int, answered: int, right: int, asked: int,
             strengths: list[str], needs_work: list[str], improved: list[str]) -> list[str]:
    who = name or "الطالب"
    if not quizzes and not asked:
        return [f"لم يسجّل {who} نشاطًا على رفيق هذا الأسبوع. نقترح تشجيعه على حل اختبار قصير في مادته."]
    lines = []
    if quizzes:
        pct = round(100 * right / answered) if answered else 0
        lines.append(f"حلّ {who} {quizzes} {'اختبارًا' if quizzes > 2 else 'اختبار' if quizzes == 1 else 'اختبارين'} "
                     f"هذا الأسبوع، وأجاب صحيحًا عن {right} من {answered} سؤالًا ({pct}٪).")
    if asked:
        lines.append(f"طرح {asked} {'سؤالًا' if asked > 10 else 'أسئلة' if asked > 2 else 'سؤال' if asked == 1 else 'سؤالين'} "
                     "على رفيق أثناء المذاكرة، وهذا مؤشر جيد على الاهتمام.")
    if improved:
        lines.append(f"تحسّن مستواه في {_join(improved)}.")
    if strengths:
        lines.append(f"يُظهر فهمًا جيدًا في {_join(strengths)}.")
    if needs_work:
        lines.append(f"يحتاج إلى مراجعة إضافية في {_join(needs_work)}، وقد اقترح عليه رفيق تدريبات في هذه الموضوعات.")
    return lines

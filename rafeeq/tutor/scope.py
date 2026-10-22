"""What the tutor may answer when the room's material has nothing: the teacher's choice, and the course's subject.

The room's teacher picks one of two settings (rafeeq/platform/service.py, room "web_answers"):
  material_then_web  (default)  a question in the course's subject that the files miss is answered from the web, marked
                                "not from the teacher's material", and kept in the teacher's inbox for review.
  material_only                 the tutor never searches the web in this room: such a question goes to the teacher's inbox.

In both settings a question outside the course's subject ("what is the capital of Japan" in a physics room) is not
searched and not sent to the teacher: the student is asked, politely, to keep to the course.
"""
from __future__ import annotations

import logging

from . import llm

log = logging.getLogger(__name__)

MATERIAL_THEN_WEB = "material_then_web"
MATERIAL_ONLY = "material_only"
MODES = (MATERIAL_THEN_WEB, MATERIAL_ONLY)

# Student-facing sentences (the web page shows them as they are, see rafeeq/web/common.student_reason).
WEB_OFF = {
    "ar": "البحث في الويب غير مفعّل في هذه الغرفة، فالإجابة من مادة الغرفة فقط.",
    "en": "Web search is turned off in this room, so answers come from the room's material only.",
}


def _room(room_id: str) -> dict | None:
    try:
        from rafeeq.platform import service

        return service.get_store().db["rooms"].get(room_id)
    except Exception:  # noqa: BLE001  (no platform store: a test or a script)
        return None


def web_mode(room_id: str) -> str:
    room = _room(room_id) or {}
    mode = room.get("web_answers")
    return mode if mode in MODES else MATERIAL_THEN_WEB


def web_allowed(room_id: str) -> bool:
    return web_mode(room_id) != MATERIAL_ONLY


def off_topic_reply(room_id: str, lang: str) -> str:
    name = (_room(room_id) or {}).get("name") or ""
    if lang == "ar":
        where = f"مادة «{name}»" if name else "هذه المادة"
        return f"سؤالك خارج {where}. أنا هنا لأساعدك في المادة: اسألني عن أي جزء من دروسها."
    where = f"the course “{name}”" if name else "this course"
    return f"Your question is outside {where}. I am here to help with the course: ask me about any part of its lessons."


def in_subject(room_id: str, question: str) -> bool:
    """Is the question in the course's subject (so a teacher of this course would answer it)? One small model call, made only
    when the material had nothing and the web is about to be searched. No model, or an unclear reply: True (as before)."""
    room = _room(room_id) or {}
    try:
        from . import m2

        topics, _ = m2.topics(room_id)
    except Exception:  # noqa: BLE001
        topics = []
    names = [t.get("name", "") for t in topics if t.get("name")][:14]
    if not room.get("name") and not names:
        return True                                  # nothing to compare with
    res = llm.ask_json(f"""A study app tutors students of ONE course.
Course name: {room.get("name") or "(unknown)"}
Course topics: {", ".join(names) or "(unknown)"}
A student of this course asked: {question[:400]}

Is this question in the SUBJECT AREA of this course, so that this course's teacher would answer it (it may be a related idea that the
course files do not cover)? General knowledge from another field (geography, celebrities, sports, another school subject), chit-chat
or personal requests are NOT.
Return JSON only: {{"in_subject": true}} or {{"in_subject": false}}""", max_tokens=40)
    if isinstance(res, dict) and res.get("in_subject") is False:
        log.info("tutor: question outside the course of room %s: %r", room_id, question[:80])
        return False
    return True

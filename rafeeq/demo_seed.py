"""Demo data for one room, so the teacher's views and the parent reports are not empty in a demo.

    RAFEEQ_DEMO_ROOM=<room_id>   in .env, then start the app once. Safe to leave set: a room is seeded once.

Before: create the room as a teacher and upload a PDF, and wait until it is "ready", so the room has
its own topics (M2). The seed then adds five demo students (password: rafeeq-demo) who joined the room,
two weeks of quiz results per topic with different strengths, some questions to Rafeeq, and two
questions waiting in the teacher's inbox.

Everything goes through the same storage the real features use:
    M1 platform store  (accounts, membership, questions this week)
    M4 learner profile (quiz and question events -> mastery, weak topics, "what next")
    M5 class store     (class report, insights, parent report, inbox)
The quiz rows carry no per-question mistakes, so "repeated mistakes" only shows real ones: take a quiz
during the demo and they appear.

This is demo data: never set RAFEEQ_DEMO_ROOM on a real class.
"""
from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

STUDENTS = [  # name, email, how strong in each topic on average (cycled over the room's topics)
    ("عمر خالد", "omar@rafeeq.demo", [0.35, 0.45, 0.8]),
    ("سارة محمود", "sara.m@rafeeq.demo", [0.9, 0.85, 0.75]),
    ("يوسف أحمد", "youssef@rafeeq.demo", [0.55, 0.3, 0.6]),
    ("مريم علي", "mariam@rafeeq.demo", [0.8, 0.7, 0.9]),
    ("كريم حسن", "karim@rafeeq.demo", [0.4, 0.65, 0.5]),
]
QUESTION_TEMPLATES = [
    "مش فاهم «{name}»، ممكن تشرحها ببساطة؟",
    "إيه أهم نقطة في «{name}»؟",
    "ممكن مثال على «{name}»؟",
]
INBOX_TEMPLATES = [
    "هل «{name}» هتيجي في الامتحان؟",
    "ممكن مرجع إضافي عن «{name}»؟",
]


def _room_topics(room_id: str) -> list[tuple[str, str]]:
    from rafeeq.knowledge import integration

    if not integration.enabled():
        return []
    from rafeeq.knowledge.service import get_service

    return [(t["topic_id"], t["name"]) for t in get_service().list_topics(room_id)["topics"]
            if t["topic_id"] != "topic_general"][:6]


def seed(room_id: str, class_store: Any, rng: random.Random | None = None) -> bool:
    """Returns True when the room was seeded now. Never raises into the app's startup."""
    try:
        return _seed(room_id, class_store, rng or random.Random(7))
    except Exception:
        log.exception("demo seed: failed for room %s", room_id)
        return False


def _seed(room_id: str, class_store: Any, rng: random.Random) -> bool:
    from sqlalchemy.orm import Session

    from rafeeq.platform import service as platform
    from rafeeq.quiz.contracts import Event
    from rafeeq.quiz.profile import service as profile_service
    from rafeeq.quiz.profile.db import get_engine
    from rafeeq.teacher import class_report, inbox

    store = platform.get_store()
    room = store.db["rooms"].get(room_id)
    if room is None:
        log.warning("demo seed: room %s does not exist", room_id)
        return False
    if room.get("demo_seeded"):
        return False
    topics = _room_topics(room_id)
    if not topics:
        log.warning("demo seed: room %s has no topics yet; upload a PDF, wait until it is ready, restart", room_id)
        return False

    now = datetime.now(timezone.utc)
    with Session(get_engine()) as session:
        for s_index, (name, email, strengths) in enumerate(STUDENTS):
            try:
                user = store.register(name, email, platform.DEMO_PASSWORD, "student", room["join_code"])
            except platform.Conflict:  # the demo account exists already (another demo room)
                user = store.login(email, platform.DEMO_PASSWORD)
                if user["user_id"] not in store.db["members"].get(room_id, []):
                    store.join_room(store.user_for_token(user["token"]), room["join_code"])
            uid = user["user_id"]
            for t_index, (topic_id, topic_name) in enumerate(topics):
                level = strengths[t_index % len(strengths)]
                for k in range(rng.randint(2, 3)):                    # two weeks of quizzes, improving a bit
                    when = now - timedelta(days=13 - 6 * k, hours=rng.randint(0, 20))
                    frac = min(1.0, max(0.0, rng.gauss(level + 0.05 * k, 0.12)))
                    total = 5
                    right = round(frac * total)
                    profile_service.record_event(session, Event(
                        user_id=uid, room_id=room_id, topic_id=topic_id, kind="quiz",
                        score=right / total, meta={"demo": True}))
                    if class_store is not None:
                        class_store.put(class_report._ns(room_id, "quiz"), uuid.uuid4().hex, {
                            "user_id": uid, "topic_id": topic_id, "score": right, "total": total,
                            "wrong": [], "time": when.isoformat(), "demo": True})
                if level < 0.6 or rng.random() < 0.3:                 # weaker topics get questions
                    text = rng.choice(QUESTION_TEMPLATES).format(name=topic_name)
                    when = now - timedelta(days=rng.randint(0, 6), hours=rng.randint(0, 20))
                    class_report.record_question(room_id, uid, text, time=when.isoformat(),
                                                 store=class_store, topic_id=topic_id)
                    profile_service.record_event(session, Event(
                        user_id=uid, room_id=room_id, topic_id=topic_id, kind="question", score=0.0,
                        meta={"text": text, "demo": True}))
                    store.record_question(uid, room_id)
            if s_index < len(INBOX_TEMPLATES) and class_store is not None:
                inbox.escalate(room_id, INBOX_TEMPLATES[s_index].format(name=topics[s_index % len(topics)][1]),
                               uid, store=class_store)

    with store._lock:
        room["demo_seeded"] = True
        store._save()
    log.info("demo seed: room %s filled with %d demo students over %d topics", room_id, len(STUDENTS), len(topics))
    return True

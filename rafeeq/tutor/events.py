"""Tell M4 that the student asked a question (contracts: QuestionEvent -> POST /events, 204). Best effort: it never breaks the answer.

EVENTS_URL  default http://127.0.0.1:8000/events.
Who is asking: the X-User-Id header, or the demo login token (rafeeq/platform/router.py) until real login exists. No user -> no event.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests

log = logging.getLogger(__name__)
DEMO_TOKENS = {"demo-token-sara-123": "u_sara"}


def current_user_id(headers) -> str | None:
    if headers.get("x-user-id"):
        return headers["x-user-id"]
    token = (headers.get("authorization") or "").removeprefix("Bearer ").strip()
    if token in DEMO_TOKENS:
        return DEMO_TOKENS[token]
    try:                                                           # a real M1 login token
        from rafeeq.platform import service
        user = service.get_store().user_for_token(token) if token else None
        return user["user_id"] if user else None
    except Exception:                                              # noqa: BLE001
        return None


def post_event(event: dict) -> bool:
    """Tell M4 about a QuestionEvent (already built). In the same app the Copilot's hook reaches M4 directly (no port, no HTTP);
    otherwise POST to EVENTS_URL. Best effort."""
    try:
        from rafeeq.copilot import clients
        hook = clients._CONNECTED.get("record_event")
        if hook is not None and not getattr(hook, "is_m3", False):      # M4's own hook (in this app), never M3's which would call back here
            hook(event)
            return True
    except Exception as e:                                         # noqa: BLE001
        log.warning("M4 record_event hook failed: %s", type(e).__name__)
    try:
        requests.post(os.getenv("EVENTS_URL", "http://127.0.0.1:8000/events"), json=event, timeout=5).raise_for_status()
        return True
    except Exception as e:                                         # noqa: BLE001
        log.warning("M4 /events not reachable: %s", type(e).__name__)
        return False


def send_question_event(user_id: str | None, room_id: str, topic_id: str | None, text: str, kind: str = "question") -> bool:
    """kind "question" when the student asks, "understood" when they finish a solution or pass a quiz (the contract's proposal 4)."""
    if not user_id:
        return False
    return post_event({"user_id": user_id, "room_id": room_id, "topic_id": topic_id or "other", "kind": kind, "text": text,
                       "time": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")})


def _record(user_id: str | None, room_id: str, topic_id: str | None, kind: str, score: float, meta: dict) -> bool:
    """A RESULT for M4's profile (a quiz or a handwritten solution): it moves the topic's mastery, which is what the student's Progress page shows.
    (The /events route only takes questions, so results go straight to M4's profile service, in this same app.) Best effort."""
    if not user_id or not topic_id or topic_id == "other":
        return False
    try:
        from sqlalchemy.orm import Session
        from rafeeq.quiz.contracts import Event
        from rafeeq.quiz.profile import service as profile_service
        from rafeeq.quiz.profile.db import get_engine
        with Session(get_engine()) as session:
            profile_service.record_event(session, Event(user_id=user_id, room_id=room_id, topic_id=topic_id, kind=kind,
                                                        score=max(0.0, min(1.0, float(score))), meta=meta))
        return True
    except Exception as e:                                         # noqa: BLE001
        log.warning("M4 profile not reachable: %s", type(e).__name__)
        return False


def record_quiz(user_id: str | None, room_id: str, topic_id: str | None, correct: int, total: int) -> bool:
    """The student finished the quiz of a topic: correct/total moves the topic's mastery."""
    return total > 0 and _record(user_id, room_id, topic_id, "quiz", correct / total, {"correct": correct, "total": total, "source": "tutor-chat"})


def record_handwriting(user_id: str | None, room_id: str, topic_id: str | None, ok_steps: int, steps: int) -> bool:
    """A checked handwritten solution: the share of steps that were right."""
    return steps > 0 and _record(user_id, room_id, topic_id, "handwriting", ok_steps / steps, {"ok_steps": ok_steps, "steps": steps, "source": "tutor-snap"})

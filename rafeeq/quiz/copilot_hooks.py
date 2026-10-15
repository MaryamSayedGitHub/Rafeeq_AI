"""Connects M4 to the Copilot (rafeeq/copilot/clients.py), in-process.

    next_step(room_id, user_id)  -> NextStep | None   "ماذا أذاكر الآن؟" and the chat's quiz suggestion
    record_event(event_dict)     -> None              every grounded question counts towards the profile

Same logic as GET /next-step and POST /events, without HTTP. Called once from install() in router.py.
Nothing here raises into the Copilot: a problem means "not available" (None), never a crash.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy.orm import Session

from rafeeq.quiz import adapters, deps
from rafeeq.quiz.profile import service as profile_service
from rafeeq.quiz.profile.db import get_engine

log = logging.getLogger(__name__)


def _topic_names(room_id: str) -> dict[str, str]:
    chunks = deps.material(deps.room_chunks, room_id, deps.get_chunks())
    return {c["topic_id"]: (c.get("topic_name") or c["topic_id"]) for c in chunks}


def next_step(room_id: str, user_id: Optional[str]) -> Optional[dict]:
    if not user_id:
        return None  # no student, no profile
    try:
        names = _topic_names(room_id)
        with Session(get_engine()) as session:
            profile = profile_service.get_profile(session, user_id, room_id, names)
        topic = adapters.pick_next_topic(profile)
        if topic is not None:
            return adapters.next_step(topic)
        if names:  # a new student: start with the first topic of the room's material
            first = next(iter(names))
            return adapters.first_topic_step(first, names[first])
        return None
    except Exception:
        log.exception("quiz: next_step for the Copilot failed")
        return None


def record_event(event: dict[str, Any]) -> None:
    try:
        parsed = adapters.question_event_to_event(adapters.QuestionEvent(**event))
        if parsed is not None:
            with Session(get_engine()) as session:
                profile_service.record_event(session, parsed)
    except Exception:
        log.exception("quiz: recording a question event failed")


def connect() -> None:
    from rafeeq.copilot import clients

    clients.connect("next_step", next_step)
    clients.connect("record_event", record_event)

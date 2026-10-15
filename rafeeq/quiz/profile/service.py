"""record_event and get_profile: the real learner-profile logic."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from rafeeq.quiz.contracts import Event, Profile, TopicProfile
from rafeeq.quiz.profile.mastery import INITIAL_MASTERY, TopicState, apply_event, is_weak
from rafeeq.quiz.profile.models import EventRow, TopicMasteryRow


def _state_of(row: TopicMasteryRow) -> TopicState:
    return TopicState(
        mastery=row.mastery,
        attempts=row.attempts,
        questions_asked=row.questions_asked,
        questions_since_quiz=row.questions_since_quiz,
    )


def record_event(session: Session, event: Event) -> TopicState:
    """Store the event and update the topic's mastery. Returns the new state."""
    row = session.scalar(
        select(TopicMasteryRow).where(
            TopicMasteryRow.user_id == event.user_id,
            TopicMasteryRow.room_id == event.room_id,
            TopicMasteryRow.topic_id == event.topic_id,
        )
    )
    if row is None:
        row = TopicMasteryRow(
            user_id=event.user_id,
            room_id=event.room_id,
            topic_id=event.topic_id,
            mastery=INITIAL_MASTERY,
            attempts=0,
            questions_asked=0,
            questions_since_quiz=0,
        )
        session.add(row)

    new = apply_event(_state_of(row), event.kind, event.score)
    row.mastery = new.mastery
    row.attempts = new.attempts
    row.questions_asked = new.questions_asked
    row.questions_since_quiz = new.questions_since_quiz

    session.add(
        EventRow(
            user_id=event.user_id,
            room_id=event.room_id,
            topic_id=event.topic_id,
            kind=event.kind,
            score=event.score,
            meta=event.meta,
        )
    )
    session.commit()
    return new


def get_profile(
    session: Session,
    user_id: str,
    room_id: str,
    topic_names: Optional[dict[str, str]] = None,
) -> Profile:
    """The student's topics. topic_names comes from M2's list_topics later;
    until then the topic_id is used as the name."""
    names = topic_names or {}
    rows = session.scalars(
        select(TopicMasteryRow)
        .where(TopicMasteryRow.user_id == user_id, TopicMasteryRow.room_id == room_id)
        .order_by(TopicMasteryRow.topic_id)
    ).all()
    return Profile(
        topics=[
            TopicProfile(
                topic_id=r.topic_id,
                name=names.get(r.topic_id, r.topic_id),
                mastery=r.mastery,
                attempts=r.attempts,
                questions_asked=r.questions_asked,
                weak=is_weak(_state_of(r)),
            )
            for r in rows
        ]
    )
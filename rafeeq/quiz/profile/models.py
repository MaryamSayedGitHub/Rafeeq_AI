"""M4's own tables. All names start with m4_ so we only ever touch our tables."""
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class EventRow(Base):
    """Every event we receive (question, quiz, handwriting, attendance)."""

    __tablename__ = "m4_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    room_id: Mapped[str] = mapped_column(String(64), index=True)
    topic_id: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))
    score: Mapped[float] = mapped_column(Float)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TopicMasteryRow(Base):
    """Current learner state: one row per (user, room, topic)."""

    __tablename__ = "m4_topic_mastery"
    __table_args__ = (UniqueConstraint("user_id", "room_id", "topic_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    room_id: Mapped[str] = mapped_column(String(64), index=True)
    topic_id: Mapped[str] = mapped_column(String(64))
    mastery: Mapped[float] = mapped_column(Float)
    attempts: Mapped[int] = mapped_column(Integer)
    questions_asked: Mapped[int] = mapped_column(Integer)
    questions_since_quiz: Mapped[int] = mapped_column(Integer)
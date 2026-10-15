"""Tables for quizzes and submitted answers (all names start with m4_)."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from rafeeq.quiz.profile.models import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class QuizRow(Base):
    __tablename__ = "m4_quizzes"

    quiz_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    room_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    mode: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class QuizItemRow(Base):
    """One question of a quiz. `data` holds the whole QuizItem as JSON."""

    __tablename__ = "m4_quiz_items"
    __table_args__ = (UniqueConstraint("quiz_id", "item_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    quiz_id: Mapped[str] = mapped_column(String(64), index=True)
    item_id: Mapped[str] = mapped_column(String(80))
    position: Mapped[int] = mapped_column(Integer)
    data: Mapped[dict] = mapped_column(JSON)


class AnswerRow(Base):
    """What a student chose for one item. chosen = -1 means skipped."""

    __tablename__ = "m4_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    quiz_id: Mapped[str] = mapped_column(String(64), index=True)
    item_id: Mapped[str] = mapped_column(String(80))
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    room_id: Mapped[str] = mapped_column(String(64), index=True)
    topic_id: Mapped[str] = mapped_column(String(64))
    chosen: Mapped[int] = mapped_column(Integer)
    correct: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
"""Tables for quizzes and submitted answers (all names start with m4_)."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint, false
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
    """What a student answered for one item. chosen = -1 means skipped (or a typed answer: see answer_text)."""

    __tablename__ = "m4_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    quiz_id: Mapped[str] = mapped_column(String(64), index=True)
    item_id: Mapped[str] = mapped_column(String(80))
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    room_id: Mapped[str] = mapped_column(String(64), index=True)
    topic_id: Mapped[str] = mapped_column(String(64))
    chosen: Mapped[int] = mapped_column(Integer)
    correct: Mapped[bool] = mapped_column(Boolean)
    # typed answers (short, math, problem, essay): what the student wrote, the share of the marks, the feedback.
    # score is NULL for rows saved before these columns existed: read them with answer_score().
    answer_text: Mapped[str] = mapped_column(Text, default="", server_default="")
    score: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    feedback: Mapped[str] = mapped_column(Text, default="", server_default="")
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


def answer_score(row: "AnswerRow") -> float:
    """The share of the marks, 0..1. Rows saved before typed questions existed only have `correct`."""
    return float(row.score) if row.score is not None else (1.0 if row.correct else 0.0)

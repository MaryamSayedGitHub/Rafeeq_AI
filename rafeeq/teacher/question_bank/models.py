
"""SQLAlchemy table for the teacher question bank (all names start with m6_).

Each row is one QuizItem owned by a teacher inside a room.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text, false
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class BankItemRow(Base):
    """One question in a teacher's question bank."""

    __tablename__ = "m6_question_bank"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ownership
    item_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    room_id: Mapped[str] = mapped_column(String(64), index=True)
    teacher_id: Mapped[str] = mapped_column(String(64), index=True)

    # question content
    kind: Mapped[str] = mapped_column(String(16))
    stem: Mapped[str] = mapped_column(Text)
    options: Mapped[list] = mapped_column(JSON, default=list)
    right_choice: Mapped[int] = mapped_column(Integer, default=0)
    answer: Mapped[str] = mapped_column(Text, default="", server_default="")
    rubric: Mapped[list] = mapped_column(JSON, default=list)
    rubric_weights: Mapped[list] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(Text, default="", server_default="")

    # teacher-facing grading guide
    grading: Mapped[dict] = mapped_column(JSON, default=dict)

    # difficulty
    difficulty: Mapped[str] = mapped_column(String(8), default="", server_default="")

    # source tracking
    file_name: Mapped[str] = mapped_column(String(256), default="", server_default="")
    page: Mapped[int] = mapped_column(Integer, default=0)
    topic_id: Mapped[str] = mapped_column(String(64), default="", server_default="")

    # teacher review state
    approved: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

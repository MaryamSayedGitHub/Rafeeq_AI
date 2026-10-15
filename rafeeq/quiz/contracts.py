"""M4 contracts: shared schemas for the learner profile, quizzes and paper grader.

Adding a field is fine. Renaming or removing one needs a yes from every consumer.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

EventKind = Literal["question", "quiz", "handwriting", "attendance"]
ItemSource = Literal["generated", "lecturer_approved"]
QuizMode = Literal["screen", "paper"]


# ---------- events and profile ----------
class Event(BaseModel):
    user_id: str
    room_id: str
    topic_id: str
    kind: EventKind
    score: float = Field(ge=0.0, le=1.0)
    meta: dict = Field(default_factory=dict)


class TopicProfile(BaseModel):
    topic_id: str
    name: str
    mastery: float
    attempts: int
    questions_asked: int
    weak: bool


class Profile(BaseModel):
    topics: list[TopicProfile]


# ---------- quiz items ----------
class QuizItem(BaseModel):
    item_id: str
    topic_id: str
    stem: str
    options: list[str]
    right_choice: int  # index into options, 0..3
    explanation: str
    file_name: str
    page: int
    source: ItemSource = "generated"

    @field_validator("options")
    @classmethod
    def _four_options(cls, v: list[str]) -> list[str]:
        if len(v) != 4:
            raise ValueError("a QuizItem needs exactly 4 options")
        return v

    @field_validator("right_choice")
    @classmethod
    def _choice_in_range(cls, v: int) -> int:
        if not 0 <= v <= 3:
            raise ValueError("right_choice must be 0..3")
        return v


class QuizCreateRequest(BaseModel):
    n: int = Field(default=5, ge=1, le=30)
    mode: QuizMode = "screen"
    topic_id: Optional[str] = None  # when set, every question comes from this topic


class QuizCreateResponse(BaseModel):
    quiz_id: str
    items: list[QuizItem]
    sheet_pdf: Optional[str] = None  # URL/path, only when mode == "paper"


# ---------- submit ----------
class Answer(BaseModel):
    item_id: str
    choice: int


class SubmitRequest(BaseModel):
    answers: list[Answer]


class ItemResult(BaseModel):
    item_id: str
    correct: bool
    right_choice: int
    explanation: str
    file_name: str
    page: int


class SubmitResponse(BaseModel):
    score: float
    items: list[ItemResult]


# ---------- paper grader ----------
class DetectedAnswer(BaseModel):
    item_id: str
    choice: Optional[int]  # None when no mark / two marks
    confidence: float


class GradeSheetResponse(BaseModel):
    answers: list[DetectedAnswer]
    score: float
    needs_review: list[str]


# ---------- instructor data ----------
class ClassProfileRow(BaseModel):
    user_id: str
    name: str
    topic_id: str
    mastery: float
    attempts: int


class WrongAnswerRow(BaseModel):
    """No names here: student_ref is an opaque reference."""

    item_id: str
    student_ref: str
    topic_id: str
    stem: str
    chosen: int
    right: int

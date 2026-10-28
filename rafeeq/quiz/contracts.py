"""M4 contracts: shared schemas for the learner profile, quizzes and paper grader.

Adding a field is fine. Renaming or removing one needs a yes from every consumer.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

EventKind = Literal["question", "quiz", "handwriting", "attendance"]
ItemSource = Literal["generated", "lecturer_approved"]
QuizMode = Literal["screen", "paper"]
# mcq: 4 options, one right | true_false: 2 options | short: a word or a phrase | math: an expression or an equation
# problem: a worked problem with a final answer | essay: a free-text answer graded against a rubric
ItemKind = Literal["mcq", "true_false", "short", "math", "problem", "essay"]
ALL_KINDS: tuple[str, ...] = ("mcq", "true_false", "short", "math", "problem", "essay")
CHOICE_KINDS = ("mcq", "true_false")  # answered by picking an option; every other kind is answered by typing


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
    options: list[str] = Field(default_factory=list)  # mcq: 4, true_false: 2, other kinds: empty
    right_choice: int = 0  # index into options (mcq 0..3, true_false 0..1); unused by the other kinds
    explanation: str
    file_name: str
    page: int
    source: ItemSource = "generated"
    kind: ItemKind = "mcq"  # old stored items have no kind: they are mcq
    answer: str = ""  # the model answer (short / math / problem / essay)
    rubric: list[str] = Field(default_factory=list)  # what a full-mark answer must contain (essay / problem)

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> "QuizItem":
        if self.kind == "mcq":
            if len(self.options) != 4:
                raise ValueError("a QuizItem needs exactly 4 options")
            if not 0 <= self.right_choice <= 3:
                raise ValueError("right_choice must be 0..3")
        elif self.kind == "true_false":
            if len(self.options) != 2:
                raise ValueError("a true_false item needs exactly 2 options")
            if not 0 <= self.right_choice <= 1:
                raise ValueError("right_choice must be 0 or 1")
        else:
            if self.options:
                raise ValueError(f"a {self.kind} item has no options")
            if not self.answer.strip():
                raise ValueError(f"a {self.kind} item needs a model answer")
        return self

    @property
    def is_choice(self) -> bool:
        return self.kind in CHOICE_KINDS


class QuizCreateRequest(BaseModel):
    n: int = Field(default=5, ge=1, le=30)
    mode: QuizMode = "screen"
    topic_id: Optional[str] = None  # when set, every question comes from this topic
    kinds: list[ItemKind] = Field(default_factory=lambda: ["mcq"])  # question types to mix; paper mode is mcq only


class QuizCreateResponse(BaseModel):
    quiz_id: str
    items: list[QuizItem]
    sheet_pdf: Optional[str] = None  # URL/path, only when mode == "paper"


# ---------- submit ----------
class Answer(BaseModel):
    item_id: str
    choice: int = -1  # mcq / true_false: the option index (-1 = skipped)
    text: str = ""  # every other kind: what the student typed


class SubmitRequest(BaseModel):
    answers: list[Answer]


class ItemResult(BaseModel):
    item_id: str
    correct: bool
    right_choice: int
    explanation: str
    file_name: str
    page: int
    score: float = 0.0  # 0..1; mcq / true_false are 0 or 1, the typed kinds can earn part of the marks
    feedback: str = ""  # typed kinds: what was right, what was missing
    needs_review: bool = False  # the grader was not sure (or was down): a teacher should look


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


"""Pydantic contracts for the M6 Question Bank API.

These are the shapes the HTTP layer sends and receives. The core question
data re-uses QuizItem from M4 so the same item can flow straight into a
student quiz without any conversion.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rafeeq.quiz.contracts import ItemKind, QuizItem


# ---------- generation request ----------

class GenerateRequest(BaseModel):
    """Body for POST /teacher/rooms/{room_id}/question-bank/generate."""

    model_config = ConfigDict(extra="forbid")

    # max 20: every question is 1-2 LLM calls inside one HTTP request
    n: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of questions to generate",
    )
    kinds: list[ItemKind] = Field(
        default_factory=lambda: ["mcq"],
        description="Question types to mix (mcq, true_false, short, math, problem, essay)",
    )
    topic_id: Optional[str] = Field(
        default=None,
        description="Restrict generation to one topic; None means all topics",
    )


Difficulty = Literal["easy", "medium", "hard"]


# ---------- grading guide ----------

class Grading(BaseModel):
    """What a correct, a partly correct and a wrong answer look like, in the teacher's words."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    correct: str = Field(min_length=1, max_length=600)
    partial: str = Field(min_length=1, max_length=600)
    wrong: str = Field(min_length=1, max_length=600)


# ---------- bank item (what the API returns) ----------

class BankItem(BaseModel):
    """A question bank item as returned by the API.

    The full QuizItem is embedded so the frontend can render it as a quiz
    card; teacher_id and approved live next to it because they only exist
    in the bank.
    """

    model_config = ConfigDict(extra="forbid")

    item: QuizItem
    teacher_id: str
    approved: bool = False
    grading: Optional[Grading] = None
    difficulty: Optional[Difficulty] = None


# ---------- approve / update ----------

class ApproveRequest(BaseModel):
    """Body for PATCH /teacher/rooms/{room_id}/question-bank/{item_id}/approve."""

    model_config = ConfigDict(extra="forbid")

    approved: bool


class UpdateRequest(BaseModel):
    """Body for PUT /teacher/rooms/{room_id}/question-bank/{item_id}."""

    model_config = ConfigDict(extra="forbid")

    stem: Optional[str] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None
    rubric: Optional[list[str]] = None
    rubric_weights: Optional[list[float]] = None
    options: Optional[list[str]] = None
    right_choice: Optional[int] = None
    grading: Optional[Grading] = None
    difficulty: Optional[Difficulty] = None
    approved: Optional[bool] = None
    @model_validator(mode="after")
    def validate_rubric_weights(self):
        if not self.rubric_weights:
            return self

        if self.rubric is not None and len(self.rubric_weights) != len(self.rubric):
            raise ValueError(
                "Number of rubric weights must match number of rubric criteria"
            )

        if any(
            not math.isfinite(weight) or not 0 <= weight <= 100
            for weight in self.rubric_weights
        ):
            raise ValueError(
                "Each rubric weight must be a finite number between 0 and 100"
            )

        if sum(self.rubric_weights) <= 0:
            raise ValueError(
                "At least one rubric weight must be greater than zero"
            )

        return self



# ---------- responses ----------

class BankItemList(BaseModel):
    items: list[BankItem]
    total: int


class GenerateResponse(BaseModel):
    generated: int
    items: list[BankItem]


class SearchIntent(BaseModel):
    """What the teacher asked for, as understood from a sentence."""

    model_config = ConfigDict(extra="forbid")

    n: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="How many questions; None = not said",
    )
    difficulty: Optional[Difficulty] = None
    difficulties: list[Difficulty] = Field(
        default_factory=list,
        description='A mix, e.g. "one hard and one easy": one question for each level, in order. Empty = not a mix',
    )
    kinds: list[ItemKind] = Field(
        default_factory=list,
        description="Empty = any type",
    )
    topic_query: str = Field(
        default="",
        description="The subject only; empty = any subject",
    )
    approved_only: bool = False


class SearchResponse(BaseModel):
    query: str
    items: list[BankItem]
    total: int
    understood: Optional[SearchIntent] = None
    requested: Optional[int] = None
    message: str = ""
    related_ids: list[str] = Field(
        default_factory=list,
        description="Items that are close to the subject but do not match it exactly",
    )
    missing_levels: list[str] = Field(
        default_factory=list,
        description="Difficulty levels of a mix for which the bank has no question",
    )
    transcript: Optional[str] = None


class BackfillResponse(BaseModel):
    updated: int

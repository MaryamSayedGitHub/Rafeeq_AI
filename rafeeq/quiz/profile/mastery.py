"""Pure mastery maths for the learner profile (no DB, easy to unit-test)."""
from __future__ import annotations

from dataclasses import dataclass

OLD_WEIGHT = 0.7
NEW_WEIGHT = 0.3
WEAK_THRESHOLD = 0.6
MIN_EVENTS_FOR_WEAK = 2
QUESTIONS_WITHOUT_QUIZ_FOR_WEAK = 3
INITIAL_MASTERY = 0.5


def update_mastery(old: float, score: float) -> float:
    """new = 0.7 * old + 0.3 * score"""
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1]")
    return OLD_WEIGHT * old + NEW_WEIGHT * score


@dataclass
class TopicState:
    mastery: float = INITIAL_MASTERY
    attempts: int = 0  # mastery-changing events (quiz, handwriting)
    questions_asked: int = 0
    questions_since_quiz: int = 0


def apply_event(state: TopicState, kind: str, score: float) -> TopicState:
    """Return the new state after one event.

    Questions only count as questions (they do not move mastery); quiz and
    handwriting events move mastery. Attendance is ignored here.
    """
    s = TopicState(**vars(state))
    if kind == "question":
        s.questions_asked += 1
        s.questions_since_quiz += 1
    elif kind in ("quiz", "handwriting"):
        s.mastery = update_mastery(s.mastery, score)
        s.attempts += 1
        if kind == "quiz":
            s.questions_since_quiz = 0
    return s


def is_weak(state: TopicState) -> bool:
    """Weak if mastery < 0.6 after 2 events, or 3+ questions without a quiz."""
    low_after_events = (
        state.attempts >= MIN_EVENTS_FOR_WEAK and state.mastery < WEAK_THRESHOLD
    )
    many_questions = state.questions_since_quiz >= QUESTIONS_WITHOUT_QUIZ_FOR_WEAK
    return low_after_events or many_questions

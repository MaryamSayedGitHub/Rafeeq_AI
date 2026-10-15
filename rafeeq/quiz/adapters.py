"""Translate between M4's own shapes and the team contract (contracts/openapi.yaml).

Everything here is a plain function: no database, no web framework. That keeps the
translation easy to test, and keeps M4's real logic (profile, quiz, grader) untouched.

Team contract                         M4 inside
-----------------------------------   ---------------------------------------------
Quiz.questions[].choices              QuizItem.options
Quiz.questions[].correct_answer       QuizItem.options[right_choice]  (text, not a number)
SubmitQuizRequest.answers (texts)     list[Answer(item_id, choice index)]
QuizResult.score (a count)            SubmitResponse.score (a share 0..1)
QuizResult.mastery_change             after minus before, read from the profile
StudentSummary.weak_topics[].score    TopicProfile.mastery
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

from rafeeq.quiz.contracts import (
    Answer,
    Event,
    Profile,
    QuizCreateResponse,
    QuizItem,
    SubmitResponse,
    TopicProfile,
)

INITIAL_MASTERY = 0.5  # same starting value as the profile

_ARABIC = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")


def _is_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))


# ---------- /quiz ----------
def to_contract_quiz(resp: QuizCreateResponse, requested_topic: Optional[str] = None) -> dict:
    """The Quiz shape. A quiz that mixes topics reports the topic of its first question."""
    items = resp.items
    topic_id = requested_topic or (items[0].topic_id if items else "")
    return {
        "quiz_id": resp.quiz_id,
        "topic_id": topic_id,
        "questions": [
            {
                "text": it.stem,
                "choices": list(it.options),
                "correct_answer": it.options[it.right_choice],
                "explanation": it.explanation,
            }
            for it in items
        ],
    }


# ---------- /quiz/{id}/submit ----------
def answers_to_choices(items: list[QuizItem], answers: list[str]) -> list[Answer]:
    """Turn the student's answer texts (one per question, in order) into choice indexes.
    A missing or unknown answer becomes -1, which M4 stores as 'skipped'."""
    out: list[Answer] = []
    for i, it in enumerate(items):
        text = answers[i].strip() if i < len(answers) else ""
        options = [o.strip() for o in it.options]
        choice = options.index(text) if text in options else -1
        out.append(Answer(item_id=it.item_id, choice=choice))
    return out


def _explanation(it: QuizItem, correct: bool) -> str:
    """The sample in contracts/samples/submit_quiz.json starts with the verdict."""
    ar = _is_arabic(it.explanation) or _is_arabic(it.stem)
    right = it.options[it.right_choice]
    if correct:
        head = "إجابة صحيحة." if ar else "Correct."
    else:
        head = f"الإجابة الصحيحة هي {right}." if ar else f"The correct answer is {right}."
    return f"{head} {it.explanation}".strip()


def to_contract_result(
    items: list[QuizItem], submit: SubmitResponse, before: Profile, after: Profile
) -> dict:
    """The QuizResult shape: a count, one explanation per question, and mastery change."""
    verdict = {r.item_id: r.correct for r in submit.items}
    before_m = {t.topic_id: t.mastery for t in before.topics}
    after_m = {t.topic_id: t.mastery for t in after.topics}

    topics: list[str] = []
    for it in items:
        if it.topic_id not in topics:
            topics.append(it.topic_id)

    return {
        "score": sum(1 for it in items if verdict.get(it.item_id)),
        "total": len(items),
        "explanations": [_explanation(it, bool(verdict.get(it.item_id))) for it in items],
        "mastery_change": [
            {
                "topic_id": t,
                "change": round(after_m.get(t, INITIAL_MASTERY) - before_m.get(t, INITIAL_MASTERY), 2),
            }
            for t in topics
        ],
    }


# ---------- /profile/summary ----------
def level_of(profile: Profile) -> str:
    """beginner / intermediate / advanced from the average mastery of topics the student
    has tried. These cut-offs are a design choice, not a tuned result."""
    tried = [t.mastery for t in profile.topics if t.attempts > 0]
    if not tried:
        return "beginner"
    avg = sum(tried) / len(tried)
    if avg < 0.4:
        return "beginner"
    if avg < 0.7:
        return "intermediate"
    return "advanced"


def to_student_summary(profile: Profile) -> dict:
    weak = sorted((t for t in profile.topics if t.weak), key=lambda t: t.mastery)
    return {
        "weak_topics": [
            {"topic_id": t.topic_id, "score": min(1.0, max(0.0, t.mastery))} for t in weak
        ],
        "level": level_of(profile),
    }


# ---------- /next-step ----------
def pick_next_topic(profile: Profile) -> Optional[TopicProfile]:
    """The weakest weak topic; if none is weak, the topic with the lowest mastery."""
    if not profile.topics:
        return None
    weak = [t for t in profile.topics if t.weak]
    pool = weak or profile.topics
    return min(pool, key=lambda t: (t.mastery, t.attempts))


def next_step(topic: TopicProfile) -> dict:
    name = topic.name or topic.topic_id
    if topic.weak:
        text = f"راجع موضوع «{name}» ثم جرّب اختبارًا قصيرًا."
    else:
        text = f"كمّل التدرّب على «{name}» باختبار قصير."
    return {"topic_id": topic.topic_id, "suggestion": text}


def first_topic_step(topic_id: str, name: str) -> dict:
    """For a student with no profile yet: start with a topic from the course."""
    return {"topic_id": topic_id, "suggestion": f"ابدأ بموضوع «{name or topic_id}» باختبار قصير."}


# ---------- /events ----------
class QuestionEvent(BaseModel):
    """The body M3 sends (contracts/openapi.yaml, QuestionEvent)."""

    user_id: str
    room_id: str
    topic_id: str
    kind: Literal["question", "understood"]
    text: str
    time: datetime


def question_event_to_event(ev: QuestionEvent) -> Optional[Event]:
    """Only 'question' is recorded. 'understood' is a proposal the team has not accepted,
    so it is accepted and ignored. A question never moves mastery by itself: it only
    counts towards 'asked 3 questions since the last quiz'."""
    if ev.kind != "question":
        return None
    return Event(
        user_id=ev.user_id,
        room_id=ev.room_id,
        topic_id=ev.topic_id,
        kind="question",
        score=0.0,
        meta={"text": ev.text, "time": ev.time.isoformat()},
    )

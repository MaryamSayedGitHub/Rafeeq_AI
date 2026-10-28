"""Translate between M4's own shapes and the team contract (contracts/openapi.yaml).

Everything here is a plain function: no database, no web framework. That keeps the
translation easy to test, and keeps M4's real logic (profile, quiz, grader) untouched.

Team contract                         M4 inside
-----------------------------------   ---------------------------------------------
Quiz.questions[].choices              QuizItem.options  (empty for the typed kinds)
Quiz.questions[].correct_answer       QuizItem.options[right_choice]  (text, not a number); the model answer for typed kinds
Quiz.questions[].kind                 QuizItem.kind (mcq, true_false, short, math, problem, essay)
SubmitQuizRequest.answers (texts)     list[Answer(item_id, choice index)] for mcq / true_false, Answer(text=...) for typed kinds
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
                "correct_answer": it.options[it.right_choice] if it.is_choice else it.answer,
                "explanation": it.explanation,
                "kind": it.kind,
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
        if not it.is_choice:   # short, math, problem, essay: the text itself is the answer
            out.append(Answer(item_id=it.item_id, choice=-1, text=text))
            continue
        options = [o.strip() for o in it.options]
        choice = options.index(text) if text in options else -1
        out.append(Answer(item_id=it.item_id, choice=choice))
    return out


def _explanation(it: QuizItem, correct: bool, score: float = 0.0, feedback: str = "") -> str:
    """The sample in contracts/samples/submit_quiz.json starts with the verdict."""
    ar = _is_arabic(it.explanation) or _is_arabic(it.stem)
    if not it.is_choice:
        # a typed answer can earn part of the marks: say how much, what was missing, then the model answer
        pct = round(score * 100)
        if correct:
            head = f"إجابة صحيحة ({pct}%)." if ar else f"Correct ({pct}%)."
        elif score > 0:
            head = f"إجابة جزئية ({pct}%)." if ar else f"Partly right ({pct}%)."
        else:
            head = "إجابة غير صحيحة." if ar else "Not correct."
        model = f"الإجابة النموذجية: {it.answer}" if ar else f"Model answer: {it.answer}"
        return " ".join(x for x in (head, feedback.strip(), model, it.explanation) if x).strip()
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
    graded = {r.item_id: r for r in submit.items}
    before_m = {t.topic_id: t.mastery for t in before.topics}
    after_m = {t.topic_id: t.mastery for t in after.topics}

    topics: list[str] = []
    for it in items:
        if it.topic_id not in topics:
            topics.append(it.topic_id)

    return {
        "score": sum(1 for it in items if verdict.get(it.item_id)),
        "total": len(items),
        "explanations": [
            _explanation(it, bool(verdict.get(it.item_id)), graded[it.item_id].score if it.item_id in graded else 0.0,
                         graded[it.item_id].feedback if it.item_id in graded else "")
            for it in items
        ],
        # one 0..1 mark per question, same order: a typed answer can earn part of the marks
        "item_scores": [round(graded[it.item_id].score, 2) if it.item_id in graded else 0.0 for it in items],
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

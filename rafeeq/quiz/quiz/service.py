"""Quiz logic: create, grade, and the instructor data built from answers."""
from __future__ import annotations

import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
import uuid
from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from rafeeq.quiz.contracts import (
    Answer,
    ClassProfileRow,
    DetectedAnswer,
    Event,
    GradeSheetResponse,
    ItemResult,
    QuizCreateResponse,
    QuizItem,
    SubmitResponse,
    WrongAnswerRow,
)
from rafeeq.quiz.profile import service as profile_service
from rafeeq.quiz.profile.models import TopicMasteryRow
from rafeeq.quiz.quiz.generator import LLM, generate_quiz
from rafeeq.quiz.quiz.grading import Grade, grade_typed_answer
from rafeeq.quiz.quiz.models import AnswerRow, QuizItemRow, QuizRow, answer_score

# M5's approved-items: (room_id, weak_topic_ids) -> lecturer-approved items
ApprovedFn = Callable[[str, list[str]], list[QuizItem]]


class QuizGenerationError(RuntimeError):
    pass


class QuizNotFound(LookupError):
    pass


class TopicNotFound(LookupError):
    pass


class AlreadySubmitted(Exception):
    pass


def create_quiz(
    session: Session,
    *,
    user_id: str,
    room_id: str,
    n: int,
    mode: str,
    llm: LLM,
    chunks: list[dict],
    approved_items: Optional[ApprovedFn] = None,
    rng: Optional[random.Random] = None,
    topic_id: Optional[str] = None,
    verifier: Optional[LLM] = None,
    kinds: tuple[str, ...] = ("mcq",),
) -> QuizCreateResponse:
    """`kinds`: the question types to mix. A paper quiz is always mcq (bubbles can only mark an option)."""
    if mode == "paper":
        kinds = ("mcq",)
    profile = profile_service.get_profile(session, user_id, room_id)
    weak = [t.topic_id for t in profile.topics if t.weak]
    if topic_id:  # student picked a topic: use only it, ignore the weak-topic mix
        chunks = [c for c in chunks if c["topic_id"] == topic_id]
        if not chunks:
            raise TopicNotFound(topic_id)
        weak = [topic_id]
    quiz_id = f"quiz-{uuid.uuid4().hex[:8]}"

    # 1. lecturer-approved items first (weak topics ahead of the rest)
    items: list[QuizItem] = []
    if approved_items is not None:
        approved = approved_items(room_id, weak) if "mcq" in kinds else []   # lecturer-approved items are all mcq
        if topic_id:
            approved = [it for it in approved if it.topic_id == topic_id]
        approved = sorted(approved, key=lambda it: it.topic_id not in weak)
        items = [it.model_copy(update={"source": "lecturer_approved"}) for it in approved[:n]]

    # 2. generate the rest from the course chunks, aimed at weak topics
    missing = n - len(items)
    if missing > 0:
        result = generate_quiz(llm, chunks, missing, weak_topic_ids=weak, rng=rng, verifier=verifier, kinds=kinds)
        for it in result.items:
            items.append(it.model_copy(update={"item_id": f"{quiz_id}-g{len(items) + 1}"}))

    if not items:
        raise QuizGenerationError("could not generate any questions, try again")

    session.add(QuizRow(quiz_id=quiz_id, room_id=room_id, user_id=user_id, mode=mode))
    for position, it in enumerate(items):
        session.add(
            QuizItemRow(quiz_id=quiz_id, item_id=it.item_id, position=position, data=it.model_dump())
        )
    session.commit()

    sheet_pdf = None
    if mode == "paper":  # printable PDF: answer sheet + the questions
        from rafeeq.quiz.sheet.make_pdf import build_sheet_pdf, sheet_path

        build_sheet_pdf(quiz_id, items, sheet_path(quiz_id))
        sheet_pdf = f"/quizzes/{quiz_id}/sheet.pdf"
    return QuizCreateResponse(quiz_id=quiz_id, items=items, sheet_pdf=sheet_pdf)


def _load_items(session: Session, quiz_id: str) -> list[QuizItem]:
    rows = session.scalars(
        select(QuizItemRow).where(QuizItemRow.quiz_id == quiz_id).order_by(QuizItemRow.position)
    ).all()
    return [QuizItem.model_validate(r.data) for r in rows]


def get_quiz_items(session: Session, *, quiz_id: str, user_id: str) -> list[QuizItem]:
    """The quiz's questions in order, only for the student who owns it."""
    quiz = session.get(QuizRow, quiz_id)
    if quiz is None:
        raise QuizNotFound(quiz_id)
    if quiz.user_id != user_id:
        raise PermissionError("this quiz belongs to another student")
    return _load_items(session, quiz_id)


REVIEW_BELOW = 0.5  # detections less sure than this are sent back for a human look


def grade_detected(items: list[QuizItem], marks: list) -> GradeSheetResponse:
    """Turn photo detections into a response. Nothing is saved: the student fixes
    the flagged answers, then calls /submit with the final list."""
    answers: list[DetectedAnswer] = []
    review: list[str] = []
    correct = 0
    for it, mark in zip(items, marks):
        answers.append(DetectedAnswer(item_id=it.item_id, choice=mark.choice, confidence=mark.confidence))
        if mark.choice is None or mark.confidence < REVIEW_BELOW:
            review.append(it.item_id)
        correct += mark.choice == it.right_choice
    score = correct / len(items) if items else 0.0
    return GradeSheetResponse(answers=answers, score=score, needs_review=review)


GRADING_THREADS = 4


def _grade_item(it: QuizItem, answer: Answer, grader_llm: Optional[LLM]) -> Grade:
    """One item -> its Grade. Choice items compare the option with the key; typed items go to grading.py."""
    if it.is_choice:
        ok = answer.choice == it.right_choice
        return Grade(1.0 if ok else 0.0)
    return grade_typed_answer(it, answer.text, grader_llm)


def submit_quiz(
    session: Session, *, quiz_id: str, user_id: str, answers: list[Answer], grader_llm: Optional[LLM] = None
) -> SubmitResponse:
    quiz = session.get(QuizRow, quiz_id)
    if quiz is None:
        raise QuizNotFound(quiz_id)
    if quiz.user_id != user_id:
        raise PermissionError("this quiz belongs to another student")
    already = session.scalar(
        select(AnswerRow.id)
        .where(AnswerRow.quiz_id == quiz_id, AnswerRow.user_id == user_id)
        .limit(1)
    )
    if already is not None:
        raise AlreadySubmitted(quiz_id)

    items = _load_items(session, quiz_id)
    given = {a.item_id: a for a in answers}
    blank = lambda it: Answer(item_id=it.item_id, choice=-1, text="")  # noqa: E731
    picked = [given.get(it.item_id) or blank(it) for it in items]

    # typed answers may need an LLM call each: do them side by side (mcq / true_false are instant)
    typed = [i for i, it in enumerate(items) if not it.is_choice and picked[i].text.strip()]
    grades: dict[int, Grade] = {}
    if typed:
        with ThreadPoolExecutor(max_workers=min(GRADING_THREADS, len(typed))) as pool:
            for i, g in zip(typed, pool.map(lambda i: _grade_item(items[i], picked[i], grader_llm), typed)):
                grades[i] = g
    results: list[ItemResult] = []
    scores_by_topic: dict[str, list[float]] = {}

    for i, it in enumerate(items):
        grade = grades.get(i) or _grade_item(items[i], picked[i], grader_llm)
        session.add(
            AnswerRow(
                quiz_id=quiz_id,
                item_id=it.item_id,
                user_id=user_id,
                room_id=quiz.room_id,
                topic_id=it.topic_id,
                chosen=picked[i].choice if it.is_choice else -1,
                correct=grade.correct,
                answer_text="" if it.is_choice else picked[i].text.strip()[:4000],
                score=grade.score,
                feedback=grade.feedback,
                needs_review=grade.needs_review,
            )
        )
        scores_by_topic.setdefault(it.topic_id, []).append(grade.score)
        results.append(
            ItemResult(
                item_id=it.item_id,
                correct=grade.correct,
                right_choice=it.right_choice,
                explanation=it.explanation,
                file_name=it.file_name,
                page=it.page,
                score=grade.score,
                feedback=grade.feedback,
                needs_review=grade.needs_review,
            )
        )
    session.commit()

    # one quiz event per topic: the score is the average share of the marks in that topic
    for topic_id, topic_scores_ in scores_by_topic.items():
        profile_service.record_event(
            session,
            Event(
                user_id=user_id,
                room_id=quiz.room_id,
                topic_id=topic_id,
                kind="quiz",
                score=sum(topic_scores_) / len(topic_scores_),
                meta={"quiz_id": quiz_id, "items": len(topic_scores_)},
            ),
        )

    total = len(results)
    score = sum(r.score for r in results) / total if total else 0.0
    return SubmitResponse(score=score, items=results)


def quiz_history(session: Session, *, user_id: str, room_id: str) -> list[dict]:
    """The student's SUBMITTED quizzes in a room, newest first: when, score, and which topics."""
    rows = session.scalars(
        select(AnswerRow).where(AnswerRow.user_id == user_id, AnswerRow.room_id == room_id).order_by(AnswerRow.id)
    ).all()
    by_quiz: dict[str, list[AnswerRow]] = {}
    for r in rows:
        by_quiz.setdefault(r.quiz_id, []).append(r)
    history = []
    for quiz_id, answers in by_quiz.items():
        correct = sum(a.correct for a in answers)
        history.append({
            "quiz_id": quiz_id, "submitted_at": max(a.created_at for a in answers).isoformat(),
            "total": len(answers), "correct": correct, "score": sum(answer_score(a) for a in answers) / len(answers),
            "topic_ids": sorted({a.topic_id for a in answers}),
        })
    history.sort(key=lambda h: h["submitted_at"], reverse=True)
    return history


WEAK_AVERAGE = 0.6  # a topic whose average quiz score is below this needs review


def topic_scores(session: Session, *, user_id: str, room_id: str) -> list[dict]:
    """Per topic, the average of this student's quiz scores in it (each quiz counts once: its correct/total for that topic).
    Quizzes only: questions asked in the chat do not count. Lowest average first."""
    rows = session.scalars(
        select(AnswerRow).where(AnswerRow.user_id == user_id, AnswerRow.room_id == room_id)
    ).all()
    per_quiz: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        per_quiz.setdefault((r.topic_id, r.quiz_id), []).append(answer_score(r))
    by_topic: dict[str, list[float]] = {}
    for (topic_id, _), flags in per_quiz.items():
        by_topic.setdefault(topic_id, []).append(sum(flags) / len(flags))
    scores = [{"topic_id": t, "average": sum(v) / len(v), "quizzes": len(v), "weak": sum(v) / len(v) < WEAK_AVERAGE}
              for t, v in by_topic.items()]
    return sorted(scores, key=lambda s: s["average"])


def quiz_review(session: Session, *, user_id: str, quiz_id: str) -> dict:
    """One submitted quiz, question by question: what the student chose, the right answer and why. Only for its owner.
    Raises QuizNotFound when the quiz does not exist or was never submitted."""
    items = get_quiz_items(session, quiz_id=quiz_id, user_id=user_id)
    answers = {a.item_id: a for a in session.scalars(
        select(AnswerRow).where(AnswerRow.quiz_id == quiz_id, AnswerRow.user_id == user_id)).all()}
    if not answers:
        raise QuizNotFound(quiz_id)
    questions = []
    for it in items:
        a = answers.get(it.item_id)
        chosen = a.chosen if a else -1
        questions.append({
            "item_id": it.item_id, "topic_id": it.topic_id, "text": it.stem, "options": list(it.options),
            "chosen": chosen if 0 <= chosen < len(it.options) else None, "right_choice": it.right_choice,
            "correct": bool(a and a.correct), "explanation": it.explanation, "file_name": it.file_name, "page": it.page,
            # the question types beyond mcq: what the student typed, the model answer, the marks and the feedback
            "kind": it.kind, "student_answer": a.answer_text if a else "", "model_answer": it.answer,
            "rubric": list(it.rubric), "score": answer_score(a) if a else 0.0,
            "feedback": (a.feedback if a else "") or "", "needs_review": bool(a and a.needs_review),
        })
    correct = sum(q["correct"] for q in questions)
    return {"quiz_id": quiz_id, "submitted_at": max(a.created_at for a in answers.values()).isoformat(),
            "total": len(questions), "correct": correct,
            "score": sum(q["score"] for q in questions) / len(questions) if questions else 0.0,
            "questions": questions}


def _student_ref(room_id: str, user_id: str) -> str:
    """Stable anonymous reference: the same student gets the same ref in a room."""
    return "s-" + hashlib.sha256(f"{room_id}:{user_id}".encode()).hexdigest()[:8]


def wrong_answers(session: Session, room_id: str) -> list[WrongAnswerRow]:
    """Instructor data. No names and no user ids."""
    rows = session.scalars(
        select(AnswerRow)
        .where(AnswerRow.room_id == room_id, AnswerRow.correct.is_(False))
        .order_by(AnswerRow.id)
    ).all()
    out: list[WrongAnswerRow] = []
    for a in rows:
        item_row = session.scalar(
            select(QuizItemRow).where(
                QuizItemRow.quiz_id == a.quiz_id, QuizItemRow.item_id == a.item_id
            )
        )
        if item_row is None:
            continue
        item = QuizItem.model_validate(item_row.data)
        out.append(
            WrongAnswerRow(
                item_id=a.item_id,
                student_ref=_student_ref(room_id, a.user_id),
                topic_id=a.topic_id,
                stem=item.stem,
                chosen=a.chosen,
                right=item.right_choice,
            )
        )
    return out


def class_profile(
    session: Session, room_id: str, names: Optional[dict[str, str]] = None
) -> list[ClassProfileRow]:
    """Instructor data. `names` comes from M1 (user_id -> name); until then the id is used."""
    names = names or {}
    rows = session.scalars(
        select(TopicMasteryRow)
        .where(TopicMasteryRow.room_id == room_id)
        .order_by(TopicMasteryRow.user_id, TopicMasteryRow.topic_id)
    ).all()
    return [
        ClassProfileRow(
            user_id=r.user_id,
            name=names.get(r.user_id, r.user_id),
            topic_id=r.topic_id,
            mastery=r.mastery,
            attempts=r.attempts,
        )
        for r in rows
    ]
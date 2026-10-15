"""M4 routes, exactly as the team contract (contracts/openapi.yaml) defines them.

    POST /quiz                  make a quiz           -> Quiz
    POST /quiz/{quiz_id}/submit send answers          -> QuizResult
    GET  /next-step?room_id=    what to do next       -> NextStep
    GET  /profile/summary?room_id=  weak topics       -> StudentSummary
    POST /events                a question was asked  -> 204

M4's own routes in api.py (paper sheet, grade a photo, class profile, wrong answers,
record_event) are included too, except /rooms/{room_id}/topics, which belongs to M2.

M1 connects this with one line in main.py:

    from rafeeq.quiz.router import install
    install(app)

install() also swaps M4's stand-in current_user_id for the real login (Bearer token).
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from rafeeq.quiz import adapters
from rafeeq.quiz import api as m4_api
from rafeeq.quiz.contracts import QuizCreateResponse
from rafeeq.quiz.deps import current_user_id, get_approved_items, get_chunks, get_llm, get_room_chunks, get_verifier, material
from rafeeq.quiz.profile import service as profile_service
from rafeeq.quiz.profile.db import get_session
from rafeeq.quiz.quiz import service as quiz_service
from rafeeq.quiz.quiz.models import AnswerRow, QuizRow
from rafeeq.quiz.sheet.make_pdf import build_sheet_pdf, sheet_path
from rafeeq.quiz.sheet.make_report import build_report

router = APIRouter(tags=["M4 quiz and profile"])


# ---------- request bodies (the contract's shapes) ----------
class MakeQuizBody(BaseModel):
    room_id: str
    topic_id: Optional[str] = None  # empty or missing = a mix of the room's topics
    count: int = Field(default=5, ge=1, le=20)


class SubmitBody(BaseModel):
    answers: list[str]  # one answer text per question, in the same order as the quiz


def _topic_names(chunks) -> dict[str, str]:
    return {c["topic_id"]: (c.get("topic_name") or c["topic_id"]) for c in chunks}


# ---------- POST /quiz ----------
@router.post("/quiz")
def make_quiz(
    body: MakeQuizBody,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
    llm=Depends(get_llm),
    chunks=Depends(get_chunks),
    approved=Depends(get_approved_items),
    verifier=Depends(get_verifier),
    room_fn=Depends(get_room_chunks),
) -> dict:
    chunks = material(room_fn, body.room_id, chunks)
    topic_id = (body.topic_id or "").strip() or None
    try:
        made = quiz_service.create_quiz(
            session,
            user_id=user_id,
            room_id=body.room_id,
            n=body.count,
            mode="screen",
            llm=llm,
            chunks=chunks,
            approved_items=approved,
            topic_id=topic_id,
            verifier=verifier,
        )
    except quiz_service.TopicNotFound:
        raise HTTPException(status_code=404, detail="topic not found in this course")
    except quiz_service.QuizGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return adapters.to_contract_quiz(QuizCreateResponse.model_validate(made), topic_id)


# ---------- POST /quiz/print (M4 extra, not in the contract) ----------
@router.post("/quiz/print")
def make_printable_quiz(
    body: MakeQuizBody,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
    llm=Depends(get_llm),
    chunks=Depends(get_chunks),
    approved=Depends(get_approved_items),
    verifier=Depends(get_verifier),
    room_fn=Depends(get_room_chunks),
):
    """Makes a NEW quiz in paper mode and returns its printable PDF (answer sheet + questions).
    The quiz id is in the X-Quiz-Id header; use it later with POST /quizzes/{id}/grade-sheet."""
    chunks = material(room_fn, body.room_id, chunks)
    topic_id = (body.topic_id or "").strip() or None
    try:
        made = quiz_service.create_quiz(
            session, user_id=user_id, room_id=body.room_id, n=body.count, mode="paper",
            llm=llm, chunks=chunks, approved_items=approved, topic_id=topic_id, verifier=verifier,
        )
    except quiz_service.TopicNotFound:
        raise HTTPException(status_code=404, detail="topic not found in this course")
    except quiz_service.QuizGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    quiz_id = made["quiz_id"] if isinstance(made, dict) else made.quiz_id
    path = sheet_path(quiz_id)
    if not path.exists():
        raise HTTPException(status_code=500, detail="the sheet PDF was not created")
    return FileResponse(path, media_type="application/pdf", filename=f"{quiz_id}.pdf",
                        headers={"X-Quiz-Id": quiz_id})


# ---------- downloads: the quiz as a printable PDF, and the score report (M4 extras) ----------
def _resolve_quiz_id(session: Session, user_id: str, quiz_id: str, room_id: Optional[str]) -> str:
    """quiz_id can be the word "latest" (with room_id): the student's most recent quiz in that room."""
    if quiz_id != "latest":
        return quiz_id
    if not room_id:
        raise HTTPException(status_code=422, detail="room_id is needed when quiz_id is 'latest'")
    row = session.scalars(
        select(QuizRow).where(QuizRow.user_id == user_id, QuizRow.room_id == room_id)
        .order_by(QuizRow.created_at.desc())
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="no quiz yet in this room")
    return row.quiz_id


def _items_or_http_error(session: Session, user_id: str, quiz_id: str):
    try:
        return quiz_service.get_quiz_items(session, quiz_id=quiz_id, user_id=user_id)
    except quiz_service.QuizNotFound:
        raise HTTPException(status_code=404, detail="quiz not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="this quiz belongs to another student")


@router.get("/quiz/{quiz_id}/print.pdf")
def quiz_print_pdf(
    quiz_id: str,
    room_id: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
):
    """The questions plus a bubble answer sheet, as a PDF. Works for any quiz of the student."""
    quiz_id = _resolve_quiz_id(session, user_id, quiz_id, room_id)
    items = _items_or_http_error(session, user_id, quiz_id)
    path = sheet_path(quiz_id)
    if not path.exists():
        try:
            build_sheet_pdf(quiz_id, items, path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
    return FileResponse(path, media_type="application/pdf", filename=f"{quiz_id}.pdf",
                        headers={"X-Quiz-Id": quiz_id})


@router.get("/quiz/{quiz_id}/report.pdf")
def quiz_report_pdf(
    quiz_id: str,
    room_id: Optional[str] = Query(default=None),
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
):
    """The score report of a SUBMITTED quiz: each question, the student's answer, the right one, why."""
    quiz_id = _resolve_quiz_id(session, user_id, quiz_id, room_id)
    items = _items_or_http_error(session, user_id, quiz_id)
    rows = session.scalars(
        select(AnswerRow).where(AnswerRow.quiz_id == quiz_id, AnswerRow.user_id == user_id)
    ).all()
    if not rows:
        raise HTTPException(status_code=409, detail="submit the quiz first, then ask for the report")
    chosen = {r.item_id: (None if r.chosen < 0 else r.chosen) for r in rows}
    quiz = {"quiz_id": quiz_id, "items": [it.model_dump() for it in items]}
    result = {"answers": [{"item_id": it.item_id, "choice": chosen.get(it.item_id)} for it in items]}
    out = sheet_path(quiz_id).with_name(f"{quiz_id}-report.pdf")
    build_report(quiz, result, out)
    return FileResponse(out, media_type="application/pdf", filename=f"{quiz_id}-report.pdf",
                        headers={"X-Quiz-Id": quiz_id})


# ---------- the student's quiz history (M4 extras) ----------
@router.get("/rooms/{room_id}/quiz-history")
def my_quiz_history(
    room_id: str,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
) -> dict:
    """Every quiz this student submitted in the room, newest first."""
    return {"quizzes": quiz_service.quiz_history(session, user_id=user_id, room_id=room_id)}


@router.get("/rooms/{room_id}/topic-scores")
def my_topic_scores(
    room_id: str,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
) -> dict:
    """This student's average quiz score per topic (quizzes only), lowest first."""
    return {"topics": quiz_service.topic_scores(session, user_id=user_id, room_id=room_id)}


@router.get("/quiz/{quiz_id}/review")
def quiz_review(
    quiz_id: str,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
) -> dict:
    """One submitted quiz with the student's answers, the right ones and the explanations."""
    try:
        return quiz_service.quiz_review(session, user_id=user_id, quiz_id=quiz_id)
    except quiz_service.QuizNotFound:
        raise HTTPException(status_code=404, detail="quiz not found or not submitted yet")
    except PermissionError:
        raise HTTPException(status_code=403, detail="this quiz belongs to another student")


# ---------- POST /quiz/{quiz_id}/submit ----------
@router.post("/quiz/{quiz_id}/submit")
def submit_quiz(
    quiz_id: str,
    body: SubmitBody,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
) -> dict:
    try:
        items = quiz_service.get_quiz_items(session, quiz_id=quiz_id, user_id=user_id)
    except quiz_service.QuizNotFound:
        raise HTTPException(status_code=404, detail="quiz not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="this quiz belongs to another student")

    room_id = session.get(QuizRow, quiz_id).room_id
    before = profile_service.get_profile(session, user_id, room_id)
    try:
        result = quiz_service.submit_quiz(
            session,
            quiz_id=quiz_id,
            user_id=user_id,
            answers=adapters.answers_to_choices(items, body.answers),
        )
    except quiz_service.AlreadySubmitted:
        raise HTTPException(status_code=409, detail="this quiz was already submitted")
    after = profile_service.get_profile(session, user_id, room_id)
    _tell_the_teacher(room_id, user_id, items, body.answers)
    return adapters.to_contract_result(items, result, before, after)


def _tell_the_teacher(room_id: str, user_id: str, items, answers: list[str]) -> None:
    """M5's class report (who is struggling, repeated mistakes) is fed by every submitted quiz,
    once per topic. Never fails the student's submission."""
    try:
        from rafeeq.teacher import class_report

        by_topic: dict[str, list[dict]] = {}
        for item, chosen in zip(items, answers):
            by_topic.setdefault(item.topic_id, []).append(
                {"question": item.stem, "chosen": chosen, "correct": item.options[item.right_choice]})
        for topic_id, rows in by_topic.items():
            class_report.record_quiz_result(room_id, user_id, topic_id, rows)
    except Exception:
        pass


# ---------- GET /profile/summary ----------
@router.get("/profile/summary")
def profile_summary(
    room_id: str = Query(...),
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
) -> dict:
    return adapters.to_student_summary(profile_service.get_profile(session, user_id, room_id))


# ---------- GET /next-step ----------
@router.get("/next-step")
def next_step(
    room_id: str = Query(...),
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
    chunks=Depends(get_chunks),
    room_fn=Depends(get_room_chunks),
) -> dict:
    names = _topic_names(material(room_fn, room_id, chunks))
    profile = profile_service.get_profile(session, user_id, room_id, names)
    topic = adapters.pick_next_topic(profile)
    if topic is not None:
        return adapters.next_step(topic)
    if names:  # a new student: start with the first topic of the course
        first = next(iter(names))
        return adapters.first_topic_step(first, names[first])
    raise HTTPException(status_code=404, detail="no topics in this room yet")


# ---------- POST /events ----------
@router.post("/events", status_code=204)
def record_question_event(
    body: adapters.QuestionEvent, session: Session = Depends(get_session)
) -> Response:
    event = adapters.question_event_to_event(body)
    if event is not None:
        profile_service.record_event(session, event)
    return Response(status_code=204)


# ---------- M4's own extra routes (paper sheet, class data) ----------
_M2_OWNS = {"/rooms/{room_id}/topics"}
_extras = APIRouter()
_extras.routes.extend(r for r in m4_api.router.routes if getattr(r, "path", None) not in _M2_OWNS)
router.include_router(_extras)


# ---------- connecting to M1's login ----------
def bearer_user_id(authorization: Optional[str] = Header(default=None)) -> str:
    """The logged-in student, from M1's Bearer token."""
    from rafeeq.platform import service as platform

    token = platform.token_from_header(authorization)
    user = platform.get_store().user_for_token(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Log in first")
    return user["user_id"]


def install(app) -> None:
    """One call from main.py: mount the routes and use the real login."""
    app.dependency_overrides[current_user_id] = bearer_user_id
    app.include_router(router)
    from rafeeq.quiz.copilot_hooks import connect

    connect()  # the Copilot's "what next" and question events use M4 directly

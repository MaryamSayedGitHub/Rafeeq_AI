"""M4 router. M1 mounts it; role checks come from M1 (stubbed here)."""
import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from rafeeq.platform import access
from rafeeq.quiz.contracts import (
    ClassProfileRow,
    Event,
    GradeSheetResponse,
    Profile,
    QuizCreateRequest,
    QuizCreateResponse,
    SubmitRequest,
    SubmitResponse,
    WrongAnswerRow,
)
from rafeeq.quiz.deps import current_user_id, get_approved_items, get_chunks, get_grader_llm, get_llm, get_room_chunks, get_verifier, material
from rafeeq.quiz.profile import service
from rafeeq.quiz.profile.db import get_session
from rafeeq.quiz.quiz import service as quiz_service
from rafeeq.quiz.sheet import grader
from rafeeq.quiz.sheet.make_pdf import sheet_path

MAX_PHOTO_BYTES = 15 * 1024 * 1024

router = APIRouter()


def _own_event(authorization: str | None, user_id: str, room_id: str) -> None:
    """An event moves a student's mastery: over HTTP only that logged-in student, in a room they belong to,
    may send it (the app's own modules record events in-process and never come through here)."""
    if access.anonymous_allowed() and not authorization:
        return
    user = access.user_from_header(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Log in first")
    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="You can only send your own events")
    access.member_by_id(user_id, room_id)


@router.post("/internal/record_event")
def record_event(event: Event, session: Session = Depends(get_session), authorization: str | None = Header(default=None)) -> dict:
    _own_event(authorization, event.user_id, event.room_id)
    state = service.record_event(session, event)
    return {"ok": True, "mastery": state.mastery, "weak": service.is_weak(state)}


@router.get("/rooms/{room_id}/me/profile", response_model=Profile)
def my_profile(
    room_id: str,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
):
    return service.get_profile(session, user_id, room_id)


@router.get("/rooms/{room_id}/topics")
def topics(room_id: str, chunks=Depends(get_chunks)) -> list[dict]:
    # TODO: use M2's list_topics(room_id) when it is live
    seen: dict[str, str] = {}
    for c in chunks:
        seen.setdefault(c["topic_id"], c["topic_name"])
    return [{"topic_id": k, "name": v} for k, v in seen.items()]


@router.post("/rooms/{room_id}/quizzes", response_model=QuizCreateResponse)
def create_quiz(
    room_id: str,
    req: QuizCreateRequest,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
    llm=Depends(get_llm),
    chunks=Depends(get_chunks),
    approved=Depends(get_approved_items),
    verifier=Depends(get_verifier),
    room_fn=Depends(get_room_chunks),
):
    access.member_by_id(user_id, room_id)
    chunks = material(room_fn, room_id, chunks)
    try:
        return quiz_service.create_quiz(
            session,
            user_id=user_id,
            room_id=room_id,
            n=req.n,
            mode=req.mode,
            llm=llm,
            chunks=chunks,
            approved_items=approved,
            topic_id=req.topic_id,
            verifier=verifier,
            kinds=tuple(req.kinds),
        )
    except quiz_service.TopicNotFound:
        raise HTTPException(status_code=404, detail="topic not found in this course")
    except quiz_service.QuizGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except (httpx.HTTPError, RuntimeError) as exc:   # the model could not be reached (timeout, rate limit, bad key)
        raise HTTPException(status_code=503, detail=f"the quiz model could not be reached, try again: {str(exc)[:120]}")


@router.post("/quizzes/{quiz_id}/submit", response_model=SubmitResponse)
def submit(
    quiz_id: str,
    req: SubmitRequest,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
    grader_llm=Depends(get_grader_llm),
):
    try:
        return quiz_service.submit_quiz(
            session, quiz_id=quiz_id, user_id=user_id, answers=req.answers, grader_llm=grader_llm
        )
    except quiz_service.QuizNotFound:
        raise HTTPException(status_code=404, detail="quiz not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="this quiz belongs to another student")
    except quiz_service.AlreadySubmitted:
        raise HTTPException(status_code=409, detail="this quiz was already submitted")


def _owned_items(session: Session, quiz_id: str, user_id: str):
    try:
        return quiz_service.get_quiz_items(session, quiz_id=quiz_id, user_id=user_id)
    except quiz_service.QuizNotFound:
        raise HTTPException(status_code=404, detail="quiz not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="this quiz belongs to another student")


@router.get("/quizzes/{quiz_id}/sheet.pdf")
def sheet_pdf(
    quiz_id: str,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
):
    _owned_items(session, quiz_id, user_id)
    path = sheet_path(quiz_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="this quiz has no printable sheet (was it made with mode=paper?)")
    return FileResponse(path, media_type="application/pdf", filename=f"{quiz_id}.pdf")


@router.post("/quizzes/{quiz_id}/grade-sheet", response_model=GradeSheetResponse)
async def grade_sheet(
    quiz_id: str,
    photo: UploadFile,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
):
    """Read a photographed answer sheet. Saves nothing: fix the flagged answers,
    then send the final list to /quizzes/{id}/submit."""
    items = _owned_items(session, quiz_id, user_id)
    data = await photo.read()
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(status_code=413, detail="photo is larger than 15 MB")
    try:
        marks = await run_in_threadpool(grader.detect_marks, data, len(items))
    except grader.SheetError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return quiz_service.grade_detected(items, marks)


@router.get("/rooms/{room_id}/class-profile", response_model=list[ClassProfileRow])
def class_profile(room_id: str, session: Session = Depends(get_session), user_id: str = Depends(current_user_id)):
    access.teacher_by_id(user_id, room_id)
    return quiz_service.class_profile(session, room_id)


@router.get("/rooms/{room_id}/wrong-answers", response_model=list[WrongAnswerRow])
def wrong_answers(room_id: str, session: Session = Depends(get_session), user_id: str = Depends(current_user_id)):
    access.teacher_by_id(user_id, room_id)
    return quiz_service.wrong_answers(session, room_id)
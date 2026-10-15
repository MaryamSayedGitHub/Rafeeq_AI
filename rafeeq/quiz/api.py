"""M4 router. M1 mounts it; role checks come from M1 (stubbed here)."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

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
from rafeeq.quiz.deps import current_user_id, get_approved_items, get_chunks, get_llm, get_room_chunks, get_verifier, material
from rafeeq.quiz.profile import service
from rafeeq.quiz.profile.db import get_session
from rafeeq.quiz.quiz import service as quiz_service
from rafeeq.quiz.sheet import grader
from rafeeq.quiz.sheet.make_pdf import sheet_path

MAX_PHOTO_BYTES = 15 * 1024 * 1024

router = APIRouter()


@router.post("/internal/record_event")
def record_event(event: Event, session: Session = Depends(get_session)) -> dict:
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
        )
    except quiz_service.TopicNotFound:
        raise HTTPException(status_code=404, detail="topic not found in this course")
    except quiz_service.QuizGenerationError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/quizzes/{quiz_id}/submit", response_model=SubmitResponse)
def submit(
    quiz_id: str,
    req: SubmitRequest,
    session: Session = Depends(get_session),
    user_id: str = Depends(current_user_id),
):
    try:
        return quiz_service.submit_quiz(
            session, quiz_id=quiz_id, user_id=user_id, answers=req.answers
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
def class_profile(room_id: str, session: Session = Depends(get_session)):
    # TODO: require_instructor(room_id) from M1, and names from M1
    return quiz_service.class_profile(session, room_id)


@router.get("/rooms/{room_id}/wrong-answers", response_model=list[WrongAnswerRow])
def wrong_answers(room_id: str, session: Session = Depends(get_session)):
    # TODO: require_instructor(room_id) from M1
    return quiz_service.wrong_answers(session, room_id)
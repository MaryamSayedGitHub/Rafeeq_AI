"""M3 room routes (contracts/openapi.yaml, tag "M3 tutor"):

  POST /rooms/{room_id}/ask                      question + level 1 or 2 -> answer + citations + "not found" / Level 2 offer, or the study file
  POST /rooms/{room_id}/vision/check-work        photo of handwriting -> every line ok | check | error with one hint, boxes, course pages
  GET  /rooms/{room_id}/questions                the signed-in student's own question log
  GET  /rooms/{room_id}/instructor/questions     anonymous questions for M5 (no names, no ids); ?format=csv
  POST /rooms/{room_id}/instructor/publish       write the anonymous file and POST it to M5_PUBLISH_URL when it is set

Who is asking comes from the Authorization token or X-User-Id (see events.py). Every request and response is strict: an unknown field is an error.
"""
from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from rafeeq.snap.service import SnapError
from rafeeq.snap.vision_check import check_photo

from . import anonymous, research, study_store
from .events import current_user_id
from .tutor_agent import ACTIONS, act, ask_payload
from .agent import run_qa

router = APIRouter(tags=["M3 tutor"])
MAX_PHOTO_BYTES = 12 * 1024 * 1024


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(_Strict):
    file_name: str
    page: int = Field(ge=1)


class AskRequest(_Strict):
    question: str = Field(min_length=1, max_length=2000)
    level: Literal[1, 2] = 1
    language: Literal["ar", "en"] | None = None
    pages: int | None = Field(None, ge=1, le=research.MAX_PAGES, description="Level 2: study file length in pages (default 1; a number in the question wins).")
    topic_id: str | None = Field(None, description="Search only the passages of this topic.")
    parent_question_id: int | None = Field(None, description="Level 2 pressed after an ungrounded Level 1: updates that question instead of adding one.")
    quiz: bool = Field(False, description="Also make the 3 MCQ. Default false: questions are only made when the student asks for them.")


class Source(_Strict):
    n: int
    title: str
    url: str


class StudyFile(_Strict):
    id: int
    title: str
    markdown: str
    words: int
    degraded: bool
    sources: list[Source]
    queries: list[str]


class MCQ(_Strict):
    question: str
    options: list[str]
    answer: int
    explanation: str


class AskResponse(_Strict):
    question_id: int
    topic_id: str
    level: Literal[1, 2]
    answer: str
    citations: list[Citation]
    grounded: bool
    steps: list[str]
    offer_level2: bool = Field(description="True when Level 1 found nothing in the material: the client should offer the student Level 2.")
    material_source: Literal["m2", "sample", "none", "given", "web"]
    citation_support: Literal["supported", "unsupported", "unchecked", "none"] = Field("none", description="Do the cited M2 passages support the answer? unchecked = answer and passage are in different scripts.")
    answer_flagged: bool = Field(False, description="True when the model's answer was withheld because its own citations did not support it.")
    language: Literal["ar", "en"] = Field("ar", description="The language the answer is written in (what was asked for, else the language of the question). A study file uses the site language.")
    web_sources: list[Source] = Field([], description="A term answered from the web (no study file): the pages it came from.")
    study_file: StudyFile | None
    quiz: list[MCQ]
    error: str | None
    event_delivered: bool


class TutorActRequest(_Strict):
    action: Literal["auto", "ask", "explain", "research", "quiz", "summary", "cards", "files", "file", "rewrite", "help", "hint", "solution", "check", "teacher"] = "auto"
    text: str = Field("", max_length=4000, description="The question, the problem, the answer to rewrite, or the steps to check (by action).")
    language: Literal["ar", "en"] | None = None
    pages: int | None = Field(None, ge=1, le=research.MAX_PAGES)
    topic_id: str | None = None
    parent_question_id: int | None = None
    study_file_id: int | None = None
    mode: Literal["simpler", "example", "compare"] | None = None
    other: str | None = Field(None, max_length=200)
    question: str | None = Field(None, max_length=2000, description="rewrite: the question the answer belongs to.")
    key: str | None = Field(None, description="hint / solution: the key returned by action help.")
    n: int | None = Field(None, ge=1, le=3)
    file_id: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    with_quiz: bool | None = Field(None, description="ask: also make the 3 MCQ (default: no, they are made on request).")
    notify: bool | None = Field(None, description="False: log the question but do not send the M4 event (the caller sends it).")


class TutorActResponse(_Strict):
    action: str = Field(description="The tool that ran (auto resolves to ask or help).")
    ok: bool
    error: str | None
    result: dict = Field(description="Shape depends on the action: ask/research = RoomAskResponse, files, file, quiz, summary, cards, rewrite, help, hint, solution, check.")


class FrequentQuestion(_Strict):
    topic_id: str
    question: str
    count: int = Field(ge=1)
    students: int = Field(ge=1, description="How many different students asked it (a number, never who).")
    unanswered: int = Field(ge=0, description="Times the course material had no answer.")
    deeper: int = Field(ge=0, description="Times a student went to Level 2.")


class Box(_Strict):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(ge=0, le=1)
    height: float = Field(ge=0, le=1)


class VisionLine(_Strict):
    i: int = Field(ge=1)
    text: str
    status: Literal["ok", "check", "error"]
    hint: str = Field(description="Exactly one short hint for a step that is not ok; empty for an ok step. Never the corrected line or the answer.")
    box: Box


class VisionCheck(_Strict):
    file_id: str
    quality_ok: bool
    retake_tip: str = Field(description="Filled when the photo is blurry or dark: ask the student to retake it.")
    type: Literal["student_work", "material", "diagram", "other"]
    language: Literal["ar", "en"]
    confidence: float = Field(ge=0, le=1)
    kind: Literal["question", "math", "code"]
    complete: bool
    needs_help: bool
    hint: str
    needs_clarification: bool = Field(False, description="A line could not be read reliably: it is marked check and the student is asked what it says; nothing was judged.")
    clarification: str = ""
    lines: list[VisionLine]
    citations: list[Citation]
    event_delivered: bool
    annotated_png_base64: str | None


class LoggedQuestion(_Strict):
    question_id: int
    ts: str
    question: str
    topic_id: str
    level: int
    grounded: bool
    study_file_id: int | None


class InstructorQuestion(_Strict):
    public_id: str
    date: str
    topic_id: str
    level: int
    grounded: bool
    question: str


class PublishResult(_Strict):
    count: int
    file: str
    posted: bool
    error: str


def _user(request: Request) -> str | None:
    return current_user_id(request.headers)


def require_instructor(x_instructor_pin: str | None = Header(None)) -> None:
    pin = os.getenv("INSTRUCTOR_PIN")
    if pin and x_instructor_pin != pin:
        raise HTTPException(401, "Instructor PIN required (header X-Instructor-Pin).")


@router.post("/rooms/{room_id}/ask", response_model=AskResponse)
def ask(room_id: str, body: AskRequest, request: Request):
    st = run_qa(room_id, body.question, body.level, _user(request), body.language, body.pages, body.topic_id, body.parent_question_id, with_quiz=body.quiz)
    return ask_payload(st, body.level)


@router.post("/rooms/{room_id}/vision/check-work", response_model=VisionCheck)
async def vision_check(room_id: str, request: Request, photo: UploadFile = File(...), language: Literal["ar", "en"] = Form("ar"),
                       annotate: bool = Form(False)):
    data = await photo.read()
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "The photo is larger than 12 MB.")
    try:
        return await run_in_threadpool(check_photo, data, room_id, _user(request), language, annotate)
    except SnapError as e:
        raise HTTPException(503 if "reader" in str(e) else 400, str(e))


@router.get("/rooms/{room_id}/questions", response_model=list[LoggedQuestion])
def my_questions(room_id: str, request: Request, limit: int = Query(50, ge=1, le=500)):
    user = _user(request)
    if not user:
        raise HTTPException(401, "Sign in first: send the Authorization token or X-User-Id.")
    return [{"question_id": r["id"], "ts": r["ts"], "question": r["question"], "topic_id": r["topic_id"] or "other", "level": r["level"] or 1,
             "grounded": bool(r["grounded"]), "study_file_id": r["study_file_id"]} for r in study_store.list_questions(room_id, user, limit)]


@router.get("/rooms/{room_id}/instructor/questions", response_model=list[InstructorQuestion], dependencies=[Depends(require_instructor)])
def instructor_questions(room_id: str, format: Literal["json", "csv"] = "json"):
    items = anonymous.rows(room_id)
    if format == "csv":
        return PlainTextResponse(anonymous.to_csv(items), media_type="text/csv")
    return items


@router.post("/rooms/{room_id}/instructor/publish", response_model=PublishResult, dependencies=[Depends(require_instructor)])
def instructor_publish(room_id: str):
    return anonymous.publish(room_id)


@router.post("/rooms/{room_id}/tutor/act", response_model=TutorActResponse)
def tutor_act(room_id: str, body: TutorActRequest, request: Request):
    """The tutor agent behind M1's Copilot: one entry for Ask (Level 1 / 2), quiz, summary, flashcards, study files, rewrite, hints and handwriting."""
    p = body.model_dump(exclude={"action", "text", "language"}, exclude_none=True)
    return act(room_id, body.action, body.text, _user(request), body.language, **p)


@router.get("/rooms/{room_id}/instructor/frequent", response_model=list[FrequentQuestion], dependencies=[Depends(require_instructor)])
def instructor_frequent(room_id: str, min_count: int = Query(anonymous.FREQUENT_MIN, ge=2, le=100)):
    """What students ask again and again (same topic, nearly the same words), most asked first. No names, no ids."""
    return anonymous.frequent(room_id, min_count)

"""M3 tutor routes. The shapes are the agreed ones in contracts/openapi.yaml (TutorRequest, TutorAnswer, CheckWorkRequest, HandwritingCheck)."""
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from rafeeq.platform import access

from .answer import tutor_answer
from .check_work import check_work
from .events import current_user_id, send_question_event

router = APIRouter(tags=["M3 tutor"])


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TutorRequest(_Strict):
    room_id: str
    question: str = Field(min_length=1)
    language: Literal["ar", "en"] | None = None


class Citation(_Strict):
    file_name: str
    page: int = Field(ge=1)


class TutorAnswer(_Strict):
    answer: str
    citations: list[Citation]
    grounded: bool
    steps: list[str]


class CheckWorkRequest(_Strict):
    room_id: str
    file_id: str
    text: str


class Box(_Strict):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(ge=0, le=1)
    height: float = Field(ge=0, le=1)


class WorkStep(_Strict):
    text: str
    correct: bool


class HandwritingCheck(_Strict):
    boxes: list[Box]
    steps: list[WorkStep]
    hint: str


@router.post("/tutor/answer", response_model=TutorAnswer)
def answer(body: TutorRequest, request: Request, background: BackgroundTasks, authorization: str | None = Header(None)):
    access.room_member(authorization, body.room_id)
    res = tutor_answer(body.room_id, body.question, body.language)
    topic = res["chunks"][0].get("topic_id") if res["chunks"] else None
    background.add_task(send_question_event, current_user_id(request.headers), body.room_id, topic, body.question)
    return {k: res[k] for k in ("answer", "citations", "grounded", "steps")}


@router.post("/tutor/check-work", response_model=HandwritingCheck)
def check(body: CheckWorkRequest, authorization: str | None = Header(None)):
    access.room_member(authorization, body.room_id)
    return check_work(body.room_id, body.file_id, body.text)

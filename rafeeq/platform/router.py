"""M1 platform routes. Thin HTTP layer: all logic lives in rafeeq/platform/service.py.

Identity: every route reads `Authorization: Bearer <token>` (the token from /auth/login or
/auth/register). Calls WITHOUT a token keep the old behaviour and return the contract
sample, so existing sample-based tests and teammates' stubs keep working.
"""
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, File, Form, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from rafeeq.copilot.supervisor import handle_message
from rafeeq.platform import service
from rafeeq.samples import load_sample
from rafeeq.snap.reader import NotAnImage, ReaderUnavailable, read_snap
from rafeeq.teacher import class_report

router = APIRouter(tags=["M1 platform"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str
    password: str


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    email: str
    password: str
    role: Literal["student", "teacher"]
    join_code: str | None = None


class JoinRoomRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str


class CreateRoomRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class CopilotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room_id: str
    text: str
    file_id: str | None = None
    mode: Literal["auto", "ask", "check_work"] = "auto"


def _store() -> service.PlatformStore:
    return service.get_store()


def _run(fn, *args):
    try:
        return fn(*args)
    except service.PlatformError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None


def _current_user(authorization: str | None) -> dict | None:
    """None when no Authorization header was sent (legacy sample mode).
    401 when a header is sent but is not a known Bearer token."""
    if not authorization:
        return None
    token = service.token_from_header(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Use: Authorization: Bearer <token>")
    user = _store().user_for_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


def _required_user(authorization: str | None) -> dict:
    user = _current_user(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Log in first")
    return user


# ---------------------------------------------------------------- accounts

@router.post("/auth/login")
def login(request: LoginRequest):
    return _run(_store().login, request.email, request.password)


@router.post("/auth/register")
def register(request: RegisterRequest):
    return _run(_store().register, request.name, request.email, request.password, request.role, request.join_code)


# ---------------------------------------------------------------- rooms

@router.get("/rooms")
def list_my_rooms(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user is None:
        return load_sample("my_rooms")
    return _store().list_rooms(user)


@router.post("/rooms")
def create_room(request: CreateRoomRequest, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user is None:
        return load_sample("create_room")
    return _run(_store().create_room, user, request.name)


@router.put("/rooms/{room_id}")
def rename_room(room_id: str, request: CreateRoomRequest, authorization: str | None = Header(default=None)):
    return _run(_store().rename_room, _required_user(authorization), room_id, request.name)


@router.post("/rooms/join")
def join_room(request: JoinRoomRequest, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user is None:
        return load_sample("join_room")
    return _run(_store().join_room, user, request.code)


@router.get("/rooms/{room_id}/join-code")
def get_join_code(room_id: str, authorization: str | None = Header(default=None)):
    return _run(_store().join_code, _required_user(authorization), room_id)


@router.post("/rooms/{room_id}/files", status_code=202)
async def upload_file(room_id: str, file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user is None:
        return load_sample("upload_file")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The file is larger than 25 MB")
    return _run(_store().add_file, user, room_id, file.filename or "file", data)


@router.get("/rooms/{room_id}/files")
def list_room_files(room_id: str, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user is None:
        return load_sample("room_files")
    return _run(_store().list_files, user, room_id)


@router.get("/rooms/{room_id}/files/{file_id}/download")
def download_room_file(room_id: str, file_id: str, authorization: str | None = Header(default=None)):
    name, data = _run(_store().read_file, _required_user(authorization), room_id, file_id)
    return Response(data, media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name, safe='')}"})


@router.delete("/rooms/{room_id}/files/{file_id}", status_code=204)
def delete_room_file(room_id: str, file_id: str, authorization: str | None = Header(default=None)):
    _run(_store().delete_file, _required_user(authorization), room_id, file_id)
    return Response(status_code=204)


@router.get("/rooms/{room_id}/students")
def list_room_students(room_id: str, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user is None:
        return load_sample("room_students")
    result = _run(_store().list_students, user, room_id)
    # M5: fill each student's level from their quiz results (null stays null without data)
    levels = class_report.student_levels(class_report.get_store(), room_id)
    for s in result["students"]:
        if s["user_id"] in levels:
            s["mastery"], s["status"] = levels[s["user_id"]]
    return result


# ---------------------------------------------------------------- snap and copilot

@router.post("/snap")
async def snap_and_ask(photo: UploadFile = File(...), room_id: str | None = Form(None),
                       authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if user is not None and room_id:
        _run(_store().room_for, user, room_id)
    data = await photo.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The photo is larger than 25 MB")
    try:
        # OpenCV and the vision call are slow and blocking: keep them off the event loop
        return await run_in_threadpool(read_snap, data)
    except NotAnImage:
        raise HTTPException(status_code=422, detail="The upload is not an image") from None
    except ReaderUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


@router.post("/copilot/message")
def copilot_message(request: CopilotRequest, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    user_id = None
    if user is not None:
        _run(_store().room_for, user, request.room_id)
        user_id = user["user_id"]
    out = handle_message(request.room_id, request.text, request.file_id, request.mode, user_id)
    if user is not None and user["role"] == "student":
        _store().record_question(user_id, request.room_id)
        # M5: keep the question for the lecturer's Class insights, under the topic of the pages the
        # answer cited (M2) when there are any; otherwise class_report groups it later.
        topic_id = None
        if out["route"] == "ask" and out["payload"].get("citations"):
            from rafeeq.copilot import clients

            topic_id = clients.optional("topic_of", request.room_id, out["payload"]["citations"])
        class_report.record_question(request.room_id, user_id, request.text, topic_id=topic_id)
    return out

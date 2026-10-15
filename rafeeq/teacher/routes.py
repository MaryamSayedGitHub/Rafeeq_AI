"""Member 5: the teacher part of Rafeeq.

JSON API (the M5 section of contracts/openapi.yaml):
    GET  /teacher/profile?teacher_id=...           -> TeacherProfile
    PUT  /teacher/profile                          -> TeacherProfile
    GET  /teacher/sessions?teacher_id=...          -> TeacherSessions (finished sessions)
    POST /teacher/sessions                         -> LecturerSession (runs until the draft is ready)
    GET  /teacher/sessions/{session_id}            -> LecturerSession
    POST /teacher/sessions/{session_id}/decision   -> LecturerSession (approve -> files, revise -> new draft)
    GET  /teacher/sessions/{session_id}/files/{name}
    GET  /teacher/rooms/{room_id}/insights         -> Insights     (real: questions + M4 quiz mistakes)
    GET  /teacher/rooms/{room_id}/report           -> ClassReport  (real: M4 quiz results per topic)

Pages (Jinja2 + HTMX, Arabic, right-to-left):
    GET  /teacher                       dashboard: profile, new session form, past sessions, class
    GET  /teacher/profile/edit          profile form
    POST /teacher/ui/profile            save the form, remember the teacher in a cookie
    POST /teacher/ui/sessions           form -> review panel (HTMX fragment)
    POST /teacher/ui/sessions/{id}/decision
    GET  /teacher/ui/sessions/{id}      the review panel as a full page (after a reload)
"""

import hashlib
import logging
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, unquote

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rafeeq.samples import load_sample
from rafeeq.teacher.lecturer.controller import RevisionLimitReached, SessionNotFound, SessionNotWaiting
from rafeeq.teacher.lecturer.long_term_store import get_profile, list_sessions, load_memory, put_profile
from rafeeq.teacher import assets as asset_lib
from rafeeq.teacher import class_report, inbox, parent_report
from rafeeq.teacher.lecturer import video_discovery, video_gen, youtube_discovery
from rafeeq.teacher.lecturer.schemas import ApproveRequest as Decision
from rafeeq.teacher.lecturer.schemas import SessionRequest
from rafeeq.teacher.lecturer.settings import AVATAR_DIR, MAX_REVISIONS, OUTPUT_DIR
from rafeeq.teacher.profile import LEVEL_AR, TeacherProfile, TeacherProfileUpdate, load_profile, make_teacher_id, save_avatar, save_profile
from rafeeq.teacher.service import AssistantNotConfigured, TeacherService
from rafeeq.web.nav import NAV

log = logging.getLogger(__name__)

router = APIRouter(tags=["M5 teacher"])
templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "web" / "templates")
templates.env.globals["LEVEL_AR"] = LEVEL_AR
# Embeds the YouTube results as JSON for the player's <script> (escapes "</" so a
# transcript chunk can never accidentally close the surrounding <script> tag).
templates.env.filters["tojson"] = lambda obj: __import__("json").dumps(obj, ensure_ascii=False).replace("</", "<\\/")

COOKIE = "rafeeq_teacher"
COOKIE_MAX_AGE = 60 * 60 * 24 * 180


def service(request: Request) -> TeacherService:
    return request.app.state.teacher


Service = Annotated[TeacherService, Depends(service)]


# A cookie may only hold ASCII, but a teacher_id can be Arabic ("سارة-أحمد"), so it is
# percent-encoded in the cookie and decoded when read.
def current_teacher_id(request: Request) -> str:
    return unquote(request.cookies.get(COOKIE, ""))


def remember_teacher(response, teacher_id: str, **options) -> None:
    response.set_cookie(COOKIE, quote(teacher_id, safe=""), **options)


class TeacherSessions(BaseModel):
    sessions: list[dict]



# ----------------------------------------------------------------- who is calling
def teacher_for(authorization: str | None, claimed_id: str = "") -> str:
    """The teacher this request acts for.

    Same convention as M1's router: no Authorization header = legacy mode (the old /teacher
    cookie pages and sample tests), so the id the caller names is used as before. When a Bearer
    token IS sent it must be a real teacher, and it decides who is acting: naming a different
    teacher_id is refused (403), so one teacher cannot read or change another's profile, avatar,
    videos or sessions through /app.
    """
    if not authorization:
        return claimed_id
    from rafeeq.platform import service as platform

    token = platform.token_from_header(authorization)
    user = platform.get_store().user_for_token(token) if token else None
    if user is None:
        raise HTTPException(401, "Invalid or expired token")
    if user.get("role") != "teacher":
        raise HTTPException(403, "Teachers only")
    if claimed_id and claimed_id != user["user_id"]:
        raise HTTPException(403, "You can only act for your own teacher account")
    return user["user_id"]


Auth = Annotated[str | None, Header()]

# ----------------------------------------------------------------- shared logic
def _profile(svc: TeacherService, teacher_id: str) -> TeacherProfile | None:
    return load_profile(svc.store, teacher_id, load_memory(svc.store, teacher_id))


def _past_sessions(svc: TeacherService, teacher_id: str) -> list[dict]:
    return [
        {
            "session_id": s["session_id"],
            "topic": s.get("topic") or "",
            "course_name": s.get("course_name") or "",
            "room_id": s.get("room_id") or "",
            "created_at": s.get("created_at") or "",
            "files": [
                {"name": n, "url": f"/teacher/sessions/{s['session_id']}/files/{n}"}
                for n in s.get("files", [])
            ],
        }
        for s in list_sessions(svc.store, teacher_id)
    ]


def _start(svc: TeacherService, req: SessionRequest) -> dict:
    try:
        return svc.controller.start(req.model_dump())
    except AssistantNotConfigured as e:
        raise HTTPException(503, f"The lecturer assistant has no LLM configured: {e}") from None


def _decide(svc: TeacherService, session_id: str, decision: Decision) -> dict:
    try:
        return svc.controller.decide(session_id, decision.decision, decision.feedback.strip())
    except SessionNotFound:
        raise HTTPException(404, "Session not found") from None
    except SessionNotWaiting:
        raise HTTPException(409, "This session is not waiting for a decision (it is already finished).") from None
    except RevisionLimitReached:
        raise HTTPException(409, f"The draft was already revised {MAX_REVISIONS} times. Approve it or start a new session.") from None
    except AssistantNotConfigured as e:
        raise HTTPException(503, str(e)) from None


def _get(svc: TeacherService, session_id: str) -> dict:
    try:
        return svc.controller.get_state(session_id)
    except SessionNotFound:
        raise HTTPException(404, "Session not found") from None
    except AssistantNotConfigured as e:
        raise HTTPException(503, str(e)) from None


def _course_rooms() -> list[dict]:
    """Rooms the teacher can link a session to: the real class rooms (M1's platform store),
    newest last. /teacher has its own sign-in, so every course room is listed."""
    try:
        from rafeeq.platform.service import get_store

        rooms = sorted(get_store().db["rooms"].values(), key=lambda r: r.get("created_at", ""))
        return [{"room_id": r["room_id"], "name": r["name"], "kind": "course"} for r in rooms]
    except Exception as e:
        log.warning("teacher: could not list class rooms: %s", e)
        return []


# ===================================================================== JSON API
@router.get("/teacher/profile", response_model=TeacherProfile)
def get_profile_api(svc: Service, authorization: Auth = None, teacher_id: str = Query(min_length=1)):
    teacher_id = teacher_for(authorization, teacher_id)
    profile = _profile(svc, teacher_id)
    if profile is None:
        raise HTTPException(404, "No profile for this teacher yet")
    return profile


@router.put("/teacher/profile", response_model=TeacherProfile)
def put_profile_api(svc: Service, body: TeacherProfileUpdate, authorization: Auth = None):
    if authorization:
        body.teacher_id = teacher_for(authorization, body.teacher_id)
    return _profile(svc, save_profile(svc.store, body))


@router.get("/teacher/sessions", response_model=TeacherSessions)
def list_sessions_api(svc: Service, authorization: Auth = None, teacher_id: str = Query(min_length=1)):
    teacher_id = teacher_for(authorization, teacher_id)
    return {"sessions": _past_sessions(svc, teacher_id)}


@router.get("/teacher/videos/search")
def search_videos_api(
    q: str = Query(min_length=2, max_length=300),
    top_k: int = Query(5, ge=1, le=10),
):
    """Real YouTube videos, found and ranked by meaning, opening at the exact timestamp
    that answers the question (Madrasa-spec Feature 2). Empty list (not an error) when
    YOUTUBE_API_KEY/Postgres/Gemini are unavailable."""
    return {"results": youtube_discovery.search(q, top_k=top_k)}


MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _avatar_file(teacher_id: str) -> Path | None:
    """The teacher's saved presenter image, if any (any extension)."""
    if not teacher_id or "/" in teacher_id or "\\" in teacher_id or teacher_id.startswith("."):
        return None
    return next(iter(sorted(AVATAR_DIR.glob(f"{teacher_id}.*"))), None)


async def _read_image(image: UploadFile | None) -> tuple[bytes | None, str]:
    if image is None or not image.filename:
        return None, "image.jpg"
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "The image is larger than 8 MB")
    return data, image.filename


def _video_key(prompt: str, duration: int, audio: bool, image: bytes | None) -> str:
    """Same teacher + same scene + same settings = same key, so a double click or a retry never pays twice."""
    digest = hashlib.sha1(f"{prompt.strip()}|{duration}|{audio}".encode() + (image or b"")).hexdigest()[:16]
    return f"video:{digest}"


@router.post("/teacher/videos/generate")
async def generate_video_api(
    svc: Service,
    authorization: Auth = None,
    teacher_id: str = Form(""),
    prompt: str = Form(..., min_length=3, max_length=800),
    duration: int = Form(5),
    audio: bool = Form(False),
    image: UploadFile | None = File(None),
):
    """Start an animated video (Lab 5). Without an image, the teacher's saved presenter image is the
    first frame if there is one, otherwise it is text-to-video. Paid and async: poll the status route.

    With a teacher token the job is also saved in the library (asset_id in the answer): a repeat of the
    same request returns the same job instead of paying twice, and the finished clip is kept on our disk."""
    teacher_id = teacher_for(authorization, teacher_id)
    image_bytes, image_name = await _read_image(image)
    used_avatar = False
    if image_bytes is None and (avatar := _avatar_file(teacher_id)) is not None:
        image_bytes, image_name, used_avatar = avatar.read_bytes(), avatar.name, True

    duration = max(2, min(duration, video_gen.MAX_DURATION_SECONDS))
    asset, key = None, ""
    if authorization:
        key = _video_key(prompt, duration, audio, image_bytes)
        asset = asset_lib.find_by_key(svc.store, teacher_id, key)
        if asset and asset["status"] in ("ready", "pending") and not asset_lib.is_stale_pending(asset):
            return {"task_id": asset["settings"].get("task_id", ""), "model": asset.get("model", ""),
                    "used_avatar": used_avatar, "asset_id": asset["asset_id"]}

    result = video_gen.submit_task(prompt.strip(), image_bytes, image_name, duration=duration, audio=audio)
    if not result["ok"]:
        status = 503 if "DASHSCOPE_API_KEY" in result["error"] else 502
        raise HTTPException(status, f"Video generation could not start: {result['error']}")

    answer = {"task_id": result["task_id"], "model": result["model"], "used_avatar": used_avatar}
    if authorization:
        used = {"task_id": result["task_id"], "prompt": prompt.strip(), "duration": duration, "audio": audio,
                "used_avatar": used_avatar}
        if asset:  # a failed or lost job: reuse the record for the new attempt
            asset = asset_lib.update_asset(svc.store, teacher_id, asset["asset_id"], status="pending",
                                           error="", settings=used)
        else:
            asset = asset_lib.register_asset(
                svc.store, teacher_id, kind="generated_video", title=prompt.strip()[:80], origin="ai_generated",
                provider="qwencloud", model=result["model"], settings_used=used, status="pending",
                idempotency_key=key)
        answer["asset_id"] = asset["asset_id"]
    return answer


@router.get("/teacher/videos/generate/{task_id}")
def video_status_api(svc: Service, task_id: str, authorization: Auth = None):
    """status: PENDING / RUNNING / SUCCEEDED / FAILED / ERROR. video_url only when SUCCEEDED (valid 24 h).
    With a teacher token: asset_id, and file_url once the clip is saved in the library."""
    owner = teacher_for(authorization, "") if authorization else ""
    asset = asset_lib.find_by_task(svc.store, owner, task_id) if owner else None

    if asset and asset["status"] == "ready" and asset_lib.asset_file(asset):
        return {"status": "SUCCEEDED", "video_url": None, "error": None,
                "asset_id": asset["asset_id"], "file_url": f"/teacher/assets/{asset['asset_id']}/file"}

    result = video_gen.poll_task(task_id)
    if asset is None:
        return result
    result = {**result, "asset_id": asset["asset_id"]}

    if result["status"] == "SUCCEEDED" and result["video_url"]:
        try:
            root, ref = asset_lib.save_file(owner, asset["asset_id"], video_gen.download_video(result["video_url"]),
                                            "video.mp4")
            asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="ready", storage_root=root,
                                   storage_ref=ref, error="")
            result["file_url"] = f"/teacher/assets/{asset['asset_id']}/file"
        except Exception as e:  # keep the provider link (24 h); the library record says what happened
            log.warning("could not save the generated video: %s: %s", type(e).__name__, str(e)[:200])
            asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="failed",
                                   error=f"The clip could not be saved: {str(e)[:150]}")
    elif result["status"] in ("FAILED", "CANCELED"):
        asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="failed", error=(result["error"] or "")[:300])
    return result


@router.post("/teacher/avatar")
async def upload_avatar_api(svc: Service, authorization: Auth = None, teacher_id: str = Form(..., min_length=1), image: UploadFile = File(...)):
    """Save the teacher's presenter image (used as the first frame of their animated videos)."""
    teacher_id = teacher_for(authorization, teacher_id)
    data, name = await _read_image(image)
    if data is None:
        raise HTTPException(422, "No image")
    try:
        filename = save_avatar(teacher_id, data, name)
    except ValueError:
        raise HTTPException(422, "Invalid teacher_id") from None
    saved = get_profile(svc.store, teacher_id)
    if saved:
        put_profile(svc.store, teacher_id, {**saved, "avatar_filename": filename})
    return {"avatar_filename": filename}


# ===================================================================== asset library
# The teacher's saved/generated files and videos (rafeeq/teacher/assets.py). These routes ALWAYS
# need a teacher's Bearer token: the owner is the token, never a client-supplied id.
class AssetCreate(BaseModel):
    """Save an external video moment (e.g. a YouTube result) to the library."""

    kind: Literal["source_video"] = "source_video"
    title: str = Field(min_length=1, max_length=200)
    source_url: str = Field(min_length=8, max_length=500, pattern=r"^https?://")
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, ge=0)
    room_id: str = Field(default="", max_length=60)
    session_id: str = Field(default="", max_length=80)
    slide_index: int | None = Field(default=None, ge=0, le=500)


def _owner(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(401, "Sign in as a teacher")
    return teacher_for(authorization, "")


def asset_view(a: dict) -> dict:
    """What the API returns: no server paths, and a file_url only when a file really exists."""
    return {
        "asset_id": a["asset_id"], "kind": a["kind"], "title": a["title"], "origin": a["origin"],
        "status": a["status"], "error": a.get("error", ""),
        "source_url": a.get("source_url", ""), "start_sec": a.get("start_sec"), "end_sec": a.get("end_sec"),
        "provider": a.get("provider", ""), "model": a.get("model", ""), "settings": a.get("settings") or {},
        "room_id": a.get("room_id", ""), "session_id": a.get("session_id", ""), "slide_index": a.get("slide_index"),
        "parent_asset_id": a.get("parent_asset_id", ""), "version": a.get("version", 1),
        "file_url": f"/teacher/assets/{a['asset_id']}/file" if asset_lib.asset_file(a) else None,
        "created_at": a.get("created_at", ""), "updated_at": a.get("updated_at", ""),
    }


@router.get("/teacher/assets")
def list_assets_api(svc: Service, authorization: Auth = None, kind: str = "", room_id: str = "",
                    session_id: str = "", status: str = ""):
    owner = _owner(authorization)
    asset_lib.backfill_sessions(svc.store, owner)  # sessions finished before the library existed (idempotent)
    found = asset_lib.list_assets(svc.store, owner, kind=kind, room_id=room_id, session_id=session_id, status=status)
    return {"assets": [asset_view(a) for a in found]}


@router.post("/teacher/assets", status_code=201)
def create_asset_api(svc: Service, body: AssetCreate, authorization: Auth = None):
    owner = _owner(authorization)
    if body.start_sec is not None and body.end_sec is not None and body.end_sec < body.start_sec:
        raise HTTPException(422, "end_sec must not be before start_sec")
    key = f"external:{body.source_url}:{body.start_sec}:{body.end_sec}"
    try:
        asset = asset_lib.register_asset(
            svc.store, owner, kind=body.kind, title=body.title, origin="external", source_url=body.source_url,
            start_sec=body.start_sec, end_sec=body.end_sec, room_id=body.room_id, session_id=body.session_id,
            slide_index=body.slide_index, idempotency_key=key)
    except asset_lib.AssetError as e:
        raise HTTPException(422, str(e)) from None
    return asset_view(asset)


@router.get("/teacher/assets/{asset_id}")
def get_asset_api(svc: Service, asset_id: str, authorization: Auth = None):
    asset = asset_lib.get_asset(svc.store, _owner(authorization), asset_id)
    if asset is None:
        raise HTTPException(404, "Asset not found")
    return asset_view(asset)


@router.get("/teacher/assets/{asset_id}/file")
def get_asset_file_api(svc: Service, asset_id: str, authorization: Auth = None):
    asset = asset_lib.get_asset(svc.store, _owner(authorization), asset_id)
    if asset is None:
        raise HTTPException(404, "Asset not found")
    path = asset_lib.asset_file(asset)
    if path is None:
        raise HTTPException(404, "This asset has no file on the server")
    return FileResponse(path, filename=path.name)


@router.post("/teacher/sessions")
def start_session_api(svc: Service, body: SessionRequest, authorization: Auth = None):
    if authorization:
        body.lecturer_id = teacher_for(authorization, body.lecturer_id)
    return _start(svc, body)


@router.get("/teacher/sessions/{session_id}")
def get_session_api(svc: Service, session_id: str):
    return _get(svc, session_id)


@router.post("/teacher/sessions/{session_id}/decision")
def decide_api(svc: Service, session_id: str, body: Decision):
    return _decide(svc, session_id, body)


@router.get("/teacher/sessions/{session_id}/files/{name}")
def download(svc: Service, session_id: str, name: str):
    # Only names the session itself recorded can be downloaded (no path tricks).
    folder, files = _session_files(svc, session_id)
    if name not in files:
        raise HTTPException(404, "File not found")
    path = OUTPUT_DIR / folder / name
    if not path.is_file():
        raise HTTPException(404, "File is missing on the server")
    return FileResponse(path, filename=name)


def _session_files(svc: TeacherService, session_id: str) -> tuple[str, list[str]]:
    try:
        return svc.controller.output_folder(session_id)
    except (SessionNotFound, AssistantNotConfigured):
        pass
    # Fallback: sessions finished before a restart without Postgres are only in the store.
    for teacher_ns in svc.store.list_namespaces(prefix=("lecturer",), max_depth=2):
        for item in svc.store.search((*teacher_ns, "sessions"), limit=200):
            if item.key == session_id:
                return session_id, item.value.get("files", [])
    raise HTTPException(404, "Session not found")


@router.get("/teacher/rooms/{room_id}/insights")
def insights(svc: Service, room_id: str):
    """Real data: the room's student questions grouped by topic + repeated quiz mistakes (class_report.py)."""
    return class_report.build_insights(svc.store, room_id)


@router.get("/teacher/rooms/{room_id}/report")
def report(svc: Service, room_id: str):
    """Real data: class mastery per topic and students at risk, from recorded quiz results (class_report.py)."""
    return class_report.build_report(svc.store, room_id)


# ------------------------------------------------------------ parent report
def _room_user(authorization: str | None, room_id: str) -> tuple[dict, dict]:
    """(user, room) for a logged-in member or the teacher of the room (M1's Bearer token)."""
    from rafeeq.platform import service as platform

    token = platform.token_from_header(authorization) if authorization else None
    user = platform.get_store().user_for_token(token) if token else None
    if user is None:
        raise HTTPException(401, "Log in first")
    room = platform.get_store().db["rooms"].get(room_id)
    if room is None:
        raise HTTPException(404, "Room not found")
    member = user["user_id"] in platform.get_store().db["members"].get(room_id, [])
    if room["teacher_id"] != user["user_id"] and not member:
        raise HTTPException(403, "Not a member of this room")
    return user, room


def _room_teacher(authorization: str | None, room_id: str) -> dict:
    """The caller must be this room's teacher. The older teacher routes predate the platform login;
    these carry students' data, so they check."""
    user, room = _room_user(authorization, room_id)
    if room["teacher_id"] != user["user_id"]:
        raise HTTPException(403, "Only this room's teacher can do this")
    return user


@router.get("/teacher/rooms/{room_id}/students/{user_id}/parent-report")
def parent_report_api(svc: Service, room_id: str, user_id: str, authorization: Auth = None):
    from rafeeq.platform import service as platform

    _room_teacher(authorization, room_id)
    if user_id not in platform.get_store().db["members"].get(room_id, []):
        raise HTTPException(404, "This student is not in the room")
    report = parent_report.build(svc.store, room_id, user_id)
    return {**report, "share_token": parent_report.make_share_token(room_id, user_id)}


@router.get("/parent-report/{token}")
def parent_report_shared(svc: Service, token: str):
    parsed = parent_report.read_share_token(token)
    if parsed is None:
        raise HTTPException(404, "This link is invalid or has expired")
    return parent_report.build(svc.store, *parsed)


# ------------------------------------------------------------ inbox (questions the tutor could not answer)
class InboxAnswerBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1, max_length=4000)


class AssistantBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room_id: str
    request: str = Field(min_length=1, max_length=2000)


@router.get("/rooms/{room_id}/inbox")
def inbox_list(room_id: str, authorization: Auth = None):
    """The room's teacher sees every question; a student sees only their own (with the teacher's answers)."""
    user, room = _room_user(authorization, room_id)
    mine = None if room["teacher_id"] == user["user_id"] else user["user_id"]
    return {"questions": inbox.list_questions(room_id, user_id=mine)}


@router.post("/rooms/{room_id}/inbox/{question_id}/answer")
def inbox_answer(room_id: str, question_id: str, body: InboxAnswerBody, authorization: Auth = None):
    _room_teacher(authorization, room_id)
    saved = inbox.answer(room_id, question_id, body.answer)
    if saved is None:
        raise HTTPException(404, "Question not found")
    return {**saved, "student_name": inbox._student_names(room_id).get(saved.get("user_id") or "", "")}


@router.post("/rooms/{room_id}/inbox/{question_id}/draft")
def inbox_draft(room_id: str, question_id: str, authorization: Auth = None):
    _room_teacher(authorization, room_id)
    question = next((q for q in inbox.list_questions(room_id) if q["question_id"] == question_id), None)
    if question is None:
        raise HTTPException(404, "Question not found")
    reply = inbox.draft(room_id, question["text"])
    if reply is None:
        raise HTTPException(503, "No model is configured for drafts (LLM_API_KEY)")
    return {"draft_reply": reply, "status": "waiting_for_approval"}


@router.post("/teacher/assistant")
def teacher_assistant(body: AssistantBody, authorization: Auth = None):
    """A draft for the teacher's request (a message to the class, a summary...), from the room's class data."""
    _room_teacher(authorization, body.room_id)
    reply = inbox.assistant(body.room_id, body.request)
    if reply is None:
        raise HTTPException(503, "No model is configured for drafts (LLM_API_KEY)")
    return {"draft_reply": reply, "status": "waiting_for_approval"}


# ======================================================================== pages
def _page(request: Request, name: str, **context) -> HTMLResponse:
    return templates.TemplateResponse(request, name, {"nav": NAV, "active": "teacher", **context})


def _fragment(request: Request, name: str, **context) -> HTMLResponse:
    return templates.TemplateResponse(request, name, context)


def _error(request: Request, message: str) -> HTMLResponse:
    # HTMX swaps only 2xx responses by default, so errors are sent as 200 with an error box.
    return _fragment(request, "teacher/_error.html", message=message)


def _class_view(store, room_id: str | None) -> dict:
    """The dashboard's class panel for one room, from real data (class_report.py)."""
    if not room_id:
        return {"insights": {"top_questions": [], "common_mistakes": []},
                "report": {"topics": [], "students_at_risk": []}, "topic_names": {}, "class_room": None}
    return {"insights": class_report.build_insights(store, room_id),
            "report": class_report.build_report(store, room_id),
            "topic_names": class_report.room_topic_names(room_id), "class_room": room_id}


@router.get("/teacher", response_class=HTMLResponse)
def dashboard(request: Request, svc: Service):
    teacher_id = current_teacher_id(request)
    profile = _profile(svc, teacher_id) if teacher_id else None
    if profile is None:
        return _page(request, "teacher/profile_form.html", profile=None, first_time=True,
                     persistent=svc.persistent)
    rooms = _course_rooms()
    wanted = request.query_params.get("room")
    room_id = wanted if any(r["room_id"] == wanted for r in rooms) else (rooms[-1]["room_id"] if rooms else None)
    return _page(
        request, "teacher/dashboard.html",
        profile=profile,
        sessions=_past_sessions(svc, teacher_id),
        rooms=rooms,
        persistent=svc.persistent,
        **_class_view(svc.store, room_id),
    )


@router.get("/teacher/profile/edit", response_class=HTMLResponse)
def edit_profile(request: Request, svc: Service):
    teacher_id = current_teacher_id(request)
    profile = _profile(svc, teacher_id) if teacher_id else None
    return _page(request, "teacher/profile_form.html", profile=profile, first_time=profile is None,
                 persistent=svc.persistent)


@router.post("/teacher/ui/profile")
async def save_profile_form(
    request: Request,
    svc: Service,
    name: str = Form(...),
    title: str = Form(""),
    subjects: str = Form(""),
    bio: str = Form(""),
    language: str = Form("Arabic"),
    student_level: str = Form("intermediate"),
    programming_language: str = Form("Python"),
    duration_minutes: int = Form(60),
    avatar: UploadFile | None = File(None),
):
    teacher_id = current_teacher_id(request)
    existing = _profile(svc, teacher_id) if teacher_id else None
    avatar_filename = existing.avatar_filename if existing else ""
    if avatar is not None and avatar.filename:
        avatar_filename = save_avatar(teacher_id or make_teacher_id(name), await avatar.read(), avatar.filename)

    try:
        update = TeacherProfileUpdate(
            teacher_id=teacher_id,
            name=name, title=title, subjects=subjects, bio=bio,
            avatar_filename=avatar_filename,
            defaults={"language": language, "student_level": student_level,
                      "programming_language": programming_language,
                      "duration_minutes": duration_minutes},
        )
    except ValidationError as e:
        return _page(request, "teacher/profile_form.html", profile=None, first_time=True,
                     persistent=svc.persistent, error=_first_error(e))
    teacher_id = save_profile(svc.store, update)
    response = RedirectResponse("/teacher", status_code=303)
    remember_teacher(response, teacher_id, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
    return response


@router.get("/teacher/avatar/{teacher_id}")
def teacher_avatar(teacher_id: str):
    path = _avatar_file(teacher_id)
    if path is None:
        raise HTTPException(404, "No avatar saved for this teacher")
    return FileResponse(path)


@router.post("/teacher/ui/sessions", response_class=HTMLResponse)
def start_session_form(
    request: Request,
    svc: Service,
    topic: str = Form(...),
    room_id: str = Form(""),
    course_name: str = Form(""),
    student_level: str = Form("intermediate"),
    duration_minutes: int = Form(60),
    language: str = Form("Arabic"),
    programming_language: str = Form("Python"),
    needs: list[str] = Form([]),
    num_quiz_questions: int = Form(5),
    learning_objectives: str = Form(""),
    notes: str = Form(""),
):
    teacher_id = current_teacher_id(request)
    profile = _profile(svc, teacher_id) if teacher_id else None
    if profile is None:
        return _error(request, "احفظ ملفك الشخصي أولًا.")
    if room_id and not course_name:
        course_name = next((r["name"] for r in _course_rooms() if r["room_id"] == room_id), "")
    try:
        req = SessionRequest(
            lecturer_id=teacher_id, lecturer_name=profile.display_name, room_id=room_id,
            topic=topic, course_name=course_name, student_level=student_level,
            duration_minutes=duration_minutes, language=language,
            programming_language=programming_language, needs=needs,
            num_quiz_questions=num_quiz_questions,
            learning_objectives=learning_objectives, notes=notes,
        )
    except ValidationError as e:
        return _error(request, _first_error(e))
    try:
        view = _start(svc, req)
    except HTTPException as e:
        return _error(request, e.detail)
    except Exception as e:  # LLM / tool failure: show it instead of a blank panel
        log.exception("teacher: generation failed")
        return _error(request, f"تعذّر تجهيز المسودة: {type(e).__name__}: {str(e)[:300]}")
    return _fragment(request, "teacher/_review.html", s=view)


@router.post("/teacher/ui/videos/search", response_class=HTMLResponse)
def search_videos_form(request: Request, svc: Service, q: str = Form(...)):
    if not current_teacher_id(request):
        return _error(request, "احفظ ملفك الشخصي أولًا.")
    results = youtube_discovery.search(q.strip(), top_k=5)
    return _fragment(request, "teacher/_video_results.html", query=q.strip(), results=results)


@router.post("/teacher/ui/videos/generate", response_class=HTMLResponse)
async def generate_video_form(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile | None = File(None),
    duration: int = Form(5),
    audio: bool = Form(False),
):
    teacher_id = current_teacher_id(request)
    if not teacher_id:
        return _error(request, "احفظ ملفك الشخصي أولًا.")

    image_bytes, image_filename = None, "image.jpg"
    if image is not None and image.filename:
        image_bytes = await image.read()
        image_filename = image.filename
        if len(image_bytes) > 8 * 1024 * 1024:  # 8MB: keep the request body reasonable
            return _error(request, "الصورة كبيرة جدًا (الحد الأقصى 8MB).")
    else:
        # No image uploaded for this specific video: fall back to the teacher's
        # saved character/avatar image (if any), so the same "presenter" shows
        # up across videos without re-uploading every time.
        avatar_file = _avatar_file(teacher_id)
        if avatar_file is not None:
            image_bytes = avatar_file.read_bytes()
            image_filename = avatar_file.name

    result = video_gen.submit_task(prompt.strip(), image_bytes, image_filename, duration=duration, audio=audio)
    if not result["ok"]:
        return _error(request, f"تعذّر بدء توليد الفيديو: {result['error']}")
    return _fragment(request, "teacher/_video_gen_status.html",
                     task_id=result["task_id"], model=result["model"], prompt=prompt.strip())


@router.get("/teacher/ui/videos/generate/{task_id}", response_class=HTMLResponse)
def generate_video_status(request: Request, task_id: str, prompt: str = Query("")):
    result = video_gen.poll_task(task_id)
    return _fragment(request, "teacher/_video_gen_status.html",
                     task_id=task_id, prompt=prompt, poll_result=result)


@router.post("/teacher/ui/sessions/{session_id}/decision", response_class=HTMLResponse)
def decide_form(
    request: Request,
    svc: Service,
    session_id: str,
    decision: str = Form(...),
    feedback: str = Form(""),
):
    if not _owns(request, svc, session_id):
        return _error(request, "الجلسة غير موجودة.")
    try:
        choice = Decision(decision=decision, feedback=feedback)
    except ValidationError:
        return _error(request, "اكتب ما تريد تعديله قبل طلب التعديل.")
    try:
        view = _decide(svc, session_id, choice)
    except HTTPException as e:
        return _error(request, e.detail)
    except Exception as e:
        log.exception("teacher: decision failed")
        return _error(request, f"حدث خطأ: {type(e).__name__}: {str(e)[:300]}")
    return _fragment(request, "teacher/_review.html", s=view)


@router.get("/teacher/ui/sessions/{session_id}", response_class=HTMLResponse)
def session_page(request: Request, svc: Service, session_id: str):
    if not _owns(request, svc, session_id):
        raise HTTPException(404, "Session not found")
    return _page(request, "teacher/session.html", s=_get(svc, session_id))


def _owns(request: Request, svc: TeacherService, session_id: str) -> bool:
    """A teacher only sees their own sessions in the pages."""
    teacher_id = current_teacher_id(request)
    try:
        brief = svc.controller.graph.get_state({"configurable": {"thread_id": session_id}}).values.get("brief", {})
    except Exception:
        return False
    return bool(teacher_id) and brief.get("lecturer_id") == teacher_id


def _first_error(e: ValidationError) -> str:
    err = e.errors()[0]
    field = ".".join(str(p) for p in err.get("loc", []))
    return f"{field}: {err.get('msg', 'invalid value')}" if field else err.get("msg", "invalid value")

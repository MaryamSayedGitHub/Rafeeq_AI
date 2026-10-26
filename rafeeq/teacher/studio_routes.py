"""M5 teacher studios: the JSON API behind the Presentation Studio, the Visual & Media Studio and
My Learning Platform (decks, themes, media, courses) and the student view of published courses.

Every route needs a teacher's Bearer token (the owner always comes from the token) except the two
student routes, which need any member of the room. Contract: contracts/openapi.yaml, "M5 studios".
"""

import hashlib
import json
import logging
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from rafeeq.teacher import assets as asset_lib
from rafeeq.teacher import courses, decks, media_tools, montage, story_video, themes, visuals
from rafeeq.teacher.lecturer import image_gen
from rafeeq.teacher.lecturer.long_term_store import get_profile, list_sessions
from rafeeq.teacher.routes import Auth, Service, _owner, asset_view

log = logging.getLogger(__name__)
router = APIRouter(tags=["M5 studios"])

MAX_THEME_FILE = themes.MAX_THEME_BYTES


def _422(e: Exception):
    return HTTPException(422, str(e))


# ===================================================================== themes
@router.get("/teacher/themes")
def list_themes_api(svc: Service, authorization: Auth = None):
    owner = _owner(authorization)
    custom = []
    for a in asset_lib.list_assets(svc.store, owner, kind="theme"):
        try:
            t = themes.normalize(a.get("settings") or {})
        except themes.ThemeError:
            continue
        custom.append({"theme_id": a["asset_id"], "asset_id": a["asset_id"], **t})
    return {"presets": themes.preset_list(), "custom": custom}


@router.post("/teacher/themes/extract", status_code=201)
async def extract_theme_api(svc: Service, authorization: Auth = None, file: UploadFile = File(...), name: str = Form("")):
    """Read the colours and font out of an existing .pptx / .potx and keep them as a library theme."""
    owner = _owner(authorization)
    data = await file.read(MAX_THEME_FILE + 1)
    try:
        theme = themes.extract_from_pptx(data, fallback_name=Path(file.filename or "").stem)
    except themes.ThemeError as e:
        raise _422(e) from None
    if name.strip():
        theme["name"] = name.strip()[:60]
    asset = asset_lib.register_asset(
        svc.store, owner, kind="theme", title=theme["name"], origin="upload", settings_used=theme,
        idempotency_key="theme:" + hashlib.sha1(data).hexdigest()[:20])
    return {**asset_view(asset), "theme": theme}


# ===================================================================== decks
class DeckCreate(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)


class DeckUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    slides: list[dict[str, Any]] | None = None
    theme_id: str | None = Field(default=None, max_length=60)


class SlideImage(BaseModel):
    asset_id: str = Field(default="", max_length=40)


class VisualRequest(BaseModel):
    prompt: str = Field(default="", max_length=600)


def deck_view(store, owner: str, d: dict) -> dict:
    theme = decks.resolve_theme(store, owner, d.get("theme_id", decks.DEFAULT_THEME_ID))
    slides = []
    for s in d["slides"]:
        pic = asset_lib.get_asset(store, owner, s["image_asset_id"]) if s.get("image_asset_id") else None
        slides.append({**s, "image_file_url": f"/teacher/assets/{pic['asset_id']}/file" if pic and asset_lib.asset_file(pic) else None})
    return {"deck_id": d["deck_id"], "title": d["title"], "course_name": d.get("course_name", ""),
            "room_id": d.get("room_id", ""), "session_id": d.get("session_id", ""),
            "theme_id": d.get("theme_id", decks.DEFAULT_THEME_ID), "theme_name": theme["name"], "theme": theme, "slides": slides,
            "slide_count": len(slides), "pptx_asset_id": d.get("pptx_asset_id", ""), "exports": d.get("exports", 0),
            "created_at": d.get("created_at", ""), "updated_at": d.get("updated_at", "")}


def _display_name(svc, owner: str) -> str:
    p = get_profile(svc.store, owner)
    return f"{p.get('title', '')} {p.get('name', '')}".strip() or owner


def _deck_from_session(svc, owner: str, session_id: str) -> dict | None:
    """The deck of one of THIS teacher's finished sessions; made from the saved draft if it does not exist yet."""
    existing = decks.find_by_session(svc.store, owner, session_id)
    if existing:
        return existing
    session = next((s for s in list_sessions(svc.store, owner) if s["session_id"] == session_id), None)
    if session is None:
        return None
    try:
        state = svc.controller.get_state(session_id)
    except Exception as e:  # session no longer in the checkpoint store, or no LLM configured
        log.info("deck for %s: draft not available (%s)", session_id, type(e).__name__)
        return None
    brief = {"topic": session.get("topic", ""), "course_name": session.get("course_name", ""),
             "lecturer_name": _display_name(svc, owner), "room_id": session.get("room_id", "")}
    return decks.create_from_draft(svc.store, owner, session_id, state.get("draft") or {}, brief)


@router.get("/teacher/decks")
def list_decks_api(svc: Service, authorization: Auth = None, room_id: str = ""):
    owner = _owner(authorization)
    for s in list_sessions(svc.store, owner):  # lessons approved before decks existed
        if any(str(f).lower().endswith(".pptx") for f in s.get("files", [])):
            if decks.find_by_session(svc.store, owner, s["session_id"]) is None:
                _deck_from_session(svc, owner, s["session_id"])
    return {"decks": [deck_view(svc.store, owner, d) for d in decks.list_decks(svc.store, owner, room_id)]}


@router.post("/teacher/decks", status_code=201)
def create_deck_api(svc: Service, body: DeckCreate, authorization: Auth = None):
    owner = _owner(authorization)
    deck = _deck_from_session(svc, owner, body.session_id)
    if deck is None:
        raise HTTPException(404, "No slides are available for this lesson")
    return deck_view(svc.store, owner, deck)


def _deck_or_404(svc, owner: str, deck_id: str) -> dict:
    deck = decks.get_deck(svc.store, owner, deck_id)
    if deck is None:
        raise HTTPException(404, "Deck not found")
    return deck


@router.get("/teacher/decks/{deck_id}")
def get_deck_api(svc: Service, deck_id: str, authorization: Auth = None):
    owner = _owner(authorization)
    return deck_view(svc.store, owner, _deck_or_404(svc, owner, deck_id))


@router.put("/teacher/decks/{deck_id}")
def update_deck_api(svc: Service, deck_id: str, body: DeckUpdate, authorization: Auth = None):
    owner = _owner(authorization)
    _deck_or_404(svc, owner, deck_id)
    try:
        deck = decks.update_deck(svc.store, owner, deck_id, title=body.title, slides=body.slides, theme_id=body.theme_id)
    except decks.DeckError as e:
        raise _422(e) from None
    return deck_view(svc.store, owner, deck)


@router.post("/teacher/decks/{deck_id}/export")
def export_deck_api(svc: Service, deck_id: str, authorization: Auth = None):
    """Build the .pptx with the deck's theme and pictures; it is saved in the library as the next version."""
    owner = _owner(authorization)
    _deck_or_404(svc, owner, deck_id)
    try:
        return asset_view(decks.export_deck(svc.store, owner, deck_id))
    except Exception as e:
        log.exception("deck export failed")
        raise HTTPException(500, f"The presentation could not be built: {type(e).__name__}") from None


@router.post("/teacher/decks/{deck_id}/slides/{index}/image")
def set_slide_image_api(svc: Service, deck_id: str, index: int, body: SlideImage, authorization: Auth = None):
    owner = _owner(authorization)
    _deck_or_404(svc, owner, deck_id)
    try:
        deck = decks.set_slide_image(svc.store, owner, deck_id, index, body.asset_id)
    except decks.DeckError as e:
        raise _422(e) from None
    return deck_view(svc.store, owner, deck)


@router.post("/teacher/decks/{deck_id}/slides/{index}/visual")
def make_slide_visual_api(svc: Service, deck_id: str, index: int, body: VisualRequest, authorization: Auth = None):
    """Two steps: the LLM writes an image prompt for the slide (or uses yours), then the diffusion model
    draws it. Paid and async: poll GET /teacher/visuals/{task_id}. The finished picture is saved in the
    library and attached to the slide."""
    owner = _owner(authorization)
    deck = _deck_or_404(svc, owner, deck_id)
    if not 0 <= index < len(deck["slides"]):
        raise HTTPException(404, "No such slide")
    slide = deck["slides"][index]
    prompt, used_llm = (body.prompt.strip(), False) if body.prompt.strip() else visuals.suggest_prompt(slide["title"], slide["bullets"])

    key = f"visual:{deck_id}:{index}:{hashlib.sha1(prompt.encode()).hexdigest()[:12]}"
    existing = asset_lib.find_by_key(svc.store, owner, key)
    if existing and existing["status"] in ("pending", "ready") and not asset_lib.is_stale_pending(existing):
        return {"task_id": existing["settings"].get("task_id", ""), "asset_id": existing["asset_id"], "prompt": prompt,
                "used_llm": used_llm}
    result = image_gen.submit_task(prompt)
    if not result["ok"]:
        raise HTTPException(503 if "DASHSCOPE_API_KEY" in result["error"] else 502,
                            f"Image generation could not start: {result['error']}")
    used = {"task_id": result["task_id"], "prompt": prompt, "deck_id": deck_id, "slide_index": index, "used_llm": used_llm}
    if existing:
        asset = asset_lib.update_asset(svc.store, owner, existing["asset_id"], status="pending", error="", settings=used)
    else:
        asset = asset_lib.register_asset(
            svc.store, owner, kind="image", title=f"{slide['title']} (AI)"[:200], origin="ai_generated", provider="qwencloud",
            model=result["model"], settings_used=used, status="pending", room_id=deck.get("room_id", ""),
            session_id=deck.get("session_id", ""), slide_index=index, idempotency_key=key)
    return {"task_id": result["task_id"], "asset_id": asset["asset_id"], "prompt": prompt, "used_llm": used_llm}


@router.get("/teacher/visuals/{task_id}")
def visual_status_api(svc: Service, task_id: str, authorization: Auth = None):
    owner = _owner(authorization)
    asset = asset_lib.find_by_task(svc.store, owner, task_id)
    if asset is None:
        raise HTTPException(404, "No such picture job")
    settings = asset.get("settings") or {}
    view = {"status": "RUNNING", "asset_id": asset["asset_id"], "file_url": None, "error": None,
            "deck_id": settings.get("deck_id", ""), "slide_index": settings.get("slide_index")}
    if asset["status"] == "ready" and asset_lib.asset_file(asset):
        return {**view, "status": "SUCCEEDED", "file_url": f"/teacher/assets/{asset['asset_id']}/file"}

    result = image_gen.poll_task(task_id)
    if result["status"] == "SUCCEEDED" and result["image_url"]:
        try:
            ext = ".png" if ".png" in result["image_url"].lower() else ".jpg"
            root, ref = asset_lib.save_file(owner, asset["asset_id"], image_gen.download_image(result["image_url"]), f"slide{ext}")
            asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="ready", storage_root=root, storage_ref=ref, error="")
            if settings.get("deck_id") is not None and settings.get("slide_index") is not None:
                try:
                    decks.set_slide_image(svc.store, owner, settings["deck_id"], settings["slide_index"], asset["asset_id"])
                except decks.DeckError:
                    pass  # the slide was removed meanwhile: the picture stays in the library
            return {**view, "status": "SUCCEEDED", "file_url": f"/teacher/assets/{asset['asset_id']}/file"}
        except Exception as e:
            asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="failed", error=f"The picture could not be saved: {str(e)[:150]}")
            return {**view, "status": "FAILED", "error": "The picture could not be saved"}
    if result["status"] in ("FAILED", "CANCELED"):
        asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="failed", error=(result["error"] or "")[:300])
        return {**view, "status": "FAILED", "error": result["error"]}
    return {**view, "status": "ERROR" if result["status"] == "ERROR" else "RUNNING", "error": result["error"]}


# ===================================================================== media
class MediaEdit(BaseModel):
    ops: dict[str, Any] = Field(default_factory=dict)
    title: str = Field(default="", max_length=200)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=2, max_length=media_tools.MAX_SPEECH_CHARS)
    voice: str = Field(default="ar-EG-SalmaNeural", max_length=60)
    title: str = Field(default="", max_length=200)
    room_id: str = Field(default="", max_length=60)


@router.get("/teacher/media/capabilities")
def media_capabilities_api(authorization: Auth = None):
    _owner(authorization)
    return {"image_editing": True, "ffmpeg": media_tools.ffmpeg_path() is not None, "speech": media_tools.speech_available(),
            "voices": [{"voice": v, "label": label} for v, label in media_tools.VOICES.items()]}


@router.post("/teacher/media/upload", status_code=201)
async def upload_media_api(svc: Service, authorization: Auth = None, file: UploadFile = File(...),
                           title: str = Form(""), room_id: str = Form("")):
    """Add your own image, video or audio file to the library (original, never changed afterwards)."""
    owner = _owner(authorization)
    kind = media_tools.kind_for(file.filename or "")
    if not kind:
        raise HTTPException(422, "Upload an image (jpg, png, webp), a video (mp4, webm, mov) or audio (mp3, wav, m4a, ogg)")
    limit = media_tools.MAX_UPLOAD_BYTES[kind]
    data = await file.read(limit + 1)
    if not data or len(data) > limit:
        raise HTTPException(413 if data else 422, f"The file is empty or larger than {limit // (1024 * 1024)} MB")
    asset = asset_lib.register_asset(
        svc.store, owner, kind=kind, title=(title.strip() or Path(file.filename or kind).stem)[:200], origin="upload",
        status="pending", room_id=room_id[:60], idempotency_key="upload:" + hashlib.sha1(data).hexdigest()[:20])
    if asset["status"] == "pending" and not asset["storage_ref"]:
        root, ref = asset_lib.save_file(owner, asset["asset_id"], data, Path(file.filename or "file").name)
        asset = asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="ready", storage_root=root, storage_ref=ref)
    return asset_view(asset)


class ThumbnailRequest(BaseModel):
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    title: str = Field(default="", max_length=200)
    start_sec: float | None = Field(default=None, ge=0)
    room_id: str = Field(default="", max_length=60)


@router.post("/teacher/media/thumbnail", status_code=201)
def save_thumbnail_api(svc: Service, body: ThumbnailRequest, authorization: Auth = None):
    """Keep the cover picture of a YouTube video as an image in the library (an exact frame cannot be
    taken from YouTube; the cover can). Only YouTube's own image host is contacted."""
    owner = _owner(authorization)
    key = f"thumb:{body.video_id}"
    existing = asset_lib.find_by_key(svc.store, owner, key)
    if existing and existing["status"] == "ready":
        return asset_view(existing)
    watch = f"https://www.youtube.com/watch?v={body.video_id}" + (f"&t={int(body.start_sec)}s" if body.start_sec else "")
    try:
        req = urllib.request.Request(f"https://i.ytimg.com/vi/{body.video_id}/hqdefault.jpg", headers={"User-Agent": "Rafeeq"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read(media_tools.MAX_UPLOAD_BYTES["image"] + 1)
    except Exception as e:
        raise HTTPException(502, f"The cover picture could not be downloaded: {type(e).__name__}") from None
    if not data or len(data) > media_tools.MAX_UPLOAD_BYTES["image"]:
        raise HTTPException(502, "The cover picture was empty or too large")
    asset = asset_lib.register_asset(
        svc.store, owner, kind="image", title=(body.title.strip() or "YouTube")[:190] + " (cover)", origin="external",
        source_url=watch, status="pending", room_id=body.room_id, idempotency_key=key)
    root, ref = asset_lib.save_file(owner, asset["asset_id"], data, "cover.jpg")
    return asset_view(asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="ready", storage_root=root, storage_ref=ref))


@router.post("/teacher/media/{asset_id}/edit", status_code=201)
def edit_media_api(svc: Service, asset_id: str, body: MediaEdit, authorization: Auth = None):
    """Edit an image, video or audio from the library. The result is a NEW version; the original stays."""
    owner = _owner(authorization)
    source = asset_lib.get_asset(svc.store, owner, asset_id)
    if source is None or source["kind"] not in ("image", "video", "generated_video", "audio"):
        raise HTTPException(404, "Media not found")
    path = asset_lib.asset_file(source)
    if path is None:
        raise HTTPException(409, "This item has no file to edit yet")
    try:
        if body.ops.get("to_gif") and source["kind"] in ("video", "generated_video"):
            trim = body.ops.get("trim") or {}
            with tempfile.TemporaryDirectory() as tmp:
                gif = story_video.to_gif(path, Path(tmp) / "clip.gif", start=float(trim.get("start") or 0),
                                         end=float(trim["end"]) if trim.get("end") else None,
                                         fps=int(body.ops.get("fps") or 12), width=int(body.ops.get("width") or 480))
                data, name, kind = gif.read_bytes(), "clip.gif", "image"
        elif source["kind"] == "image":
            data, name, kind = media_tools.edit_image(path.read_bytes(), path.suffix, body.ops), f"edited{path.suffix.lower()}", "image"
        else:
            with tempfile.TemporaryDirectory() as tmp:
                media_kind = "audio" if source["kind"] == "audio" else "video"
                out = media_tools.edit_media(path, Path(tmp), media_kind, body.ops)
                data, name = out.read_bytes(), out.name
                kind = "audio" if out.suffix == ".mp3" else "video"
    except (media_tools.MediaError, story_video.StoryError, ValueError) as e:
        raise HTTPException(503 if "ffmpeg is not installed" in str(e) else 422, str(e)) from None
    key = "edit:" + hashlib.sha1(f"{asset_id}|{json.dumps(body.ops, sort_keys=True)}|{body.title}".encode()).hexdigest()[:24]
    new = asset_lib.register_asset(
        svc.store, owner, kind=kind, title=(body.title.strip() or source["title"])[:200], origin="edited", status="pending",
        parent_asset_id=asset_id, version=source["version"] + 1, room_id=source.get("room_id", ""),
        session_id=source.get("session_id", ""), settings_used={"ops": body.ops, "from": asset_id}, idempotency_key=key)
    if new["status"] == "ready" and asset_lib.asset_file(new):   # the same edit was already done: no duplicate
        return asset_view(new)
    root, ref = asset_lib.save_file(owner, new["asset_id"], data, name)
    return asset_view(asset_lib.update_asset(svc.store, owner, new["asset_id"], status="ready", storage_root=root, storage_ref=ref))


@router.post("/teacher/media/speech", status_code=201)
def speech_api(svc: Service, body: SpeechRequest, authorization: Auth = None):
    """Narration: read a text aloud with a natural voice and keep it as an audio file in the library."""
    owner = _owner(authorization)
    asset = asset_lib.register_asset(
        svc.store, owner, kind="audio", title=(body.title.strip() or body.text.strip()[:50]), origin="ai_generated",
        provider="edge-tts", model=body.voice, status="pending", room_id=body.room_id,
        settings_used={"text": body.text.strip(), "voice": body.voice},
        idempotency_key="speech:" + hashlib.sha1(f"{body.voice}|{body.text.strip()}".encode()).hexdigest()[:24])
    if asset["status"] == "ready" and asset_lib.asset_file(asset):
        return asset_view(asset)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = media_tools.synthesize_speech(body.text, body.voice, Path(tmp) / "speech.mp3")
            root, ref = asset_lib.save_file(owner, asset["asset_id"], out.read_bytes(), "speech.mp3")
    except media_tools.MediaError as e:
        asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="failed", error=str(e)[:300])
        raise HTTPException(422 if "Unknown voice" in str(e) or "Write the text" in str(e) else 503, str(e)) from None
    return asset_view(asset_lib.update_asset(svc.store, owner, asset["asset_id"], status="ready", storage_root=root, storage_ref=ref))


# ===================================================================== smart montage (a recorded lecture -> a finished video)
class MontageRequest(BaseModel):
    """What to do to the video. Every step is optional; at least one must be on."""
    cut_silence: bool = False
    captions: bool = False
    burn_captions: bool = False
    chapters: bool = False
    intro: bool = False
    outro: bool = False
    title: str = Field(default="", max_length=montage.MAX_TITLE)
    subtitle: str = Field(default="", max_length=montage.MAX_TITLE)
    outro_text: str = Field(default="", max_length=montage.MAX_TITLE)
    music_asset_id: str = Field(default="", max_length=40)
    # decisions made in the montage editor, on the ORIGINAL video's timeline (None = the team decides)
    keep: list[list[float]] | None = Field(default=None, max_length=400)
    segments: list[dict] | None = Field(default=None, max_length=3000)
    chapter_list: list[dict] | None = Field(default=None, max_length=30)
    stickers: list[dict] | None = Field(default=None, max_length=montage.MAX_STICKERS * 2)     # GIFs / pictures placed on the timeline


def _montage_source(svc, owner: str, asset_id: str) -> tuple[dict, Path]:
    source = asset_lib.get_asset(svc.store, owner, asset_id)
    if source is None or source["kind"] not in ("video", "generated_video") or source["status"] != "ready":
        raise HTTPException(404, "Video not found")
    path = asset_lib.asset_file(source)
    if path is None:
        raise HTTPException(404, "This video has no file yet")
    return source, path


def _media_error(e: Exception) -> HTTPException:
    return HTTPException(503 if "ffmpeg is not installed" in str(e) else 422, str(e))


@router.get("/teacher/media/gifs")
def montage_gifs_api(q: str = "", authorization: Auth = None):
    """Search Giphy for the editor (empty q = trending). Needs GIPHY_API_KEY."""
    _owner(authorization)
    try:
        return {"gifs": montage.giphy_search(q)}
    except media_tools.MediaError as e:
        raise HTTPException(503 if "GIPHY_API_KEY" in str(e) else 502, str(e)) from None


@router.post("/teacher/media/{asset_id}/montage/analyze")
def montage_analyze_api(svc: Service, asset_id: str, authorization: Auth = None):
    """For the editor: length, size, sound and where the long pauses are (nothing is changed)."""
    _, path = _montage_source(svc, _owner(authorization), asset_id)
    try:
        return montage.analyze(path)
    except media_tools.MediaError as e:
        raise _media_error(e) from None


@router.post("/teacher/media/{asset_id}/montage/transcribe")
def montage_transcribe_api(svc: Service, asset_id: str, authorization: Auth = None):
    """For the editor: caption lines and suggested chapters of the original video, to be corrected by the teacher."""
    _, path = _montage_source(svc, _owner(authorization), asset_id)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return montage.transcribe_for_editor(path, Path(tmp))
    except media_tools.MediaError as e:
        raise _media_error(e) from None


@router.post("/teacher/media/{asset_id}/montage", status_code=201)
def montage_start_api(svc: Service, asset_id: str, body: MontageRequest, authorization: Auth = None):
    """Start a montage of one of the teacher's videos. Returns the new (pending) library video; poll
    GET /teacher/montages/{id}. The original video is never changed."""
    owner = _owner(authorization)
    source = asset_lib.get_asset(svc.store, owner, asset_id)
    if source is None or source["kind"] not in ("video", "generated_video") or source["status"] != "ready":
        raise HTTPException(404, "Video not found")
    try:
        started = montage.start(svc.store, owner, source, body.model_dump())
    except media_tools.MediaError as e:
        raise HTTPException(503 if "ffmpeg is not installed" in str(e) else 422, str(e)) from None
    return asset_view(started)


@router.get("/teacher/montages/{asset_id}")
def montage_status_api(svc: Service, asset_id: str, authorization: Auth = None):
    owner = _owner(authorization)
    found = montage.view(svc.store, owner, asset_id)
    if found is None:
        raise HTTPException(404, "Montage not found")
    return asset_view(found)


@router.get("/teacher/montages/{asset_id}/captions.srt")
def montage_captions_api(svc: Service, asset_id: str, authorization: Auth = None):
    owner = _owner(authorization)
    found = montage.view(svc.store, owner, asset_id)
    srt = ((found or {}).get("settings") or {}).get("montage", {}).get("srt", "")
    if not srt:
        raise HTTPException(404, "This video has no captions")
    return PlainTextResponse(srt, media_type="text/plain; charset=utf-8",
                             headers={"Content-Disposition": 'attachment; filename="captions.srt"'})


# ===================================================================== story videos (about 30 s, several scenes)
class StoryCreate(BaseModel):
    topic: str = Field(min_length=3, max_length=400)
    scenes: int = Field(default=story_video.DEFAULT_SCENES, ge=story_video.MIN_SCENES, le=story_video.MAX_SCENES)
    style: str = Field(default="storybook", max_length=20)
    language: str = Field(default="ar", pattern="^(ar|en)$")
    voice: str = Field(default="", max_length=60)
    fade: bool = True
    vision: bool = False
    room_id: str = Field(default="", max_length=60)
    retry: bool = False


def _story_or_404(svc, owner: str, asset_id: str) -> dict:
    a = asset_lib.get_asset(svc.store, owner, asset_id)
    if a is None or not (a.get("settings") or {}).get("story"):
        raise HTTPException(404, "Story not found")
    return a


@router.post("/teacher/stories", status_code=201)
def create_story_api(svc: Service, body: StoryCreate, authorization: Auth = None):
    """Plan a multi-scene video (an LLM writes the scenes) and start it. Every scene is a paid video job and
    each one continues from the last frame of the one before. Poll GET /teacher/stories/{id}. The same
    request returns the same story; a failed one restarts only with retry=true."""
    owner = _owner(authorization)
    if media_tools.ffmpeg_path() is None:
        raise HTTPException(503, "ffmpeg is not installed on the server, so scenes cannot be joined into one video")
    try:
        asset = story_video.start_story(svc.store, owner, body.topic, scenes=body.scenes, style=body.style,
                                        language=body.language, voice=body.voice, fade=body.fade, vision=body.vision,
                                        room_id=body.room_id, retry=body.retry)
    except story_video.StoryError as e:
        raise HTTPException(503 if "DASHSCOPE_API_KEY" in str(e) else 422, str(e)) from None
    return story_video.story_view(asset)


@router.get("/teacher/stories/{asset_id}")
def story_status_api(svc: Service, asset_id: str, authorization: Auth = None):
    """Progress of a story. Each call also moves it one step forward (so the page only has to poll)."""
    owner = _owner(authorization)
    _story_or_404(svc, owner, asset_id)
    return story_video.story_view(story_video.advance(svc.store, owner, asset_id))



# ===================================================================== courses (My Learning Platform)
class CourseCreate(BaseModel):
    title: str = Field(min_length=2, max_length=160)
    description: str = Field(default="", max_length=1000)
    room_id: str = Field(default="", max_length=60)


class CourseUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=160)
    description: str | None = Field(default=None, max_length=1000)
    room_id: str | None = Field(default=None, max_length=60)


class LessonCreate(BaseModel):
    title: str = Field(min_length=2, max_length=160)
    notes: str = Field(default="", max_length=2000)
    session_id: str = Field(default="", max_length=80)
    asset_ids: list[str] = Field(default_factory=list, max_length=courses.MAX_ASSETS_PER_LESSON)


class LessonUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=160)
    notes: str | None = Field(default=None, max_length=2000)


class LessonMove(BaseModel):
    direction: str = Field(pattern="^(up|down)$")


class LessonAsset(BaseModel):
    add: str = Field(default="", max_length=40)
    remove: str = Field(default="", max_length=40)


class PublishRequest(BaseModel):
    room_id: str = Field(min_length=1, max_length=60)


def course_view(store, owner: str, c: dict) -> dict:
    lessons = []
    for lesson in c["lessons"]:
        items = [asset_view(a) for aid in lesson["asset_ids"] if (a := asset_lib.get_asset(store, owner, aid))]
        lessons.append({"lesson_id": lesson["lesson_id"], "title": lesson["title"], "notes": lesson.get("notes", ""),
                        "session_id": lesson.get("session_id", ""), "assets": items})
    return {"course_id": c["course_id"], "title": c["title"], "description": c.get("description", ""),
            "room_id": c.get("room_id", ""), "published_room_id": c.get("published_room_id", ""),
            "published_at": c.get("published_at", ""), "unpublished_changes": courses.has_unpublished_changes(c),
            "lessons": lessons, "created_at": c.get("created_at", ""), "updated_at": c.get("updated_at", "")}


def _course_or_404(svc, owner: str, course_id: str) -> dict:
    course = courses.get_course(svc.store, owner, course_id)
    if course is None:
        raise HTTPException(404, "Course not found")
    return course


def _course_call(fn, *args, **kw):
    try:
        result = fn(*args, **kw)
    except courses.CourseError as e:
        raise _422(e) from None
    if result is None:
        raise HTTPException(404, "Course not found")
    return result


@router.get("/teacher/courses")
def list_courses_api(svc: Service, authorization: Auth = None):
    owner = _owner(authorization)
    return {"courses": [course_view(svc.store, owner, c) for c in courses.list_courses(svc.store, owner)]}


@router.post("/teacher/courses", status_code=201)
def create_course_api(svc: Service, body: CourseCreate, authorization: Auth = None):
    owner = _owner(authorization)
    try:
        c = courses.create_course(svc.store, owner, body.title, body.description, body.room_id)
    except courses.CourseError as e:
        raise _422(e) from None
    return course_view(svc.store, owner, c)


@router.get("/teacher/courses/{course_id}")
def get_course_api(svc: Service, course_id: str, authorization: Auth = None):
    owner = _owner(authorization)
    return course_view(svc.store, owner, _course_or_404(svc, owner, course_id))


@router.put("/teacher/courses/{course_id}")
def update_course_api(svc: Service, course_id: str, body: CourseUpdate, authorization: Auth = None):
    owner = _owner(authorization)
    c = _course_call(courses.update_course, svc.store, owner, course_id, title=body.title, description=body.description, room_id=body.room_id)
    return course_view(svc.store, owner, c)


@router.delete("/teacher/courses/{course_id}", status_code=204)
def delete_course_api(svc: Service, course_id: str, authorization: Auth = None):
    owner = _owner(authorization)
    if not courses.delete_course(svc.store, owner, course_id):
        raise HTTPException(404, "Course not found")


@router.post("/teacher/courses/{course_id}/lessons", status_code=201)
def add_lesson_api(svc: Service, course_id: str, body: LessonCreate, authorization: Auth = None):
    owner = _owner(authorization)
    c = _course_call(courses.add_lesson, svc.store, owner, course_id, body.title, session_id=body.session_id,
                     notes=body.notes, asset_ids=body.asset_ids)
    return course_view(svc.store, owner, c)


@router.put("/teacher/courses/{course_id}/lessons/{lesson_id}")
def update_lesson_api(svc: Service, course_id: str, lesson_id: str, body: LessonUpdate, authorization: Auth = None):
    owner = _owner(authorization)
    c = _course_call(courses.update_lesson, svc.store, owner, course_id, lesson_id, title=body.title, notes=body.notes)
    return course_view(svc.store, owner, c)


@router.delete("/teacher/courses/{course_id}/lessons/{lesson_id}")
def remove_lesson_api(svc: Service, course_id: str, lesson_id: str, authorization: Auth = None):
    owner = _owner(authorization)
    return course_view(svc.store, owner, _course_call(courses.remove_lesson, svc.store, owner, course_id, lesson_id))


@router.post("/teacher/courses/{course_id}/lessons/{lesson_id}/move")
def move_lesson_api(svc: Service, course_id: str, lesson_id: str, body: LessonMove, authorization: Auth = None):
    owner = _owner(authorization)
    return course_view(svc.store, owner, _course_call(courses.move_lesson, svc.store, owner, course_id, lesson_id, body.direction))


@router.post("/teacher/courses/{course_id}/lessons/{lesson_id}/assets")
def lesson_assets_api(svc: Service, course_id: str, lesson_id: str, body: LessonAsset, authorization: Auth = None):
    owner = _owner(authorization)
    c = _course_call(courses.set_lesson_assets, svc.store, owner, course_id, lesson_id, add=body.add, remove=body.remove)
    return course_view(svc.store, owner, c)


def _platform_user(authorization: str | None) -> dict:
    from rafeeq.platform import service as platform

    token = platform.token_from_header(authorization) if authorization else None
    user = platform.get_store().user_for_token(token) if token else None
    if user is None:
        raise HTTPException(401, "Sign in first")
    return user


@router.post("/teacher/courses/{course_id}/publish")
def publish_course_api(svc: Service, course_id: str, body: PublishRequest, authorization: Auth = None):
    """Show the course to the students of one of YOUR rooms (a snapshot: publish again after editing)."""
    from rafeeq.platform import service as platform

    owner = _owner(authorization)
    user = _platform_user(authorization)
    try:
        platform.get_store()._owned_room(user, body.room_id)
    except platform.PlatformError as e:
        raise HTTPException(e.status, e.message) from None
    c = _course_call(courses.publish, svc.store, owner, course_id, body.room_id)
    return course_view(svc.store, owner, c)


@router.post("/teacher/courses/{course_id}/unpublish")
def unpublish_course_api(svc: Service, course_id: str, authorization: Auth = None):
    owner = _owner(authorization)
    return course_view(svc.store, owner, _course_call(courses.unpublish, svc.store, owner, course_id))


# ----------------------------------------------------------------- what students see
def _member_of(authorization: str | None, room_id: str) -> dict:
    from rafeeq.platform import service as platform

    user = _platform_user(authorization)
    try:
        platform.get_store().room_for(user, room_id)
    except platform.PlatformError as e:
        raise HTTPException(e.status, e.message) from None
    return user


@router.get("/rooms/{room_id}/courses")
def room_courses_api(svc: Service, room_id: str, authorization: Auth = None):
    """The courses a teacher published to this room (students and the teacher of the room)."""
    _member_of(authorization, room_id)
    out = []
    for c in courses.published_for_room(svc.store, room_id):
        lessons = [{"lesson_id": l["lesson_id"], "title": l["title"], "notes": l["notes"],
                    "assets": [{**{k: a[k] for k in ("asset_id", "title", "kind", "source_url", "start_sec", "end_sec")},
                                "file_url": f"/rooms/{room_id}/courses/{c['course_id']}/assets/{a['asset_id']}/file" if a["has_file"] else None}
                               for a in l["assets"]]} for l in c["lessons"]]
        out.append({"course_id": c["course_id"], "title": c["title"], "description": c["description"],
                    "published_at": c["published_at"], "lessons": lessons})
    return {"courses": out}


@router.get("/rooms/{room_id}/courses/{course_id}/assets/{asset_id}/file")
def room_course_file_api(svc: Service, room_id: str, course_id: str, asset_id: str, authorization: Auth = None):
    _member_of(authorization, room_id)
    owner = courses.published_owner(svc.store, room_id, course_id, asset_id)
    asset = asset_lib.get_asset(svc.store, owner, asset_id) if owner else None
    path = asset_lib.asset_file(asset) if asset else None
    if path is None:
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=path.name)

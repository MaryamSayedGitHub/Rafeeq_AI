"""The teacher's studios, each on its own page: Video Discovery, Presentations, Visual & Media,
My Learning Platform (plus the student view of published courses).

Every page talks to the same JSON API as the rest of the UI (rafeeq/teacher/routes.py and
rafeeq/teacher/studio_routes.py), so what one studio saves shows up in all the others. The project
being worked on (?room=&session=) travels in the URL between studios.
"""
from __future__ import annotations

from urllib.parse import quote

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from rafeeq.web import api as A
from rafeeq.web.common import (api, api_message, check_csrf, flash, lang_of, notice, redirect, render, require_user, tr, url as u)
from rafeeq.web.views_rooms import ensure_session, gather, load_room
from rafeeq.web.views_teacher import _image_part, _image_too_large, _int

STORY_STYLES = ("storybook", "flat", "clay", "realistic")
MEDIA_KINDS = ("image", "video", "generated_video", "audio")
COURSE_SKIP_KINDS = ("theme",)
MAX_MEDIA_BODY = {"image": 15, "video": 1024, "audio": 40}


# ------------------------------------------------------------------ shared helpers
def _msg(request: Request, res: A.ApiResult, specific: dict | None = None) -> str:
    """The API's own sentence for 'this tool is not set up' (503), the generic text otherwise."""
    if res.status == 503 and res.message:
        return res.message
    return api_message(request, res, specific)


def _done(request: Request, res: A.ApiResult, ok_key: str, specific: dict | None = None) -> bool:
    ensure_session(request, res)
    flash(request, "success" if res.ok else "warning", tr(request, ok_key) if res.ok else _msg(request, res, specific))
    return res.ok


def _num(form, name: str) -> float | None:
    try:
        return float(str(form.get(name, "")).strip())
    except ValueError:
        return None


async def project(request: Request, user: dict) -> dict:
    """The room and lesson the teacher is working on, taken from ?room= and ?session= (only their own)."""
    rooms_res, sessions_res = await gather(
        api(request).get("/rooms", token=user["token"]),
        api(request).get("/teacher/sessions", token=user["token"], params={"teacher_id": user["user_id"]}))
    ensure_session(request, rooms_res)
    ensure_session(request, sessions_res)
    rooms = rooms_res.get("rooms", [])
    sessions = sessions_res.get("sessions", [])
    session_id = request.query_params.get("session", "")[:80]
    session = next((s for s in sessions if s["session_id"] == session_id), None)
    room_id = request.query_params.get("room", "")[:60] or (session or {}).get("room_id", "")
    room = next((r for r in rooms if r["room_id"] == room_id), None)
    return {"rooms": rooms, "sessions": sessions, "room": room, "session": session,
            "room_names": {r["room_id"]: r["name"] for r in rooms},
            "q": {"room": room["room_id"] if room else "", "session": session["session_id"] if session else ""}}


def page(request: Request, template: str, tab: str, proj: dict, **extra):
    return render(request, template, room=None, tab=tab, proj=proj, nav_q=proj["q"], **extra)


async def _assets(request: Request, user: dict, **params) -> A.ApiResult:
    res = await api(request).get("/teacher/assets", token=user["token"], params=params, timeout=60)
    ensure_session(request, res)
    return res


# ------------------------------------------------------------------ Lesson Studio entry
async def studio_entry(request: Request):
    """/studio: the Lesson Studio of the room being worked on (or the first room)."""
    user = require_user(request, "teacher")
    proj = await project(request, user)
    room = proj["room"] or (proj["rooms"][0] if proj["rooms"] else None)
    if room is None:
        flash(request, "warning", tr(request, "studios.need_room"))
        return redirect(request, "/rooms")
    return redirect(request, f"/rooms/{room['room_id']}/studio")


# ------------------------------------------------------------------ Video Discovery Studio
async def video_page(request: Request):
    user = require_user(request, "teacher")
    proj = await project(request, user)
    params = {"kind": "source_video"}
    if proj["room"]:
        params["room_id"] = proj["room"]["room_id"]
    moments_res = await _assets(request, user, **params)
    covers = [a for a in (await _assets(request, user, kind="image")).get("assets", []) if a.get("origin") == "external"]
    query = request.query_params.get("q", "")[:300] or (proj["session"] or {}).get("topic", "")
    return page(request, "studios/video.html", "video", proj, query=query, moments_res=moments_res,
                moments=moments_res.get("assets", []), covers=covers[:6],
                history=await _video_history(request, user, proj["room"]["room_id"] if proj["room"] else ""))


async def _video_history(request: Request, user: dict, room_id: str = "") -> list[dict]:
    """The teacher's earlier searches (empty when the API cannot say)."""
    res = await api(request).get("/teacher/videos/searches", token=user["token"], params={"room_id": room_id}, timeout=30)
    ensure_session(request, res)
    return res.get("searches", []) if res.ok else []


async def video_search(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    q = str(form.get("q", "")).strip()
    if len(q) < 3:
        return notice(request, tr(request, "video.need_question"), "warning")
    room_id = str(form.get("room_id", ""))[:60]
    res = await api(request).get("/teacher/videos/search", token=user["token"],
                                 params={"q": q[:300], "top_k": 5, "room_id": room_id}, timeout=180)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    return render(request, "partials/video_results.html", room_id=room_id, query=q,
                  results=res.data.get("results", []), history=await _video_history(request, user, room_id), history_oob=True)


async def video_history_open(request: Request):
    """An earlier search, shown again from the history: the saved results, nothing is searched again."""
    user = require_user(request, "teacher")
    res = await api(request).get(f"/teacher/videos/searches/{quote(request.path_params['search_id'], safe='')}",
                                 token=user["token"], timeout=30)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, tr(request, "video.history_gone"), "warning")
    return render(request, "partials/video_results.html", room_id=res.data.get("room_id", ""), query=res.data.get("query", ""),
                  results=res.data.get("results", []), from_history=res.data.get("time", ""))


async def video_history_delete(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    res = await api(request).delete(f"/teacher/videos/searches/{quote(request.path_params['search_id'], safe='')}",
                                    token=user["token"], timeout=30)
    ensure_session(request, res)
    room_id = str(form.get("room_id", ""))[:60]
    return render(request, "partials/video_history.html", history=await _video_history(request, user, room_id), room_id=room_id)


async def video_moment_save(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    body = {"title": str(form.get("title", "")).strip()[:200], "source_url": str(form.get("source_url", "")).strip()[:500],
            "start_sec": _num(form, "start_sec"), "end_sec": _num(form, "end_sec"), "room_id": str(form.get("room_id", ""))[:60]}
    if not body["title"] or not body["source_url"]:
        return notice(request, tr(request, "library.bad_item"), "warning")
    res = await api(request).post("/teacher/assets", token=user["token"], json=body)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    return notice(request, tr(request, "library.saved"), "success")


async def video_cover_save(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    body = {"video_id": str(form.get("video_id", "")).strip(), "title": str(form.get("title", "")).strip()[:190],
            "room_id": str(form.get("room_id", ""))[:60]}
    start = _num(form, "start_sec")
    if start is not None:
        body["start_sec"] = max(0.0, start)
    res = await api(request).post("/teacher/media/thumbnail", token=user["token"], json=body, timeout=60)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, tr(request, "video.cover_failed") if res.status == 502 else api_message(request, res), "warning")
    return notice(request, tr(request, "video.cover_saved"), "success")


# ------------------------------------------------------------------ Visual & Media Studio
async def media_page(request: Request):
    user = require_user(request, "teacher")
    proj = await project(request, user)
    assets_res, caps_res = await gather(_assets(request, user), api(request).get("/teacher/media/capabilities", token=user["token"]))
    ensure_session(request, caps_res)
    items = [a for a in assets_res.get("assets", []) if a["kind"] in MEDIA_KINDS]
    if proj["room"]:
        items = [a for a in items if a.get("room_id") in ("", proj["room"]["room_id"])]
    selected_id = request.query_params.get("asset", "")[:40]
    selected = next((a for a in items if a["asset_id"] == selected_id), None)
    if selected is None and selected_id:
        selected = next((a for a in assets_res.get("assets", []) if a["asset_id"] == selected_id and a["kind"] in MEDIA_KINDS), None)
    music = [a for a in assets_res.get("assets", []) if a["kind"] == "audio" and a["status"] == "ready"]
    return page(request, "studios/media.html", "media", proj, res=assets_res, items=items, selected=selected, music=music,
                caps=caps_res.data if caps_res.ok else {}, caps_ok=caps_res.ok)


async def media_upload(request: Request):
    user = require_user(request, "teacher")
    try:
        if int(request.headers.get("content-length", "0")) > (MAX_MEDIA_BODY["video"] + 2) * 1024 * 1024:
            flash(request, "warning", tr(request, "media.too_big"))
            return redirect(request, "/media")
    except ValueError:
        pass
    form = await check_csrf(request)
    upload = form.get("file")
    if upload is None or not getattr(upload, "filename", ""):
        flash(request, "warning", tr(request, "media.need_file"))
        return redirect(request, "/media")
    data = await upload.read(MAX_MEDIA_BODY["video"] * 1024 * 1024 + 1)
    res = await api(request).post("/teacher/media/upload", token=user["token"], timeout=600,
                                  data={"title": str(form.get("title", "")).strip()[:200], "room_id": str(form.get("room_id", ""))[:60]},
                                  files={"file": (upload.filename, data, upload.content_type or "application/octet-stream")})
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "media.uploaded"))
        return redirect(request, "/media", asset=res.data["asset_id"], room=str(form.get("room_id", "")))
    flash(request, "warning", tr(request, "media.bad_file") if res.kind in (A.INVALID,) or res.status == 413 else _msg(request, res))
    return redirect(request, "/media")


def _ops_from_form(form, kind: str) -> dict:
    ops: dict = {}
    if kind == "image":
        w, h = _num(form, "crop_w"), _num(form, "crop_h")
        if w is not None and h is not None and (w < 100 or h < 100):
            ops["crop"] = {"x": _num(form, "crop_x") or 0, "y": _num(form, "crop_y") or 0, "w": w, "h": h}
        if str(form.get("rotate", "0")) in ("90", "180", "270"):
            ops["rotate"] = int(str(form.get("rotate")))
        if str(form.get("flip", "")) in ("h", "v", "both"):
            ops["flip"] = str(form.get("flip"))
        width = _num(form, "width")
        if width:
            ops["width"] = width
        for name in ("brightness", "contrast"):
            value = _num(form, name)
            if value is not None:
                ops[name] = value
        if form.get("grayscale"):
            ops["grayscale"] = True
    else:
        start, end = _num(form, "trim_start"), _num(form, "trim_end")
        if end:
            ops["trim"] = {"start": start or 0, "end": end}
        volume = _num(form, "volume")
        if volume is not None and volume != 1:
            ops["volume"] = volume
        if form.get("mute"):
            ops["mute"] = True
        if form.get("extract_audio"):
            ops["extract_audio"] = True
        if form.get("to_gif"):
            ops["to_gif"] = True
    return ops


async def media_edit(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    asset_id = request.path_params["asset_id"]
    src = await api(request).get(f"/teacher/assets/{quote(asset_id, safe='')}", token=user["token"])
    ensure_session(request, src)
    if not src.ok or src.data["kind"] not in MEDIA_KINDS:
        raise HTTPException(status_code=404, detail="asset")
    kind = "image" if src.data["kind"] == "image" else "audio" if src.data["kind"] == "audio" else "video"
    ops = _ops_from_form(form, "image" if kind == "image" else "media")
    if not ops:
        flash(request, "warning", tr(request, "media.no_edit"))
        return redirect(request, "/media", asset=asset_id)
    res = await api(request).post(f"/teacher/media/{quote(asset_id, safe='')}/edit", token=user["token"], timeout=300,
                                  json={"ops": ops, "title": str(form.get("title", "")).strip()[:200]})
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "media.edited"))
        return redirect(request, "/media", asset=res.data["asset_id"])
    flash(request, "warning", _msg(request, res))
    return redirect(request, "/media", asset=asset_id)


async def media_speech(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    body = {"text": str(form.get("text", "")).strip(), "voice": str(form.get("voice", "ar-EG-SalmaNeural")),
            "title": str(form.get("title", "")).strip()[:200], "room_id": str(form.get("room_id", ""))[:60]}
    if len(body["text"]) < 2:
        flash(request, "warning", tr(request, "media.need_text"))
        return redirect(request, "/media")
    res = await api(request).post("/teacher/media/speech", token=user["token"], json=body, timeout=120)
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "media.speech_saved"))
        return redirect(request, "/media", asset=res.data["asset_id"])
    flash(request, "warning", _msg(request, res))
    return redirect(request, "/media")


async def media_video_generate(request: Request):
    user = require_user(request, "teacher")
    if _image_too_large(request):
        return notice(request, tr(request, "video.bad_image"), "warning")
    form = await check_csrf(request)
    prompt = str(form.get("prompt", "")).strip()
    if len(prompt) < 5:
        return notice(request, tr(request, "video.need_prompt"), "warning")
    image = await _image_part(form.get("image"))
    if image is False:
        return notice(request, tr(request, "video.bad_image"), "warning")
    data = {"teacher_id": user["user_id"], "prompt": prompt[:800],
            "duration": str(_int(form.get("duration"), 2, 6, 5)), "audio": "true" if form.get("audio") else "false"}
    res = await api(request).post("/teacher/videos/generate", token=user["token"], data=data,
                                  files={"image": image} if image else None, timeout=60)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, tr(request, "video.no_key") if res.status == 503 else api_message(request, res), "warning")
    return render(request, "partials/video_job.html", poll_base="/media/video/generate/", task_id=res.data["task_id"],
                  job=None, used_avatar=res.data.get("used_avatar", False), prompt=prompt)


async def media_video_status(request: Request):
    user = require_user(request, "teacher")
    task_id = request.path_params["task_id"]
    res = await api(request).get(f"/teacher/videos/generate/{quote(task_id, safe='')}", token=user["token"], timeout=30)
    ensure_session(request, res)
    job = dict(res.data) if res.ok else {"status": "ERROR", "video_url": None, "error": api_message(request, res)}
    saved = bool(job.get("file_url") and job.get("asset_id"))
    if saved:
        job["video_url"] = u(request, f"/assets/{quote(job['asset_id'], safe='')}/file")
    return render(request, "partials/video_job.html", poll_base="/media/video/generate/", task_id=task_id, job=job,
                  used_avatar=False, prompt=request.query_params.get("prompt", "")[:800], saved=saved,
                  edit_url=u(request, "/media", asset=job["asset_id"]) if saved else "")


# ------------------------------------------------------------------ Smart montage of a recorded lecture
async def montage_start(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    asset_id = request.path_params.get("asset_id") or str(form.get("video_asset_id", ""))[:40]
    if not asset_id:
        return notice(request, tr(request, "studio.video_none"), "warning")
    body = {name: bool(form.get(name)) for name in ("cut_silence", "captions", "burn_captions", "chapters", "intro", "outro")}
    body.update({"title": str(form.get("title", "")).strip()[:120], "subtitle": str(form.get("subtitle", "")).strip()[:120],
                 "outro_text": str(form.get("outro_text", "")).strip()[:120],
                 "music_asset_id": str(form.get("music_asset_id", ""))[:40] if form.get("music") else ""})
    if not any(body[k] for k in ("cut_silence", "captions", "chapters", "intro", "outro")) and not body["music_asset_id"]:
        return notice(request, tr(request, "montage.pick_one"), "warning")
    if body["intro"] and not body["title"]:
        return notice(request, tr(request, "montage.need_title"), "warning")
    res = await api(request).post(f"/teacher/media/{quote(asset_id, safe='')}/montage", token=user["token"], json=body, timeout=60)
    ensure_session(request, res)
    if not res.ok:
        key = "media.no_ffmpeg" if res.status == 503 and "ffmpeg" in (res.message or "") else None
        return notice(request, tr(request, key) if key else (res.message or api_message(request, res)), "warning")
    return render(request, "partials/montage_job.html", job=res.data)


async def montage_status(request: Request):
    user = require_user(request, "teacher")
    res = await api(request).get(f"/teacher/montages/{quote(request.path_params['asset_id'], safe='')}", token=user["token"], timeout=30)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    return render(request, "partials/montage_job.html", job=res.data)


async def montage_captions(request: Request):
    user = require_user(request, "teacher")
    res = await api(request).get(f"/teacher/montages/{quote(request.path_params['asset_id'], safe='')}", token=user["token"], timeout=30)
    srt = ((res.data or {}).get("settings") or {}).get("montage", {}).get("srt", "") if res.ok else ""
    if not srt:
        raise HTTPException(status_code=404, detail="captions")
    return PlainTextResponse(srt, media_type="text/plain; charset=utf-8",
                             headers={"Content-Disposition": 'attachment; filename="captions.srt"'})


# ------------------------------------------------------------------ montage editor (live preview, then the agents render it)
async def montage_editor_page(request: Request):
    user = require_user(request, "teacher")
    asset_id = request.path_params["asset_id"]
    proj = await project(request, user)
    asset_res, assets_res, caps_res = await gather(
        api(request).get(f"/teacher/assets/{quote(asset_id, safe='')}", token=user["token"]),
        _assets(request, user), api(request).get("/teacher/media/capabilities", token=user["token"]))
    ensure_session(request, asset_res)
    asset = asset_res.data if asset_res.ok else None
    if not asset or asset.get("kind") not in ("video", "generated_video") or asset.get("status") != "ready" or not asset.get("file_url"):
        flash(request, "warning", tr(request, "editor.no_video"))
        return redirect(request, "/media")
    music = [a for a in assets_res.get("assets", []) if a["kind"] == "audio" and a["status"] == "ready"]
    pictures = [a for a in assets_res.get("assets", []) if a["kind"] == "image" and a["status"] == "ready"]
    return page(request, "studios/montage_editor.html", "media", proj, asset=asset, music=music, pictures=pictures,
                caps=caps_res.data if caps_res.ok else {})


async def montage_editor_api(request: Request):
    """The editor's calls (JSON in, JSON out): analyze (pauses), transcribe (captions + chapters), gifs (Giphy search),
    visuals + visual_status (Voice-to-Visual), render."""
    user = require_user(request, "teacher")
    await check_csrf(request)
    asset_id, action = quote(request.path_params["asset_id"], safe=""), request.path_params["action"]
    base = f"/teacher/media/{asset_id}/montage"
    if action in ("analyze", "transcribe"):
        res = await api(request).post(f"{base}/{action}", token=user["token"], timeout=1200)   # a 90 min lecture takes minutes
    elif action == "gifs":
        try:
            q = str((await request.json()).get("q", ""))[:80]
        except (ValueError, AttributeError):
            q = ""
        res = await api(request).get("/teacher/media/gifs", token=user["token"], params={"q": q}, timeout=30)
    elif action == "visuals":                      # Voice-to-Visual: plan the pictures from what is said
        try:
            body = await request.json()
        except ValueError:
            body = {}
        segs = body.get("segments") if isinstance(body, dict) else None
        res = await api(request).post(f"{base}/visuals", token=user["token"], json={"segments": segs if isinstance(segs, list) else None},
                                      timeout=1200)
    elif action == "visual_status":
        try:
            task_id = str((await request.json()).get("task_id", ""))[:80]
        except (ValueError, AttributeError):
            task_id = ""
        if not task_id:
            return JSONResponse({"error": tr(request, "err.invalid")}, status_code=400)
        res = await api(request).get(f"/teacher/visuals/{quote(task_id, safe='')}", token=user["token"], timeout=30)
    elif action == "render":
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": tr(request, "err.invalid")}, status_code=400)
        res = await api(request).post(base, token=user["token"], json=body, timeout=60)
    else:
        raise HTTPException(status_code=404, detail="action")
    ensure_session(request, res)
    if not res.ok:
        key = "media.no_ffmpeg" if res.status == 503 and "ffmpeg" in (res.message or "") else None
        text = tr(request, key) if key else (res.message or api_message(request, res))
        return JSONResponse({"error": text}, status_code=res.status if 400 <= res.status < 600 else 502)
    if action == "render":
        return render(request, "partials/montage_job.html", job=res.data)
    return JSONResponse(res.data)


# ------------------------------------------------------------------ Story video (several scenes, about 30 s)
async def story_page(request: Request):
    user = require_user(request, "teacher")
    proj = await project(request, user)
    assets_res, caps_res = await gather(_assets(request, user, kind="generated_video"),
                                        api(request).get("/teacher/media/capabilities", token=user["token"]))
    ensure_session(request, caps_res)
    stories = [a for a in assets_res.get("assets", []) if (a.get("settings") or {}).get("story")]
    return page(request, "studios/story.html", "story", proj, stories=stories[:8], room_names={}, caps=caps_res.data if caps_res.ok else {},
                styles=STORY_STYLES, topic=request.query_params.get("topic", "")[:400] or (proj["session"] or {}).get("topic", ""))


async def story_create(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    topic = str(form.get("topic", "")).strip()
    if len(topic) < 3:
        return notice(request, tr(request, "story.need_topic"), "warning")
    body = {"topic": topic[:400], "scenes": _int(form.get("scenes"), 2, 8, 6), "style": str(form.get("style", "storybook"))[:20],
            "language": lang_of(request),
            "voice": str(form.get("voice", ""))[:60], "fade": bool(form.get("fade")), "vision": bool(form.get("vision")),
            "room_id": str(form.get("room_id", ""))[:60], "retry": bool(form.get("retry"))}
    res = await api(request).post("/teacher/stories", token=user["token"], json=body, timeout=180)
    ensure_session(request, res)
    if not res.ok:
        key = "video.no_key" if res.status == 503 else None
        return notice(request, tr(request, key) if key else api_message(request, res), "warning")
    return render(request, "partials/story_job.html", job=res.data)


async def story_status(request: Request):
    user = require_user(request, "teacher")
    asset_id = request.path_params["asset_id"]
    res = await api(request).get(f"/teacher/stories/{quote(asset_id, safe='')}", token=user["token"], timeout=120)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    return render(request, "partials/story_job.html", job=res.data)


# ------------------------------------------------------------------ Presentation Studio
async def presentations_page(request: Request):
    user = require_user(request, "teacher")
    proj = await project(request, user)
    params = {"room_id": proj["room"]["room_id"]} if proj["room"] else {}
    decks_res, themes_res = await gather(api(request).get("/teacher/decks", token=user["token"], params=params, timeout=60),
                                         api(request).get("/teacher/themes", token=user["token"]))
    ensure_session(request, decks_res)
    ensure_session(request, themes_res)
    carry = {k: v for k, v in {"asset": request.query_params.get("asset", "")[:40],
                               "theme": request.query_params.get("theme", "")[:40], **proj["q"]}.items() if v}
    keep = {k: v for k, v in carry.items() if k in ("asset", "theme")}
    return page(request, "studios/presentations.html", "presentations", proj, res=decks_res, decks=decks_res.get("decks", []),
                themes_res=themes_res, presets=themes_res.get("presets", []), custom=themes_res.get("custom", []), carry=carry,
                keep=keep)


async def presentation_open(request: Request):
    """From a finished lesson: open (or create) the editable deck of that lesson."""
    user = require_user(request, "teacher")
    sid = request.query_params.get("session", "")[:80]
    res = await api(request).post("/teacher/decks", token=user["token"], json={"session_id": sid}, timeout=60) if sid else None
    if res is not None:
        ensure_session(request, res)
    if res is None or not res.ok:
        flash(request, "warning", tr(request, "deck.no_slides"))
        return redirect(request, "/presentations")
    return redirect(request, f"/presentations/{res.data['deck_id']}")


async def theme_extract(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    upload = form.get("file")
    if upload is None or not getattr(upload, "filename", ""):
        flash(request, "warning", tr(request, "theme.need_file"))
        return redirect(request, "/presentations")
    data = await upload.read(30 * 1024 * 1024 + 1)
    res = await api(request).post("/teacher/themes/extract", token=user["token"], timeout=60,
                                  data={"name": str(form.get("name", "")).strip()[:60]},
                                  files={"file": (upload.filename, data, upload.content_type or "application/octet-stream")})
    _done(request, res, "theme.extracted", {A.INVALID: "theme.bad_file"})
    return redirect(request, "/presentations")


async def deck_page(request: Request):
    user = require_user(request, "teacher")
    proj = await project(request, user)
    deck_id = request.path_params["deck_id"]
    deck_res, themes_res, images_res = await gather(
        api(request).get(f"/teacher/decks/{quote(deck_id, safe='')}", token=user["token"]),
        api(request).get("/teacher/themes", token=user["token"]),
        _assets(request, user, kind="image", status="ready"))
    ensure_session(request, deck_res)
    if deck_res.kind == A.NOT_FOUND:
        raise HTTPException(status_code=404, detail="deck")
    deck = deck_res.data if deck_res.ok else None
    asset_id = request.query_params.get("asset", "")[:40]
    pick = next((a for a in images_res.get("assets", []) if a["asset_id"] == asset_id), None)
    return page(request, "studios/deck.html", "presentations", proj, res=deck_res, deck=deck,
                presets=themes_res.get("presets", []), custom=themes_res.get("custom", []),
                images=images_res.get("assets", []), pick=pick,
                chosen_theme=request.query_params.get("theme", "")[:40])


async def deck_save(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    deck_id = request.path_params["deck_id"]
    current = await api(request).get(f"/teacher/decks/{quote(deck_id, safe='')}", token=user["token"])
    ensure_session(request, current)
    if not current.ok:
        raise HTTPException(status_code=404, detail="deck")
    slides = []
    for i, old in enumerate(current.data["slides"]):
        title = str(form.get(f"title_{i}", old["title"])).strip()
        bullets = [line.strip() for line in str(form.get(f"bullets_{i}", "")).splitlines() if line.strip()]
        slides.append({**{k: old.get(k) for k in ("diagram_steps", "image_asset_id")}, "title": title or old["title"],
                       "bullets": bullets, "notes": str(form.get(f"notes_{i}", "")).strip()})
    body = {"title": str(form.get("title", "")).strip()[:200] or current.data["title"], "slides": slides,
            "theme_id": str(form.get("theme_id", current.data["theme_id"]))[:60]}
    res = await api(request).put(f"/teacher/decks/{quote(deck_id, safe='')}", token=user["token"], json=body)
    _done(request, res, "deck.saved")
    return redirect(request, f"/presentations/{deck_id}")


async def deck_slide_image(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    deck_id, index = request.path_params["deck_id"], request.path_params["index"]
    res = await api(request).post(f"/teacher/decks/{quote(deck_id, safe='')}/slides/{index}/image", token=user["token"],
                                  json={"asset_id": str(form.get("asset_id", ""))[:40]})
    _done(request, res, "deck.image_set")
    return redirect(request, f"/presentations/{deck_id}")


async def deck_slide_visual(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    deck_id, index = request.path_params["deck_id"], request.path_params["index"]
    res = await api(request).post(f"/teacher/decks/{quote(deck_id, safe='')}/slides/{index}/visual", token=user["token"],
                                  json={"prompt": str(form.get("prompt", "")).strip()[:600]}, timeout=120)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, tr(request, "visual.no_key") if res.status == 503 else _msg(request, res), "warning")
    return render(request, "partials/visual_job.html", deck_id=deck_id, task_id=res.data["task_id"], job=None,
                  prompt=res.data.get("prompt", ""), used_llm=res.data.get("used_llm", False))


async def deck_visual_status(request: Request):
    user = require_user(request, "teacher")
    deck_id, task_id = request.path_params["deck_id"], request.path_params["task_id"]
    res = await api(request).get(f"/teacher/visuals/{quote(task_id, safe='')}", token=user["token"], timeout=30)
    ensure_session(request, res)
    job = dict(res.data) if res.ok else {"status": "ERROR", "error": api_message(request, res)}
    response = render(request, "partials/visual_job.html", deck_id=deck_id, task_id=task_id, job=job,
                      prompt=request.query_params.get("prompt", "")[:600], used_llm=False)
    if job.get("status") == "SUCCEEDED":
        response.headers["HX-Refresh"] = "true"   # the slide now has its picture: show it
    return response


async def deck_export(request: Request):
    user = require_user(request, "teacher")
    await check_csrf(request)
    deck_id = request.path_params["deck_id"]
    res = await api(request).post(f"/teacher/decks/{quote(deck_id, safe='')}/export", token=user["token"], timeout=180)
    _done(request, res, "deck.exported")
    return redirect(request, f"/presentations/{deck_id}")


# ------------------------------------------------------------------ My Learning Platform
async def learning_page(request: Request):
    user = require_user(request, "teacher")
    proj = await project(request, user)
    courses_res = await api(request).get("/teacher/courses", token=user["token"], timeout=60)
    ensure_session(request, courses_res)
    asset_id = request.query_params.get("asset", "")[:40]
    pick = None
    if asset_id:
        found = await api(request).get(f"/teacher/assets/{quote(asset_id, safe='')}", token=user["token"])
        pick = found.data if found.ok else None
    cards = [{**c, "stats": course_stats(course_outline(c, _teacher_file(request)))} for c in courses_res.get("courses", [])]
    published = [c for c in cards if c.get("published_room_id")]
    if published:
        sums = await gather(*[api(request).get(f"/teacher/courses/{quote(c['course_id'], safe='')}/progress", token=user["token"])
                              for c in published])
        for c, r in zip(published, sums):
            c["class_progress"] = r.data if r.ok else None
    return page(request, "studios/learning.html", "learning", proj, res=courses_res, courses=cards,
                pick=pick, keep={"asset": asset_id} if pick else {})


def _teacher_file(request: Request):
    def link(a: dict) -> str:
        return u(request, f"/assets/{a['asset_id']}/file") if a.get("file_url") else ""
    return link


async def course_create(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    body = {"title": str(form.get("title", "")).strip(), "description": str(form.get("description", "")).strip()[:1000],
            "room_id": str(form.get("room_id", ""))[:60]}
    if len(body["title"]) < 2:
        flash(request, "warning", tr(request, "course.need_title"))
        return redirect(request, "/learning")
    res = await api(request).post("/teacher/courses", token=user["token"], json=body)
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "course.created"))
        return redirect(request, f"/learning/{res.data['course_id']}")
    flash(request, "warning", api_message(request, res))
    return redirect(request, "/learning")


async def course_page(request: Request):
    user = require_user(request, "teacher")
    proj = await project(request, user)
    cid = request.path_params["course_id"]
    course_res, library_res = await gather(api(request).get(f"/teacher/courses/{quote(cid, safe='')}", token=user["token"]),
                                           _assets(request, user, status="ready"))
    ensure_session(request, course_res)
    if not course_res.ok:
        raise HTTPException(status_code=404, detail="course")
    library = [a for a in library_res.get("assets", []) if a["kind"] not in COURSE_SKIP_KINDS]
    asset_id = request.query_params.get("asset", "")[:40]
    course = course_res.data
    modules = course_outline(course, _teacher_file(request))
    class_progress = None
    if course.get("published_room_id"):
        r = await api(request).get(f"/teacher/courses/{quote(cid, safe='')}/progress", token=user["token"])
        class_progress = r.data if r.ok else None
    return page(request, "studios/course.html", "learning", proj, res=course_res, course=course, library=library,
                pick=next((a for a in library if a["asset_id"] == asset_id), None), modules=modules,
                stats=course_stats(modules), class_progress=class_progress)


async def course_preview(request: Request):
    """The course as a student will see it (the current draft, not the published copy)."""
    user = require_user(request, "teacher")
    proj = await project(request, user)
    cid = request.path_params["course_id"]
    res = await api(request).get(f"/teacher/courses/{quote(cid, safe='')}", token=user["token"])
    ensure_session(request, res)
    if not res.ok:
        raise HTTPException(status_code=404, detail="course")
    modules = course_outline(res.data, _teacher_file(request))
    view = _player(modules, request.query_params.get("item", "")[:120])
    return page(request, "learn/player.html", "learning", proj, course=res.data, modules=modules, stats=course_stats(modules),
                done=set(), percent=0, mode="preview", base=u(request, f"/learning/{cid}/preview"), back=u(request, f"/learning/{cid}"), **view)


def _course_url(cid: str) -> str:
    return f"/learning/{cid}"


async def _course_post(request: Request, path: str, ok_key: str, json: dict | None = None, to: str | None = None):
    user = require_user(request, "teacher")
    await check_csrf(request)
    cid = request.path_params["course_id"]
    res = await api(request).post(f"/teacher/courses/{quote(cid, safe='')}{path}", token=user["token"], json=json or {})
    _done(request, res, ok_key)
    return redirect(request, to or _course_url(cid))


async def course_update(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    cid = request.path_params["course_id"]
    res = await api(request).put(f"/teacher/courses/{quote(cid, safe='')}", token=user["token"],
                                 json={"title": str(form.get("title", "")).strip(), "description": str(form.get("description", "")).strip()[:1000]})
    _done(request, res, "course.saved")
    return redirect(request, _course_url(cid))


async def course_delete(request: Request):
    user = require_user(request, "teacher")
    await check_csrf(request)
    cid = request.path_params["course_id"]
    res = await api(request).request("DELETE", f"/teacher/courses/{quote(cid, safe='')}", token=user["token"])
    _done(request, res, "course.deleted")
    return redirect(request, "/learning")


async def lesson_add(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    cid = request.path_params["course_id"]
    sid = str(form.get("session_id", ""))[:80]
    asset_ids = [str(a)[:40] for a in form.getlist("asset_ids")][:20]
    title = str(form.get("title", "")).strip()
    if sid and not asset_ids:   # a finished lesson brings its own files (slides, quiz, code)
        found = await _assets(request, user, session_id=sid, status="ready")
        asset_ids = [a["asset_id"] for a in found.get("assets", []) if a["kind"] != "theme"][:20]
    if len(title) < 2:
        flash(request, "warning", tr(request, "course.need_title"))
        return redirect(request, _course_url(cid))
    res = await api(request).post(f"/teacher/courses/{quote(cid, safe='')}/lessons", token=user["token"],
                                  json={"title": title[:160], "notes": str(form.get("notes", "")).strip()[:2000],
                                        "session_id": sid, "asset_ids": asset_ids})
    _done(request, res, "course.lesson_added")
    return redirect(request, _course_url(cid))


async def lesson_update(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    cid, lid = request.path_params["course_id"], request.path_params["lesson_id"]
    res = await api(request).put(f"/teacher/courses/{quote(cid, safe='')}/lessons/{quote(lid, safe='')}", token=user["token"],
                                 json={"title": str(form.get("title", "")).strip(), "notes": str(form.get("notes", "")).strip()[:2000]})
    _done(request, res, "course.saved")
    return redirect(request, _course_url(cid))


async def lesson_delete(request: Request):
    user = require_user(request, "teacher")
    await check_csrf(request)
    cid, lid = request.path_params["course_id"], request.path_params["lesson_id"]
    res = await api(request).request("DELETE", f"/teacher/courses/{quote(cid, safe='')}/lessons/{quote(lid, safe='')}", token=user["token"])
    _done(request, res, "course.lesson_removed")
    return redirect(request, _course_url(cid))


async def lesson_move(request: Request):
    form = await request.form()
    direction = "up" if str(form.get("direction")) == "up" else "down"
    return await _course_post(request, f"/lessons/{quote(request.path_params['lesson_id'], safe='')}/move",
                              "course.saved", {"direction": direction})


async def lesson_assets(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    cid, lid = request.path_params["course_id"], request.path_params["lesson_id"]
    body = {"add": str(form.get("add", ""))[:40], "remove": str(form.get("remove", ""))[:40]}
    res = await api(request).post(f"/teacher/courses/{quote(cid, safe='')}/lessons/{quote(lid, safe='')}/assets",
                                  token=user["token"], json=body)
    _done(request, res, "course.saved")
    return redirect(request, _course_url(cid))


async def course_publish(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    cid = request.path_params["course_id"]
    res = await api(request).post(f"/teacher/courses/{quote(cid, safe='')}/publish", token=user["token"],
                                  json={"room_id": str(form.get("room_id", ""))[:60]})
    _done(request, res, "course.publish_done")
    return redirect(request, _course_url(cid))


async def course_unpublish(request: Request):
    return await _course_post(request, "/unpublish", "course.unpublished")


# ------------------------------------------------------------------ learning platform: a course as modules and items
VIDEO_KINDS = ("video", "generated_video")


def course_outline(course: dict, file_url) -> list[dict]:
    """The course as the player shows it: modules (lessons) of items. `file_url(asset)` gives a file's link or ''.
    Item ids match rafeeq/teacher/course_progress.py: '<lesson>:<asset>' and '<lesson>:notes'."""
    from rafeeq.web.common import yt_embed

    modules = []
    for n, lesson in enumerate(course.get("lessons", []), 1):
        items = []
        if (lesson.get("notes") or "").strip():
            items.append({"id": f"{lesson['lesson_id']}:notes", "type": "reading", "title": lesson["title"], "notes": lesson["notes"],
                          "kind": "notes"})
        for a in lesson.get("assets", []):
            src = file_url(a)
            kind = a.get("kind", "")
            if kind in VIDEO_KINDS:
                kind_type = "video" if src else "file"
            elif kind == "source_video":
                kind_type = "youtube" if yt_embed(a.get("source_url", ""), a.get("start_sec"), a.get("end_sec")) else "link"
            elif kind in ("audio", "image"):
                kind_type = kind if src else "file"
            else:
                kind_type = "file"
            items.append({"id": f"{lesson['lesson_id']}:{a['asset_id']}", "type": kind_type, "kind": kind, "title": a["title"],
                          "src": src, "source_url": a.get("source_url", ""), "start": a.get("start_sec"), "end": a.get("end_sec"),
                          "embed": yt_embed(a.get("source_url", ""), a.get("start_sec"), a.get("end_sec")) if kind == "source_video" else ""})
        modules.append({"n": n, "lesson_id": lesson["lesson_id"], "title": lesson["title"], "items": items})
    return modules


def course_stats(modules: list[dict]) -> dict:
    items = [i for m in modules for i in m["items"]]
    return {"modules": len(modules), "items": len(items),
            "videos": sum(1 for i in items if i["type"] in ("video", "youtube")),
            "materials": sum(1 for i in items if i["type"] not in ("video", "youtube")),
            "cover": next((i["src"] for i in items if i["type"] == "image" and i["src"]), "")}


def _player(modules: list[dict], item_id: str) -> dict:
    """The current item and its neighbours (the first item when item_id is unknown)."""
    flat = [i for m in modules for i in m["items"]]
    idx = next((k for k, i in enumerate(flat) if i["id"] == item_id), 0)
    current = flat[idx] if flat else None
    module = next((m for m in modules if current and any(i["id"] == current["id"] for i in m["items"])), None)
    return {"current": current, "module": module, "prev": flat[idx - 1] if flat and idx > 0 else None,
            "next": flat[idx + 1] if flat and idx + 1 < len(flat) else None, "index": idx + 1 if flat else 0, "count": len(flat)}


# ------------------------------------------------------------------ what students see
def _room_file(request: Request, room_id: str, course_id: str):
    def link(a: dict) -> str:
        return u(request, f"/rooms/{room_id}/courses/{course_id}/assets/{a['asset_id']}/file") if a.get("file_url") else ""
    return link


async def _progress(request: Request, user: dict, room_id: str, course_id: str) -> dict:
    res = await api(request).get(f"/rooms/{quote(room_id, safe='')}/courses/{quote(course_id, safe='')}/progress",
                                 token=user["token"], timeout=30)
    return res.data if res.ok else {"done": [], "last": "", "percent": 0, "total": 0}


async def learn_page(request: Request):
    """The room's course catalog: one card per published course, with the student's own progress."""
    user, room, rooms = await load_room(request, request.path_params["room_id"])
    res = await api(request).get(f"/rooms/{quote(room['room_id'], safe='')}/courses", token=user["token"])
    ensure_session(request, res)
    cards = []
    for c in res.get("courses", []):
        modules = course_outline(c, _room_file(request, room["room_id"], c["course_id"]))
        cards.append({**c, "stats": course_stats(modules)})
    if user.get("role") != "teacher" and cards:
        progress = await gather(*[_progress(request, user, room["room_id"], c["course_id"]) for c in cards])
        for c, p in zip(cards, progress):
            c["progress"] = p
    return render(request, "room/learn.html", room=room, rooms=rooms, tab="learn", res=res, courses=cards)


async def learn_course(request: Request):
    """The course player: the outline on one side, the current item, previous / next, mark as complete."""
    user, room, rooms = await load_room(request, request.path_params["room_id"])
    cid = request.path_params["course_id"]
    res = await api(request).get(f"/rooms/{quote(room['room_id'], safe='')}/courses", token=user["token"])
    ensure_session(request, res)
    course = next((c for c in res.get("courses", []) if c["course_id"] == cid), None)
    if course is None:
        raise HTTPException(status_code=404, detail="course")
    modules = course_outline(course, _room_file(request, room["room_id"], cid))
    student = user.get("role") != "teacher"
    progress = await _progress(request, user, room["room_id"], cid) if student else {"done": [], "last": "", "percent": 0}
    item_id = request.query_params.get("item", "")[:120] or progress.get("last", "")
    view = _player(modules, item_id)
    if student and view["current"] and view["current"]["id"] != progress.get("last"):
        await api(request).post(f"/rooms/{quote(room['room_id'], safe='')}/courses/{quote(cid, safe='')}/progress",
                                token=user["token"], json={"item_id": view["current"]["id"], "opened": True}, timeout=30)
    base = u(request, f"/rooms/{room['room_id']}/learn/{cid}")
    return render(request, "learn/player.html", room=room, rooms=rooms, tab="learn", course=course, modules=modules,
                  stats=course_stats(modules), done=set(progress.get("done", [])), percent=progress.get("percent", 0),
                  mode="student" if student else "teacher", base=base, back=u(request, f"/rooms/{room['room_id']}/learn"), **view)


async def learn_progress(request: Request):
    """Mark the current item complete (or not) and go on to the next item."""
    user, room, _ = await load_room(request, request.path_params["room_id"])
    form = await check_csrf(request)
    cid = request.path_params["course_id"]
    item_id = str(form.get("item_id", ""))[:120]
    done = str(form.get("done", "1")) == "1"
    await api(request).post(f"/rooms/{quote(room['room_id'], safe='')}/courses/{quote(cid, safe='')}/progress",
                            token=user["token"], json={"item_id": item_id, "done": done}, timeout=30)
    go = str(form.get("next", "")) or item_id
    return redirect(request, f"/rooms/{room['room_id']}/learn/{cid}", item=go)


async def learn_file(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"])
    p = request.path_params
    res = await api(request).get(f"/rooms/{quote(room['room_id'], safe='')}/courses/{quote(p['course_id'], safe='')}"
                                 f"/assets/{quote(p['asset_id'], safe='')}/file", token=user["token"], timeout=120)
    ensure_session(request, res)
    if not res.ok:
        raise HTTPException(status_code=404, detail="file")
    headers = {"Cache-Control": "private, max-age=600", "X-Content-Type-Options": "nosniff"}
    if request.query_params.get("download"):
        headers["Content-Disposition"] = res.headers.get("content-disposition", "attachment")
    return Response(res.content, media_type=res.headers.get("content-type", "application/octet-stream"), headers=headers)

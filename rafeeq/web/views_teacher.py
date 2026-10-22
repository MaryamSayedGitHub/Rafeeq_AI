"""Teacher workspace inside a room: Overview, Students, Class insights, Lesson studio, Profile."""
from __future__ import annotations

from urllib.parse import quote

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from rafeeq.web import api as A
from rafeeq.web.common import (api, api_message, check_csrf, flash, notice, redirect, render, require_user, tr, url as u)
from rafeeq.web.views_rooms import ensure_session, gather, load_room, topic_names

LEVELS = ("beginner", "intermediate", "advanced")
NEEDS = ("slides", "code", "quiz")


def _join_code_of(res: A.ApiResult) -> str | None:
    return res.get("join_code")


# ---------------------------------------------------------------- overview

def _attention(request: Request, room_id: str, files_res, students_res, report_res, join_code) -> list:
    items = []
    if files_res.ok:
        files = files_res.data.get("files", [])
        if not files:
            items.append({"icon": "upload", "tone": "amber", "text": tr(request, "attn.no_files"),
                          "action": tr(request, "attn.no_files.action"), "href": f"/rooms/{room_id}/materials"})
        processing = sum(1 for f in files if f["status"] in ("uploaded", "indexing"))
        failed = sum(1 for f in files if f["status"] == "failed")
        if failed:
            items.append({"icon": "alert", "tone": "red", "text": tr(request, "attn.failed", n=failed),
                          "action": tr(request, "attn.view_materials"), "href": f"/rooms/{room_id}/materials"})
        if processing:
            items.append({"icon": "refresh", "tone": "blue", "text": tr(request, "attn.processing", n=processing),
                          "action": tr(request, "attn.view_materials"), "href": f"/rooms/{room_id}/materials"})
    if students_res.ok and not students_res.data.get("students"):
        items.append({"icon": "users", "tone": "amber", "text": tr(request, "attn.no_students"),
                      "code": join_code})
    if report_res.ok and report_res.data.get("students_at_risk"):
        n = len(report_res.data["students_at_risk"])
        items.append({"icon": "alert", "tone": "red", "text": tr(request, "attn.at_risk", n=n),
                      "action": tr(request, "attn.view_students"), "href": f"/rooms/{room_id}/students"})
    return items


def _studio_link(room_id: str, topic: str, notes: str) -> str:
    return f"/rooms/{room_id}/studio?from=insights&topic={quote(topic)}&notes={quote(notes)}"


def _next_step(request: Request, room_id: str, files_res, insights_res, report_res, names: dict) -> dict:
    if files_res.ok and not files_res.data.get("files"):
        return {"text": tr(request, "next.upload"), "action": tr(request, "attn.no_files.action"),
                "href": f"/rooms/{room_id}/materials"}
    if insights_res.ok and insights_res.data.get("top_questions"):
        top = max(insights_res.data["top_questions"], key=lambda q: q["count"])
        name = names.get(top["topic_id"], top["topic_id"])
        notes = tr(request, "insights.notes_questions", _iso=False, q=top["example"])
        return {"text": tr(request, "next.review_topic", topic=name, n=top["count"]),
                "action": tr(request, "next.review.action"), "href": _studio_link(room_id, name, notes)}
    if report_res.ok and report_res.data.get("topics"):
        weakest = min(report_res.data["topics"], key=lambda t: t["mastery"])
        if weakest["status"] != "on_track":
            name = names.get(weakest["topic_id"], weakest["topic_id"])
            return {"text": tr(request, "next.weak_topic", topic=name), "action": tr(request, "next.review.action"),
                    "href": _studio_link(room_id, name, "")}
    return {"text": tr(request, "next.wait"), "action": None, "href": None}


async def overview_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    rid, tok = room["room_id"], user["token"]
    files_res, students_res, insights_res, report_res, code_res, settings_res, (topics_res, names) = await gather(
        api(request).get(f"/rooms/{rid}/files", token=tok),
        api(request).get(f"/rooms/{rid}/students", token=tok),
        api(request).get(f"/teacher/rooms/{rid}/insights", token=tok),
        api(request).get(f"/teacher/rooms/{rid}/report", token=tok),
        api(request).get(f"/rooms/{rid}/join-code", token=tok),
        api(request).get(f"/rooms/{rid}/settings", token=tok),
        topic_names(request, user, rid),
    )
    ensure_session(request, files_res)
    join_code = _join_code_of(code_res)
    students = students_res.get("students", [])
    files = files_res.get("files", [])
    stats = {
        "students": len(students) if students_res.ok else None,
        "files_ready": sum(1 for f in files if f["status"] == "ready") if files_res.ok else None,
        "files_total": len(files),
        "questions": sum(s["questions_this_week"] for s in students) if students_res.ok else None,
        "at_risk": len(report_res.data.get("students_at_risk", [])) if report_res.ok else None,
    }
    return render(request, "room/overview.html", room=room, rooms=rooms, tab="overview", stats=stats,
                  join_code=join_code, created=request.query_params.get("created") == "1",
                  attention=_attention(request, rid, files_res, students_res, report_res, join_code),
                  next_step=_next_step(request, rid, files_res, insights_res, report_res, names),
                  report_res=report_res, topic_names=names, settings=settings_res.data if settings_res.ok else None)


async def room_settings_save(request: Request):
    """The teacher chooses whether the tutor may answer from the web in this room (rafeeq/tutor/scope.py)."""
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    mode = str(form.get("web_answers", ""))
    res = await api(request).put(f"/rooms/{room['room_id']}/settings", token=user["token"], json={"web_answers": mode})
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "settings.saved"))
    else:
        flash(request, "error", api_message(request, res))
    return redirect(request, f"/rooms/{room['room_id']}/overview")


# ---------------------------------------------------------------- students

async def students_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    rid, tok = room["room_id"], user["token"]
    students_res, report_res, code_res, (_, names) = await gather(
        api(request).get(f"/rooms/{rid}/students", token=tok),
        api(request).get(f"/teacher/rooms/{rid}/report", token=tok),
        api(request).get(f"/rooms/{rid}/join-code", token=tok),
        topic_names(request, user, rid),
    )
    ensure_session(request, students_res)
    at_risk = [{"name": s["name"], "topics": [names.get(tid, tid) for tid in s["topic_ids"]]}
               for s in report_res.get("students_at_risk", [])]
    return render(request, "room/students.html", room=room, rooms=rooms, tab="students", students_res=students_res,
                  report_res=report_res, at_risk=at_risk, join_code=_join_code_of(code_res), topic_names=names)


async def students_draft(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    text = str(form.get("request", "")).strip()
    if not text:
        return notice(request, tr(request, "assistant.need_text"), "warning")
    res = await api(request).post("/teacher/assistant", token=user["token"], timeout=120,
                                  json={"room_id": room["room_id"], "request": text[:2000]})
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    return render(request, "partials/assistant_draft.html", draft=res.data)


# ---------------------------------------------------------------- insights

async def insights_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    rid, tok = room["room_id"], user["token"]
    insights_res, report_res, (_, names), frequent_res = await gather(
        api(request).get(f"/teacher/rooms/{rid}/insights", token=tok),
        api(request).get(f"/teacher/rooms/{rid}/report", token=tok),
        topic_names(request, user, rid),
        api(request).get(f"/rooms/{rid}/instructor/frequent", token=tok),         # M3: what students ask again and again (anonymous)
    )
    ensure_session(request, insights_res)
    cards = []
    if insights_res.ok:
        mistakes: dict[str, list] = {}
        for m in insights_res.data.get("common_mistakes", []):
            mistakes.setdefault(m["topic_id"], []).append(m["description"])
        for q in sorted(insights_res.data.get("top_questions", []), key=lambda q: -q["count"]):
            name = names.get(q["topic_id"], q["topic_id"])
            notes = [tr(request, "insights.notes_questions", _iso=False, q=q["example"])]
            notes += [tr(request, "insights.notes_mistake", _iso=False, m=m) for m in mistakes.get(q["topic_id"], [])]
            cards.append({"name": name, "count": q["count"], "example": q["example"],
                          "mistakes": mistakes.get(q["topic_id"], []),
                          "href": _studio_link(rid, name, "\n".join(notes))})
    frequent = []
    if frequent_res.ok and isinstance(frequent_res.data, list):
        for f in frequent_res.data:
            name = names.get(f["topic_id"], f["topic_id"])
            note = tr(request, "insights.notes_questions", _iso=False, q=f["question"])
            frequent.append({**f, "name": name, "href": _studio_link(rid, name, note)})
    return render(request, "room/insights.html", room=room, rooms=rooms, tab="insights", insights_res=insights_res,
                  report_res=report_res, cards=cards, topic_names=names, frequent=frequent)


# ---------------------------------------------------------------- studio

async def _profile_defaults(request: Request, user: dict) -> dict:
    res = await api(request).get("/teacher/profile", token=user["token"], params={"teacher_id": user["user_id"]})
    return res.data.get("defaults", {}) if res.ok else {}


async def studio_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    defaults, sessions_res = await gather(
        _profile_defaults(request, user),
        api(request).get("/teacher/sessions", token=user["token"], params={"teacher_id": user["user_id"]}),
    )
    past = [s for s in sessions_res.get("sessions", []) if s.get("room_id") == room["room_id"]]
    q = request.query_params
    prefill = {"topic": q.get("topic", "")[:200], "notes": q.get("notes", "")[:2000],
               "from_insights": q.get("from") == "insights"}
    return render(request, "room/studio.html", room=room, rooms=rooms, tab="studio", defaults=defaults,
                  sessions_res=sessions_res, past=past, prefill=prefill)


def _int(value, lo, hi, default):
    try:
        return min(hi, max(lo, int(value)))
    except (TypeError, ValueError):
        return default


async def studio_generate(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    topic = str(form.get("topic", "")).strip()
    needs = [n for n in form.getlist("needs") if n in NEEDS]
    if len(topic) < 2:
        return notice(request, tr(request, "studio.need_topic"), "warning")
    if not needs:
        return notice(request, tr(request, "studio.need_one"), "warning")
    body = {
        "lecturer_id": user["user_id"],
        "lecturer_name": user["name"],
        "room_id": room["room_id"],
        "topic": topic[:200],
        "course_name": str(form.get("course_name", "")).strip()[:120] or room["name"],
        "needs": needs,
        "duration_minutes": _int(form.get("duration_minutes"), 15, 240, 60),
        "language": str(form.get("language", "Arabic")) if form.get("language") in ("Arabic", "English") else "Arabic",
    }
    level = str(form.get("student_level", ""))
    if level in LEVELS:
        body["student_level"] = level
    if "code" in needs and str(form.get("programming_language", "")).strip():
        body["programming_language"] = str(form.get("programming_language")).strip()[:40]
    if "quiz" in needs:
        body["num_quiz_questions"] = _int(form.get("num_quiz_questions"), 1, 20, 5)
    objectives = [line.strip() for line in str(form.get("learning_objectives", "")).splitlines() if line.strip()]
    if objectives:
        body["learning_objectives"] = objectives[:10]
    notes = str(form.get("notes", "")).strip()
    if notes:
        body["notes"] = notes[:3000]
    res = await api(request).post("/teacher/sessions", token=user["token"], json=body, timeout=300)
    ensure_session(request, res)
    if not res.ok:
        msg = tr(request, "session.no_llm") if res.status == 503 else api_message(request, res)
        return notice(request, msg, "warning")
    return render(request, "partials/session.html", room=room, session=res.data)


async def session_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    res = await api(request).get(f"/teacher/sessions/{request.path_params['session_id']}", token=user["token"])
    ensure_session(request, res)
    if res.kind == A.NOT_FOUND:
        raise HTTPException(status_code=404, detail="session")
    return render(request, "room/session_page.html", room=room, rooms=rooms, tab="studio", res=res,
                  session=res.data if res.ok else None)


async def session_decision(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    decision = str(form.get("decision", ""))
    feedback = str(form.get("feedback", "")).strip()
    if decision not in ("approve", "revise"):
        return notice(request, tr(request, "err.generic"), "warning")
    if decision == "revise" and not feedback:
        return notice(request, tr(request, "session.feedback_required"), "warning")
    body = {"decision": decision}
    if feedback:
        body["feedback"] = feedback[:2000]
    sid = request.path_params["session_id"]
    res = await api(request).post(f"/teacher/sessions/{sid}/decision", token=user["token"], json=body, timeout=300)
    ensure_session(request, res)
    if res.kind == A.CONFLICT:
        return notice(request, tr(request, "session.conflict"), "warning")
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    return render(request, "partials/session.html", room=room, session=res.data)


async def session_file(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    sid, name = request.path_params["session_id"], request.path_params["name"]
    res = await api(request).get(f"/teacher/sessions/{quote(sid, safe='')}/files/{quote(name, safe='')}",
                                 token=user["token"], timeout=120)
    ensure_session(request, res)
    if not res.ok:
        raise HTTPException(status_code=404, detail="file")
    safe = name.replace('"', "")
    return Response(res.content, media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(safe, safe='')}"})


# ---------------------------------------------------------------- image upload helpers (M5)
# Shared by the avatar upload and the Visual & Media Studio (rafeeq/web/views_studios.py).

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def _image_too_large(request: Request) -> bool:
    try:
        return int(request.headers.get("content-length", "0")) > MAX_IMAGE_BYTES + 1024 * 1024
    except ValueError:
        return False


async def _image_part(upload) -> tuple | None | bool:
    """None = no image chosen, False = not acceptable, else an httpx file tuple."""
    if upload is None or not getattr(upload, "filename", ""):
        return None
    if (upload.content_type or "") not in IMAGE_TYPES:
        return False
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        return False
    return (upload.filename, data, upload.content_type)


LIBRARY_KINDS = ("pptx", "code", "quiz", "image", "source_video", "generated_video", "video", "audio", "theme")


async def library_page(request: Request):
    """My Library: the hub. Every project (lessons, presentations, courses) and every asset this teacher
    saved or generated in any studio, with filters by type and room and actions to use an item elsewhere."""
    user = require_user(request, "teacher")
    kind = request.query_params.get("kind", "")
    kind = kind if kind in LIBRARY_KINDS else ""
    room_filter = request.query_params.get("room", "")[:60]
    params = {k: v for k, v in {"kind": kind, "room_id": room_filter}.items() if v}
    assets_res, rooms_res, sessions_res, decks_res, courses_res = await gather(
        api(request).get("/teacher/assets", token=user["token"], params=params, timeout=60),
        api(request).get("/rooms", token=user["token"]),
        api(request).get("/teacher/sessions", token=user["token"], params={"teacher_id": user["user_id"]}),
        api(request).get("/teacher/decks", token=user["token"], timeout=60),
        api(request).get("/teacher/courses", token=user["token"], timeout=60))
    for res in (assets_res, rooms_res):
        ensure_session(request, res)
    rooms = rooms_res.get("rooms", [])
    return render(request, "library.html", res=assets_res, assets=assets_res.get("assets", []), rooms=rooms, room=None,
                  tab="library", kind=kind, room_filter=room_filter, kinds=LIBRARY_KINDS,
                  room_names={r["room_id"]: r["name"] for r in rooms},
                  lessons=sessions_res.get("sessions", [])[:8], decks=decks_res.get("decks", [])[:8],
                  courses=courses_res.get("courses", [])[:8])


async def asset_file(request: Request):
    """Streams one of the teacher's stored files to the browser (?download=1 saves it instead of showing it)."""
    user = require_user(request, "teacher")
    res = await api(request).get(f"/teacher/assets/{quote(request.path_params['asset_id'], safe='')}/file",
                                 token=user["token"], timeout=60)
    if not res.ok:
        raise HTTPException(status_code=404, detail="asset")
    headers = {"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"}
    if request.query_params.get("download"):
        headers["Content-Disposition"] = res.headers.get("content-disposition", "attachment")
    return Response(res.content, media_type=res.headers.get("content-type", "application/octet-stream"), headers=headers)


# ---------------------------------------------------------------- profile

async def profile_page(request: Request):
    user = require_user(request, "teacher")
    res = await api(request).get("/teacher/profile", token=user["token"], params={"teacher_id": user["user_id"]})
    ensure_session(request, res)
    return render(request, "profile.html", res=res, profile=res.data if res.ok else None,
                  is_new=res.kind == A.NOT_FOUND, rooms=[], room=None, tab="profile")


async def profile_save(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    body = {
        "teacher_id": user["user_id"],
        "name": str(form.get("name", "")).strip()[:80] or user["name"],
        "title": str(form.get("title", "")).strip()[:20],
        "subjects": [s.strip() for s in str(form.get("subjects", "")).replace("،", ",").split(",") if s.strip()][:10],
        "bio": str(form.get("bio", "")).strip()[:600],
        "defaults": {
            "language": str(form.get("language", "Arabic")) if form.get("language") in ("Arabic", "English") else "Arabic",
            "student_level": str(form.get("student_level")) if form.get("student_level") in LEVELS else "intermediate",
            "programming_language": str(form.get("programming_language", "")).strip()[:40] or "python",
            "duration_minutes": _int(form.get("duration_minutes"), 15, 240, 60),
        },
    }
    res = await api(request).put("/teacher/profile", token=user["token"], json=body)
    ensure_session(request, res)
    flash(request, "success" if res.ok else "error", tr(request, "profile.saved") if res.ok else api_message(request, res))
    return redirect(request, "/profile")


async def profile_avatar_save(request: Request):
    """The presenter image used as the first frame of the teacher's animated videos."""
    user = require_user(request, "teacher")
    if _image_too_large(request):
        flash(request, "warning", tr(request, "video.bad_image"))
        return redirect(request, "/profile")
    form = await check_csrf(request)
    image = await _image_part(form.get("image"))
    if not image:
        flash(request, "warning", tr(request, "video.bad_image"))
        return redirect(request, "/profile")
    res = await api(request).post("/teacher/avatar", token=user["token"], data={"teacher_id": user["user_id"]},
                                  files={"image": image}, timeout=60)
    ensure_session(request, res)
    flash(request, "success" if res.ok else "error",
          tr(request, "profile.avatar_saved") if res.ok else api_message(request, res))
    return redirect(request, "/profile")


async def profile_avatar(request: Request):
    user = require_user(request, "teacher")
    res = await api(request).get(f"/teacher/avatar/{quote(user['user_id'], safe='')}", token=user["token"])
    if not res.ok:
        raise HTTPException(status_code=404, detail="avatar")
    return Response(res.content, media_type=res.headers.get("content-type", "image/jpeg"),
                    headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- parent report

async def parent_report_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    uid = request.path_params["user_id"]
    res = await api(request).get(f"/teacher/rooms/{room['room_id']}/students/{quote(uid, safe='')}/parent-report",
                                 token=user["token"])
    ensure_session(request, res)
    share_url = ""
    if res.ok and res.data.get("share_token"):
        share_url = f"{request.url.scheme}://{request.url.netloc}" + u(request, f"/p/{res.data['share_token']}")
    return render(request, "room/parent_report.html", room=room, rooms=rooms, tab="students",
                  report_res=res, share_url=share_url)


async def parent_report_shared(request: Request):
    res = await api(request).get(f"/parent-report/{quote(request.path_params['token'], safe='')}")
    return render(request, "parent/shared.html", report=res.data if res.ok else None,
                  status=200 if res.ok else 404)


# ---------------------------------------------------------------- inbox

async def inbox_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    res = await api(request).get(f"/rooms/{room['room_id']}/inbox", token=user["token"])
    ensure_session(request, res)
    return render(request, "room/inbox.html", room=room, rooms=rooms, tab="inbox", inbox_res=res)


async def inbox_draft(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    await check_csrf(request)
    qid = request.path_params["question_id"]
    res = await api(request).post(f"/rooms/{room['room_id']}/inbox/{quote(qid, safe='')}/draft",
                                  token=user["token"], timeout=90)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    return render(request, "partials/inbox_draft.html", question_id=qid, draft=res.data["draft_reply"])


async def inbox_answer(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    text = str(form.get("answer", "")).strip()
    qid = request.path_params["question_id"]
    if text:
        res = await api(request).post(f"/rooms/{room['room_id']}/inbox/{quote(qid, safe='')}/answer",
                                      token=user["token"], json={"answer": text[:4000]})
        ensure_session(request, res)
        flash(request, "success" if res.ok else "error", tr(request, "inbox.sent") if res.ok else api_message(request, res))
    return redirect(request, f"/rooms/{room['room_id']}/inbox")

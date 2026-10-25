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
    students = students_res.data.get("students", []) if students_res.ok else []
    if report_res.ok:
        n = len(report_res.data.get("students_at_risk", []))
    else:   # the class report did not load: the students list still carries each student's level
        n = sum(1 for s in students if s.get("status") == "struggling")
    if n:
        items.append({"icon": "alert", "tone": "red", "text": tr(request, "attn.at_risk", n=n),
                      "action": tr(request, "attn.view_students"), "href": f"/rooms/{room_id}/students"})
    no_data = sum(1 for s in students if s.get("status") is None)
    if students and no_data:
        items.append({"icon": "quiz", "tone": "blue", "text": tr(request, "attn.no_data", n=no_data),
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


def _student_charts(students: list) -> tuple[dict, list]:
    """Data for the overview charts, from the students list (it carries each student's level and questions)."""
    levels = {"on_track": 0, "watch": 0, "struggling": 0, "none": 0}
    for s in students:
        levels[s.get("status") if s.get("status") in levels else "none"] += 1
    bars = [{"name": s["name"], "mastery": s.get("mastery"), "status": s.get("status"),
             "questions": s.get("questions_this_week", 0)} for s in students]
    bars.sort(key=lambda b: (b["mastery"] is None, b["mastery"] if b["mastery"] is not None else 0, b["name"]))
    return levels, bars


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
    levels, bars = _student_charts(students) if students_res.ok else ({}, [])
    return render(request, "room/overview.html", room=room, rooms=rooms, tab="overview", stats=stats,
                  levels=levels, bars=bars, max_questions=max([b["questions"] for b in bars] + [1]),
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
    if not report_res.ok and students_res.ok:   # the students list carries each level: still show who needs support
        at_risk = [{"name": s["name"], "topics": []} for s in students_res.data.get("students", [])
                   if s.get("status") == "struggling"]
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

def _public_base(request: Request) -> str:
    """Where parents open the report (ends in /p/). PUBLIC_BASE_URL wins over the address the teacher is using."""
    import os

    configured = os.environ.get("PUBLIC_BASE_URL", "").strip().rstrip("/")
    origin = configured or f"{request.url.scheme}://{request.url.netloc}"
    return origin + u(request, "/p/")


async def parent_report_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    uid = request.path_params["user_id"]
    res = await api(request).get(f"/teacher/rooms/{room['room_id']}/students/{quote(uid, safe='')}/parent-report",
                                 token=user["token"])
    ensure_session(request, res)
    share_url = ""
    delivery = {}
    if res.ok and res.data.get("share_token"):
        share_url = _public_base(request) + res.data["share_token"]
        d = await api(request).get(f"/teacher/rooms/{room['room_id']}/students/{quote(uid, safe='')}/parent-delivery",
                                   token=user["token"], params={"base": _public_base(request)})
        delivery = d.data if d.ok and isinstance(d.data, dict) else {}
    return render(request, "room/parent_report.html", room=room, rooms=rooms, tab="students",
                  report_res=res, share_url=share_url, delivery=delivery, student_id=uid)


async def parent_contact_save(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    uid = request.path_params["user_id"]
    form = await check_csrf(request)
    res = await api(request).put(f"/teacher/rooms/{room['room_id']}/students/{quote(uid, safe='')}/parent-contact",
                                 token=user["token"],
                                 json={"email": str(form.get("email", "")), "phone": str(form.get("phone", ""))})
    ensure_session(request, res)
    flash(request, "success" if res.ok else "error",
          tr(request, "parent.contact_saved") if res.ok else tr(request, "parent.contact_bad"))
    return redirect(request, f"/rooms/{room['room_id']}/students/{uid}/report")


async def parent_report_send_email(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    uid = request.path_params["user_id"]
    form = await check_csrf(request)
    res = await api(request).post(
        f"/teacher/rooms/{room['room_id']}/students/{quote(uid, safe='')}/parent-report/send-email",
        token=user["token"], json={"base": _public_base(request), "subject": str(form.get("subject", "")),
                                   "text": str(form.get("text", ""))})
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "parent.email_sent"))
    else:
        flash(request, "error", res.message or api_message(request, res))
    return redirect(request, f"/rooms/{room['room_id']}/students/{uid}/report")


# ---------------------------------------------------------------- all parents at once (import, review, approve, send)

def _batch_path(room_id: str, tail: str = "") -> str:
    return f"/teacher/rooms/{room_id}/parent-batch{tail}"


def _names(request: Request, items: list, limit: int = 6) -> str:
    out = []
    for it in items[:limit]:
        why = it.get("reason", "")
        label = it.get("name", "")
        out.append(f"{label} ({tr(request, 'batch.reason.' + why)})" if why else label)
    more = len(items) - limit
    return "، ".join(out) + (f" +{more}" if more > 0 else "")


async def parent_batch_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "teacher")
    res = await api(request).get(_batch_path(room["room_id"]), token=user["token"])
    ensure_session(request, res)
    data = res.data if res.ok and isinstance(res.data, dict) else {"drafts": [], "summary": {}, "email_ready": False}
    share_base = _public_base(request)
    return render(request, "room/parent_batch.html", room=room, rooms=rooms, tab="students", batch_res=res,
                  batch=data, localhost=share_base.startswith(("http://127.0.0.1", "http://localhost")))


async def _batch_post(request: Request, tail: str, json: dict | None = None, method: str = "post"):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    call = getattr(api(request), method)
    res = await call(_batch_path(room["room_id"], tail), token=user["token"], json=json if json is not None else {})
    ensure_session(request, res)
    return room, form, res


def _back(request: Request, room: dict):
    return redirect(request, f"/rooms/{room['room_id']}/parent-batch")


async def parent_batch_template(request: Request):
    from starlette.responses import Response as FileResponse_

    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    res = await api(request).get(_batch_path(room["room_id"], "/template"), token=user["token"], timeout=60)
    ensure_session(request, res)
    if not res.ok:
        flash(request, "error", res.message or api_message(request, res))
        return _back(request, room)
    return FileResponse_(res.content, media_type=res.headers.get(
        "content-type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        headers={"Content-Disposition": 'attachment; filename="rafeeq-parents.xlsx"', "X-Content-Type-Options": "nosniff"})


async def parent_batch_import(request: Request):
    import base64

    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    sheet = form.get("sheet")
    raw = await sheet.read() if hasattr(sheet, "read") else b""
    if not raw:
        flash(request, "error", tr(request, "batch.f.upload"))
        return _back(request, room)
    res = await api(request).post(_batch_path(room["room_id"], "/import"), token=user["token"],
                                  json={"filename": getattr(sheet, "filename", "") or "", "data_b64": base64.b64encode(raw).decode()})
    ensure_session(request, res)
    if not res.ok:
        flash(request, "error", res.message or api_message(request, res))
        return _back(request, room)
    d = res.data
    flash(request, "success", tr(request, "batch.f.imported", n=d["saved"]))
    if d["unmatched"]:
        flash(request, "warning", tr(request, "batch.f.unmatched", names=_names(request, d["unmatched"])))
    if d["invalid"]:
        flash(request, "warning", tr(request, "batch.f.invalid", names=_names(request, d["invalid"])))
    return _back(request, room)


async def parent_batch_prepare(request: Request):
    room, _, res = await _batch_post(request, "/prepare", {"base": _public_base(request)})
    if res.ok:
        flash(request, "success", tr(request, "batch.f.prepared", n=res.data["prepared"]))
        if res.data["no_contact"]:
            flash(request, "warning", tr(request, "batch.f.no_contact", names="، ".join(res.data["no_contact"][:6])))
    else:
        flash(request, "error", res.message or api_message(request, res))
    return _back(request, room)


async def parent_batch_approve_all(request: Request):
    room, _, res = await _batch_post(request, "/approve-all")
    flash(request, "success" if res.ok else "error",
          tr(request, "batch.f.approved", n=res.data["approved"]) if res.ok else (res.message or api_message(request, res)))
    return _back(request, room)


async def parent_batch_send_emails(request: Request):
    room, _, res = await _batch_post(request, "/send-emails")
    if res.ok:
        flash(request, "success", tr(request, "batch.f.sent", n=res.data["sent"]))
        if res.data["failed"]:
            flash(request, "error", tr(request, "batch.f.failed", names="، ".join(
                f"{f['name']}: {f['error']}" for f in res.data["failed"][:3])))
    else:
        flash(request, "error", res.message or api_message(request, res))
    return _back(request, room)


async def parent_batch_edit(request: Request):
    uid = quote(request.path_params["user_id"], safe="")
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    res = await api(request).put(_batch_path(room["room_id"], "/" + uid), token=user["token"],
                                 json={"subject": str(form.get("subject", "")), "text": str(form.get("text", ""))})
    ensure_session(request, res)
    flash(request, "success" if res.ok else "error",
          tr(request, "batch.f.saved") if res.ok else (res.message or api_message(request, res)))
    return _back(request, room)


async def parent_batch_status(request: Request):
    uid = quote(request.path_params["user_id"], safe="")
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    res = await api(request).post(_batch_path(room["room_id"], f"/{uid}/status"), token=user["token"],
                                  json={"status": str(form.get("status", ""))})
    ensure_session(request, res)
    if not res.ok:
        flash(request, "error", res.message or api_message(request, res))
    return _back(request, room)


async def parent_batch_wa_done(request: Request):
    uid = quote(request.path_params["user_id"], safe="")
    room, _, res = await _batch_post(request, f"/{uid}/whatsapp-sent")
    if not res.ok:
        flash(request, "error", res.message or api_message(request, res))
    return _back(request, room)


# ---------------------------------------------------------------- the student gives the parent's contact

async def my_parent_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "student")
    res = await api(request).get(f"/rooms/{room['room_id']}/my-parent-contact", token=user["token"])
    ensure_session(request, res)
    return render(request, "room/my_parent.html", room=room, rooms=rooms, tab="study",
                  contact=res.data if res.ok and isinstance(res.data, dict) else {"email": "", "phone": ""})


async def my_parent_save(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    form = await check_csrf(request)
    res = await api(request).put(f"/rooms/{room['room_id']}/my-parent-contact", token=user["token"],
                                 json={"email": str(form.get("email", "")), "phone": str(form.get("phone", ""))})
    ensure_session(request, res)
    flash(request, "success" if res.ok else "error",
          tr(request, "myparent.saved") if res.ok else tr(request, "parent.contact_bad"))
    return redirect(request, f"/rooms/{room['room_id']}/my-parent")


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

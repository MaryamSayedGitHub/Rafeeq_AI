"""Student workspace inside a room: Ask Rafeeq (Copilot + photo), Practice, Progress.

Materials and the source viewer are shared with teachers and live here too.
"""
from __future__ import annotations

import json
import secrets
from urllib.parse import quote

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response

from rafeeq.web import api as A
from rafeeq.web.common import api, api_message, check_csrf, flash, notice, redirect, render, tr
from rafeeq.web.i18n import strip_isolates
from rafeeq.web.views_rooms import ensure_session, gather, load_room, topic_names
from rafeeq.web.views_tutor import explain_turn

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
MAX_BYTES = 25 * 1024 * 1024


def too_large(request: Request) -> bool:
    """Reject before the multipart body is parsed (and spooled to disk)."""
    try:
        return int(request.headers.get("content-length", "0")) > MAX_BYTES + 1024 * 1024
    except ValueError:
        return False


# ---------------------------------------------------------------- Ask Rafeeq

async def study_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "student")
    inbox_res = await api(request).get(f"/rooms/{room['room_id']}/inbox", token=user["token"])
    my_questions = inbox_res.data.get("questions", [])[:6] if inbox_res.ok else []
    return render(request, "room/study.html", room=room, rooms=rooms, tab="study", my_questions=my_questions,
                  prefill=strip_isolates(request.query_params.get("q", ""))[:500])


async def study_message(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    form = await check_csrf(request)
    mode = str(form.get("mode", "auto"))
    mode = mode if mode in ("auto", "ask", "check_work") else "auto"
    file_id = str(form.get("file_id", "")).strip() or None
    text = strip_isolates(str(form.get("text", ""))).strip()
    question = strip_isolates(str(form.get("question", ""))).strip()

    explain_text = ""
    if file_id and mode == "ask":
        shown = question or tr(request, "snap.default_question")
        explain_text = f"{question}\n{text}".strip() if question else text     # what the photo says (+ what they asked), not "explain the photo"
        text = f"{shown}\n\n{text}" if text else shown
    elif file_id and mode == "check_work":
        shown = tr(request, "snap.check")
    else:
        shown = text
    if not text:
        return notice(request, tr(request, "error.empty_question"), "warning")

    if file_id and mode == "ask":                                  # explain what is in the photo: the material first, then the web
        turn = await explain_turn(request, user, room, shown, explain_text or text)
        if turn is not None:
            return turn
    body = {"room_id": room["room_id"], "text": text[:4000], "mode": mode}
    if file_id:
        body["file_id"] = file_id
    res = await api(request).post("/copilot/message", token=user["token"], json=body, timeout=120)
    ensure_session(request, res)
    names: dict = {}
    if res.ok and res.data.get("route") in ("quiz", "plan"):
        _, names = await topic_names(request, user, room["room_id"])
    return render(request, "partials/chat_turn.html", room=room, shown=shown, file_id=file_id, res=res, turn_id=secrets.token_hex(4),
                  reply=res.data if res.ok else None, topic_names=names,
                  error=None if res.ok else api_message(request, res))


async def study_photo(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    if too_large(request):
        return notice(request, tr(request, "materials.too_big"), "warning")
    form = await check_csrf(request)
    photo = form.get("photo")
    if photo is None or not getattr(photo, "filename", ""):
        return notice(request, tr(request, "snap.choose_file"), "warning")
    if (photo.content_type or "").lower() not in IMAGE_TYPES:
        return notice(request, tr(request, "snap.not_image"), "warning")
    data = await photo.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return notice(request, tr(request, "materials.too_big"), "warning")
    res = await api(request).post("/snap", token=user["token"], data={"room_id": room["room_id"]},
                                  files={"photo": (photo.filename, data, photo.content_type)}, timeout=90)
    ensure_session(request, res)
    if not res.ok:
        key = "snap.not_image" if res.kind == A.INVALID else "snap.reader_unavailable" if res.kind == A.UNAVAILABLE else None
        return notice(request, tr(request, key) if key else api_message(request, res), "warning")
    return render(request, "partials/snap_card.html", room=room, snap=res.data)


# ---------------------------------------------------------------- materials (shared)

async def _materials_data(request: Request, user: dict, room_id: str):
    files_res, (topics_res, _) = await gather(
        api(request).get(f"/rooms/{room_id}/files", token=user["token"]),
        topic_names(request, user, room_id),
    )
    ensure_session(request, files_res)
    return files_res, topics_res


async def materials_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"])
    files_res, topics_res = await _materials_data(request, user, room["room_id"])
    return render(request, "room/materials.html", room=room, rooms=rooms, tab="materials",
                  files_res=files_res, topics_res=topics_res)


async def materials_table(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"])
    files_res = await api(request).get(f"/rooms/{room['room_id']}/files", token=user["token"])
    ensure_session(request, files_res)
    return render(request, "partials/files_table.html", room=room, files_res=files_res)


async def materials_upload(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    if too_large(request):
        flash(request, "warning", tr(request, "materials.too_big"))
        return redirect(request, f"/rooms/{room['room_id']}/materials")
    form = await check_csrf(request)
    upload = form.get("file")
    if upload is None or not getattr(upload, "filename", ""):
        flash(request, "warning", tr(request, "materials.no_file"))
        return redirect(request, f"/rooms/{room['room_id']}/materials")
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        flash(request, "warning", tr(request, "materials.too_big"))
        return redirect(request, f"/rooms/{room['room_id']}/materials")
    res = await api(request).post(f"/rooms/{room['room_id']}/files", token=user["token"],
                                  files={"file": (upload.filename, data, upload.content_type or "application/octet-stream")},
                                  timeout=120)
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "materials.uploaded", name=upload.filename))
    else:
        flash(request, "error", api_message(request, res))
    return redirect(request, f"/rooms/{room['room_id']}/materials")


async def materials_download(request: Request):
    """The teacher's material, saved by the teacher or any student of the room."""
    user, room, _ = await load_room(request, request.path_params["room_id"])
    res = await api(request).get(f"/rooms/{room['room_id']}/files/{quote(request.path_params['file_id'], safe='')}/download",
                                 token=user["token"], timeout=60)
    if not res.ok:
        raise HTTPException(status_code=404, detail="file")
    return Response(res.content, media_type=res.headers.get("content-type", "application/octet-stream"),
                    headers={"Content-Disposition": res.headers.get("content-disposition", "attachment"), "X-Content-Type-Options": "nosniff"})


async def room_rename(request: Request):
    """The teacher renames the room; the students' pages read the name from the room, so they show the new one at once."""
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    form = await check_csrf(request)
    name = strip_isolates(str(form.get("name", ""))).strip()
    back = str(form.get("next", "")) if str(form.get("next", "")).startswith(f"/rooms/{room['room_id']}/") else f"/rooms/{room['room_id']}/overview"
    if not name:
        flash(request, "warning", tr(request, "room.rename_empty"))
        return redirect(request, back)
    res = await api(request).put(f"/rooms/{room['room_id']}", token=user["token"], json={"name": name[:100]})
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "room.renamed"))
    else:
        flash(request, "error", api_message(request, res))
    return redirect(request, back)


async def materials_delete(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "teacher")
    await check_csrf(request)
    file_id = request.path_params["file_id"]
    res = await api(request).delete(f"/rooms/{room['room_id']}/files/{quote(file_id, safe='')}", token=user["token"])
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "materials.deleted"))
    else:
        flash(request, "error", api_message(request, res))
    return redirect(request, f"/rooms/{room['room_id']}/materials")


async def source_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"])
    file_name = request.query_params.get("file_name", "")
    try:
        page = max(1, int(request.query_params.get("page", "1")))
    except ValueError:
        page = 1
    # The contract declares room_id here with the path parameter component but the path has no
    # {room_id}; it is sent as a query parameter (see docs/frontend/HANDOFF.md, contract issues).
    res = await api(request).get("/sources/page", token=user["token"],
                                 params={"room_id": room["room_id"], "file_name": file_name, "page": page})
    ensure_session(request, res)
    back = f"/rooms/{room['room_id']}/" + ("study" if user["role"] == "student" else "materials")
    return render(request, "room/source.html", room=room, rooms=rooms, tab="materials", res=res,
                  file_name=file_name, page=page, back=back)


# ---------------------------------------------------------------- practice

async def practice_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "student")
    topics_res, _ = await topic_names(request, user, room["room_id"])
    return render(request, "room/practice.html", room=room, rooms=rooms, tab="practice", topics_res=topics_res,
                  selected=request.query_params.get("topic", ""))


async def practice_quiz(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    form = await check_csrf(request)
    topic_id = str(form.get("topic_id", "")).strip()
    try:
        count = min(10, max(3, int(form.get("count", 5))))
    except ValueError:
        count = 5
    res = await api(request).post("/quiz", token=user["token"], timeout=120,
                                  json={"room_id": room["room_id"], "topic_id": topic_id, "count": count})
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    return render(request, "partials/quiz_form.html", room=room, quiz=res.data)


async def practice_submit(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    form = await check_csrf(request)
    quiz_id = str(form.get("quiz_id", ""))
    try:
        n = min(20, max(1, int(form.get("n", 0))))
    except ValueError:
        return notice(request, tr(request, "err.generic"), "warning")
    answers = [str(form.get(f"a_{i}", "")) for i in range(n)]
    questions = [str(form.get(f"q_{i}", "")) for i in range(n)]
    if any(not a for a in answers):
        return notice(request, tr(request, "practice.answer_all"), "warning")
    return await _submit_quiz(request, user, room, quiz_id, answers, questions, str(form.get("topic_id", "")))


async def _submit_quiz(request: Request, user: dict, room: dict, quiz_id: str, answers: list[str], questions: list[str], topic_id: str):
    n = len(answers)
    res, (_, names) = await gather(
        api(request).post(f"/quiz/{quiz_id}/submit", token=user["token"], json={"answers": answers}),
        topic_names(request, user, room["room_id"]),
    )
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    rows = [{"q": q, "a": a, "e": e} for q, a, e in zip(questions, answers, res.data.get("explanations", []) + [""] * n)]
    return render(request, "partials/quiz_result.html", room=room, result=res.data, rows=rows,
                  topic_names=names, topic_id=topic_id, quiz_id=quiz_id)


async def practice_pdf(request: Request):
    """The quiz as a printable PDF (questions + bubble answer sheet), or, after submitting, the score report PDF."""
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    kind = request.path_params["kind"]
    if kind not in ("sheet", "report"):
        raise HTTPException(status_code=404)
    path = "print.pdf" if kind == "sheet" else "report.pdf"
    res = await api(request).get(f"/quiz/{quote(request.path_params['quiz_id'], safe='')}/{path}", token=user["token"],
                                 params={"room_id": room["room_id"]}, timeout=60)
    ensure_session(request, res)
    if not res.ok:
        flash(request, "warning", api_message(request, res))
        return redirect(request, f"/rooms/{room['room_id']}/practice")
    return Response(res.content, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="quiz-{kind}.pdf"', "X-Content-Type-Options": "nosniff"})


async def practice_grade(request: Request):
    """The student solved the printed sheet and uploads its photo: M4 reads the marks. Clear marks are submitted and scored;
    unclear ones come back as the quiz form, filled in, so the student fixes them first (nothing is saved until the submit)."""
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    if too_large(request):
        return notice(request, tr(request, "materials.too_big"), "warning")
    form = await check_csrf(request)
    quiz_id = request.path_params["quiz_id"]
    photo = form.get("photo")
    if photo is None or not getattr(photo, "filename", ""):
        return notice(request, tr(request, "snap.choose_file"), "warning")
    if (photo.content_type or "").lower() not in IMAGE_TYPES:
        return notice(request, tr(request, "snap.not_image"), "warning")
    data = await photo.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        return notice(request, tr(request, "materials.too_big"), "warning")
    try:
        questions = json.loads(str(form.get("quiz_json", "[]")))
        quiz = [{"text": str(q["text"]), "choices": [str(c) for c in q["choices"]]} for q in questions][:20]
    except (ValueError, KeyError, TypeError):
        return notice(request, tr(request, "err.generic"), "warning")
    topic_id = str(form.get("topic_id", ""))
    res = await api(request).post(f"/quizzes/{quote(quiz_id, safe='')}/grade-sheet", token=user["token"],
                                  files={"photo": (photo.filename, data, photo.content_type)}, timeout=90)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    detected = res.data.get("answers", [])
    if len(detected) != len(quiz):
        return notice(request, tr(request, "err.generic"), "warning")
    picked = [q["choices"][a["choice"]] if isinstance(a.get("choice"), int) and 0 <= a["choice"] < len(q["choices"]) else "" for q, a in zip(quiz, detected)]
    if res.data.get("needs_review") or not all(picked):
        flagged = {i for i, a in enumerate(detected) if a["item_id"] in res.data.get("needs_review", []) or not picked[i]}
        return render(request, "partials/quiz_form.html", room=room, quiz={"quiz_id": quiz_id, "topic_id": topic_id, "questions": quiz},
                      picked=picked, flagged=flagged, review_note=True)
    return await _submit_quiz(request, user, room, quiz_id, picked, [q["text"] for q in quiz], topic_id)


# ---------------------------------------------------------------- progress

async def progress_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "student")
    params = {"room_id": room["room_id"]}
    scores_res, history_res, (topics_res, names) = await gather(
        api(request).get(f"/rooms/{room['room_id']}/topic-scores", token=user["token"]),
        api(request).get(f"/rooms/{room['room_id']}/quiz-history", token=user["token"]),
        topic_names(request, user, room["room_id"]),
    )
    ensure_session(request, scores_res)
    scores = scores_res.data.get("topics", []) if scores_res.ok else []
    return render(request, "room/progress.html", room=room, rooms=rooms, tab="progress",
                  scores_res=scores_res, scores=scores, weak=[s for s in scores if s["weak"]],
                  history_res=history_res, topic_names=names)


async def quiz_review_page(request: Request):
    """One past quiz: every question with my answer, the right one, and why (the wrong ones first)."""
    user, room, rooms = await load_room(request, request.path_params["room_id"], "student")
    res, (_, names) = await gather(
        api(request).get(f"/quiz/{quote(request.path_params['quiz_id'], safe='')}/review", token=user["token"]),
        topic_names(request, user, room["room_id"]),
    )
    ensure_session(request, res)
    if res.kind == A.NOT_FOUND or res.kind == A.FORBIDDEN:
        raise HTTPException(status_code=404, detail="quiz")
    return render(request, "room/quiz_review.html", room=room, rooms=rooms, tab="progress", res=res, review=res.data if res.ok else None,
                  topic_names=names)


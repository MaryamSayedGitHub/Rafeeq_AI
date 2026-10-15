"""The tutor agent inside Ask Rafeeq (M3, under the Copilot): Level 2 study files, quizzes on the answer and on a study file, summary, flashcards,
"explain it again", the three hints and the solution, and the room's study files.

The page the student types in is M1's (views_student.study_message -> POST /copilot/message). Everything the answer offers next goes through ONE route,
POST /rooms/{room_id}/tutor/act (contracts/openapi.yaml, tag M3 tutor), so the UI only uses the contract like the rest of the app.
Only the study-file download and the file pages read the saved file directly (they are plain files, not agent work).
"""
from __future__ import annotations

import html as htmllib
import re
import secrets

from starlette.concurrency import run_in_threadpool

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from rafeeq.tutor import chats, events, export, memory, study_store
from rafeeq.web.common import api, api_message, check_csrf, current_user, flash, lang_of, notice, redirect, render, tr
from rafeeq.web.i18n import strip_isolates
from rafeeq.web.i18n import strip_isolates as _strip
from rafeeq.web.views_rooms import ensure_session, load_room

ACTIONS = {"research", "quiz", "summary", "cards", "rewrite", "hint", "solution", "teacher"}


def _int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


async def agent_action(request: Request):
    """One click under an answer -> the agent -> the matching piece of page."""
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    form = await check_csrf(request)
    action = str(form.get("action", ""))
    if action not in ACTIONS:
        raise HTTPException(status_code=404)
    text = strip_isolates(str(form.get("text", ""))).strip()
    asked_lang = str(form.get("lang", ""))                         # the language the ANSWER was written in: its quiz and rewrites follow it
    body: dict = {"action": action, "text": text[:4000], "language": lang_of(request)}   # a study file and its tools: the site language
    if action in ("rewrite", "quiz") and not _int(form.get("study_file_id")) and asked_lang in ("ar", "en"):
        body["language"] = asked_lang
    if action == "research":
        if not text:
            return notice(request, tr(request, "error.empty_question"), "warning")
        body["pages"] = max(1, min(_int(form.get("pages"), 1), 10))
        if _int(form.get("parent")):
            body["parent_question_id"] = _int(form.get("parent"))
    elif action in ("quiz", "summary", "cards"):
        if _int(form.get("study_file_id")):
            body["study_file_id"] = _int(form.get("study_file_id"))
        elif action == "quiz":                                     # right after an answer from the material: M4's questions for that topic
            body["topic_id"] = str(form.get("topic_id", ""))[:80] or None
            body["question"] = strip_isolates(str(form.get("question", "")))[:2000]
            body["text"] = body["question"]
    elif action == "rewrite":
        body["mode"] = str(form.get("mode", ""))
        body["question"] = strip_isolates(str(form.get("question", "")))[:2000]
        if str(form.get("other", "")).strip():
            body["other"] = strip_isolates(str(form.get("other", "")))[:200]
    else:                                                          # hint / solution
        body["key"] = str(form.get("key", ""))
        if action == "hint":
            body["n"] = max(1, min(_int(form.get("n"), 1), 3))
    res = await api(request).post(f"/rooms/{room['room_id']}/tutor/act", token=user["token"], json=body, timeout=240)
    ensure_session(request, res)
    if not res.ok:
        return notice(request, api_message(request, res), "warning")
    out = res.data
    if not out["ok"] and action == "teacher":
        return notice(request, tr(request, "tutor.teacher_fail"), "warning")
    if not out["ok"] and action in ("hint", "solution"):           # an unknown key: say so instead of an empty hint
        return notice(request, tr(request, "tutor.tool_fail"), "warning")
    ctx = dict(room=room, out=out, r=out["result"], turn_id=secrets.token_hex(4), question=text, key=body.get("key", ""),
               formats=list(export.FORMATS), quiz_topic=body.get("topic_id") or "")
    if action == "research":
        return render(request, "partials/tutor_study_turn.html", pages=body["pages"], **ctx)
    if action == "quiz":
        return render(request, "partials/tutor_quiz.html", fid=body.get("study_file_id"), **ctx)
    if action == "summary":
        return render(request, "partials/tutor_summary.html", **ctx)
    if action == "cards":
        return render(request, "partials/tutor_cards.html", **ctx)
    if action == "rewrite":
        return render(request, "partials/tutor_rewrite.html", **ctx)
    if action == "teacher":
        return render(request, "partials/tutor_teacher.html", **ctx)
    return render(request, "partials/tutor_hint.html" if action == "hint" else "partials/tutor_solution.html", **ctx)


async def quiz_result(request: Request):
    """The student finished the quiz shown in the chat: correct/total is a result for M4, so the Progress page follows it."""
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    form = await check_csrf(request)
    total, correct = _int(form.get("total"), 0), _int(form.get("correct"), -1)
    topic = str(form.get("topic", ""))[:80]
    if not 1 <= total <= 10 or not 0 <= correct <= total:
        raise HTTPException(status_code=422)
    saved = await run_in_threadpool(events.record_quiz, user["user_id"], room["room_id"], topic, correct, total)
    return JSONResponse({"saved": saved})


# ---------------------------------------------------------------- the room's study files

async def files_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "student")
    res = await api(request).post(f"/rooms/{room['room_id']}/tutor/act", token=user["token"], json={"action": "files", "language": lang_of(request)})
    ensure_session(request, res)
    files = res.data["result"].get("files", []) if res.ok else []
    return render(request, "room/study_files.html", room=room, rooms=rooms, tab="study-files", files=files, ok=res.ok,
                  error=None if res.ok else api_message(request, res))


async def file_page(request: Request):
    user, room, rooms = await load_room(request, request.path_params["room_id"], "student")
    fid = _int(request.path_params["fid"], 0)
    res = await api(request).post(f"/rooms/{room['room_id']}/tutor/act", token=user["token"],
                                  json={"action": "file", "study_file_id": fid, "language": lang_of(request)})
    ensure_session(request, res)
    if not res.ok or not res.data["ok"]:
        raise HTTPException(status_code=404)
    return render(request, "room/study_file.html", room=room, rooms=rooms, tab="study-files", f=res.data["result"], formats=list(export.FORMATS))


async def file_download(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    fid = _int(request.path_params["fid"], 0)
    fmt = request.query_params.get("format", "md")
    sf = study_store.get_study_file(fid)
    if not study_store.owns(sf, room["room_id"], user["user_id"]) or fmt not in ("md", *export.FORMATS):
        raise HTTPException(status_code=404)
    if fmt == "md":
        mime, data = "text/markdown; charset=utf-8", sf["markdown"].encode("utf-8")
    else:
        mime, fn = export.FORMATS[fmt]
        try:
            data = fn(sf["markdown"])
        except RuntimeError:
            raise HTTPException(status_code=502, detail="The export is not available on this server")
    return Response(data, media_type=mime, headers={"Content-Disposition": f'attachment; filename="study_file_{fid}.{fmt}"'})


async def file_delete(request: Request):
    """Delete one of the student's study files (the Word / PDF they downloaded are theirs and stay)."""
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    await check_csrf(request)
    fid = _int(request.path_params["fid"], 0)
    sf = study_store.get_study_file(fid)
    if not study_store.owns(sf, room["room_id"], user["user_id"]):
        raise HTTPException(status_code=404)
    study_store.delete_study_file(fid)
    flash(request, "success", tr(request, "tutor.files.deleted", title=sf["title"]))
    return redirect(request, f"/rooms/{room['room_id']}/study-files")


__all__ = ["agent_action", "files_page", "file_page", "file_download", "file_delete"]


async def explain_turn(request: Request, user: dict, room: dict, shown: str, text: str):
    """A photo the student asked to have explained: the room's material first, then the web right away. -> the chat turn, or None when the agent failed."""
    res = await api(request).post(f"/rooms/{room['room_id']}/tutor/act", token=user["token"], timeout=300,
                                  json={"action": "explain", "text": text[:4000], "language": lang_of(request), "pages": 1})
    ensure_session(request, res)
    if not res.ok or not res.data["ok"] and not res.data["result"]:
        return None
    out, r = res.data, res.data["result"]
    turn = secrets.token_hex(4)
    if out["action"] == "help":                                    # a problem in the photo: hints, one click at a time
        lang = lang_of(request)
        from rafeeq.tutor.connect import LEAD
        reply = {"route": "ask", "reply": LEAD[lang], "payload": {"answer": LEAD[lang], "citations": [], "grounded": True, "steps": [], "help_key": r["key"], "problem_kind": r["kind"]}}
    elif r.get("study_file"):                                      # not in the material: it came from the web
        return render(request, "partials/tutor_study_turn.html", room=room, out=out, r=r, turn_id=turn, question=text, shown_text=shown, explained=True,
                      pages=1, formats=list(export.FORMATS))
    else:
        reply = {"route": "ask", "reply": r["answer"], "payload": {k: r[k] for k in ("answer", "citations", "grounded", "steps", "topic_id", "question_id", "answer_flagged", "language", "web_sources", "material_source")}}
    return render(request, "partials/chat_turn.html", room=room, shown=shown, file_id=None, res=res, reply=reply, turn_id=turn, topic_names={}, error=None)


# ---------------------------------------------------------------- chat history and memory (per account)
_USER_BUBBLE = re.compile(r'<div class="bubble bubble-user"><p>(.*?)</p>', re.S)
_TAGS = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S)


def _text_of(fragment: str) -> str:
    return " ".join(htmllib.unescape(_TAGS.sub(" ", fragment)).split())


def split_turn(page_html: str) -> tuple[str, str]:
    """(what the student wrote, what Rafeeq said) out of a rendered chat turn: plain text, for the history and the short memory."""
    m = _USER_BUBBLE.search(page_html)
    if not m:
        return "", ""
    return _text_of(m.group(1)), _text_of(page_html[m.end():])[:1200]


def remembering(handler, actions: set[str] | None = None):
    """Wrap a chat handler: the student's memory goes into the request (short memory of this chat + the long facts), what they say about themselves is
    learned, and the turn is saved in the chat history. `actions`: save the turn only for these agent actions (None = always)."""
    async def wrapped(request: Request):
        user = current_user(request)
        rid = request.path_params["room_id"]
        cid = request.headers.get("x-chat-id", "")
        uid = user["user_id"] if user else None
        form = await request.form() if request.method == "POST" else {}
        if uid and chats.valid(cid) and str(form.get("mode", "auto")) != "check_work" and "action" not in form:
            await run_in_threadpool(memory.extract, uid, strip_text(form))      # something important about the student is kept by itself
        token = memory.CURRENT.set(memory.context_for(uid, rid, cid) if uid and chats.valid(cid) else None)
        try:
            resp = await handler(request)
        finally:
            memory.CURRENT.reset(token)
        if uid and chats.valid(cid) and resp.status_code == 200 and (actions is None or str(form.get("action", "")) in actions):
            body = bytes(resp.body).decode("utf-8", "replace")
            who, said = split_turn(body)
            if who and said:
                chats.save_turn(uid, rid, cid, who, said, body)
        return resp
    return wrapped


def strip_text(form) -> str:
    return _strip(str(form.get("text", "") or form.get("question", ""))).strip()


async def study_page(request: Request):
    """Ask Rafeeq: a new chat, or a saved one (?chat=...), the chat list and what Rafeeq remembers."""
    user, room, rooms = await load_room(request, request.path_params["room_id"], "student")
    rid, uid = room["room_id"], user["user_id"]
    wanted = request.query_params.get("chat", "")
    saved = chats.turns(uid, rid, wanted) if chats.valid(wanted) else []
    cid = wanted if saved else chats.new_id()
    inbox_res = await api(request).get(f"/rooms/{rid}/inbox", token=user["token"])                  # the teacher's answers to this student's questions
    my_questions = inbox_res.data.get("questions", [])[:6] if inbox_res.ok else []
    return render(request, "room/study.html", room=room, rooms=rooms, tab="study", my_questions=my_questions, prefill=_strip(request.query_params.get("q", ""))[:500],
                  chat_id=cid, saved=saved, chat_list=chats.list_chats(uid, rid), notes=memory.notes(uid), recent=memory.recent_topics(uid, rid))


async def chat_delete(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    await check_csrf(request)
    if chats.delete_chat(user["user_id"], room["room_id"], request.path_params["cid"]):
        flash(request, "success", tr(request, "chats.deleted"))
    return redirect(request, f"/rooms/{room['room_id']}/study")


async def memory_delete(request: Request):
    """Delete one remembered note, or all of them (forget everything)."""
    user, room, _ = await load_room(request, request.path_params["room_id"], "student")
    form = await check_csrf(request)
    nid = request.path_params.get("nid")
    if nid is None:
        memory.clear(user["user_id"])
        flash(request, "success", tr(request, "memory.cleared"))
    elif memory.delete_note(user["user_id"], _int(nid, 0)):
        flash(request, "success", tr(request, "memory.deleted"))
    chat = str(form.get("chat", ""))
    back = {"chat": chat} if chats.valid(chat) and chats.turns(user["user_id"], room["room_id"], chat) else {}
    return redirect(request, f"/rooms/{room['room_id']}/study", **back)

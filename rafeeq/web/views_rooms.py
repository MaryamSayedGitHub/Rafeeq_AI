"""My rooms (with the first-time empty states), joining, creating, and the room loader."""
from __future__ import annotations

import asyncio

from starlette.exceptions import HTTPException
from starlette.requests import Request

from rafeeq.web import api as A
from rafeeq.web.common import Redirect, api, api_message, check_csrf, flash, redirect, render, require_user, tr, url


def ensure_session(request: Request, res: A.ApiResult) -> None:
    """If the API says the token is no longer valid, end the web session too."""
    if res.kind == A.AUTH:
        lang = request.session.get("lang")
        request.session.clear()
        if lang:
            request.session["lang"] = lang
        flash(request, "warning", tr(request, "err.session"))
        raise Redirect(url(request, "/login"))


async def load_room(request: Request, room_id: str, role: str | None = None) -> tuple[dict, dict, list]:
    """(user, room, all rooms). 404 if this user has no such room."""
    user = require_user(request, role)
    res = await api(request).get("/rooms", token=user["token"])
    ensure_session(request, res)
    if not res.ok:
        raise HTTPException(status_code=502, detail=api_message(request, res))
    rooms = res.data.get("rooms", [])
    room = next((r for r in rooms if r["room_id"] == room_id), None)
    if room is None:
        raise HTTPException(status_code=404, detail="room")
    return user, room, rooms


async def topic_names(request: Request, user: dict, room_id: str) -> tuple[A.ApiResult, dict]:
    res = await api(request).get(f"/rooms/{room_id}/topics", token=user["token"])
    names = {t["topic_id"]: t["name"] for t in res.get("topics", [])} if res.ok else {}
    return res, names


async def rooms_index(request: Request):
    user = require_user(request)
    rooms_res = await api(request).get("/rooms", token=user["token"])
    ensure_session(request, rooms_res)
    return render(request, "rooms/index.html", rooms_res=rooms_res, rooms=rooms_res.get("rooms", []),
                  error=None if rooms_res.ok else api_message(request, rooms_res))


async def rooms_join(request: Request):
    user = require_user(request, "student")
    form = await check_csrf(request)
    code = str(form.get("code", "")).strip()
    res = await api(request).post("/rooms/join", token=user["token"], json={"code": code})
    ensure_session(request, res)
    if res.ok:
        flash(request, "success", tr(request, "rooms.joined", name=res.data["name"]))
        return redirect(request, f"/rooms/{res.data['room_id']}")
    rooms_res = await api(request).get("/rooms", token=user["token"])
    msg = api_message(request, res, {A.NOT_FOUND: "err.code_not_found"})
    return render(request, "rooms/index.html", status=400, rooms_res=rooms_res,
                  rooms=rooms_res.get("rooms", []), join_error=msg, code=code)


async def rooms_create(request: Request):
    user = require_user(request, "teacher")
    form = await check_csrf(request)
    name = str(form.get("name", "")).strip()
    res = await api(request).post("/rooms", token=user["token"], json={"name": name})
    ensure_session(request, res)
    if res.ok:
        return redirect(request, f"/rooms/{res.data['room']['room_id']}/overview", created="1")
    rooms_res = await api(request).get("/rooms", token=user["token"])
    return render(request, "rooms/index.html", status=400, rooms_res=rooms_res, rooms=rooms_res.get("rooms", []),
                  create_error=api_message(request, res), name=name)


async def room_home(request: Request):
    user, room, _ = await load_room(request, request.path_params["room_id"])
    tab = "overview" if user["role"] == "teacher" else "study"
    return redirect(request, f"/rooms/{room['room_id']}/{tab}")


async def gather(*coros):
    return await asyncio.gather(*coros)

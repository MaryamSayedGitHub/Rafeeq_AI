"""Who may call what: one place for the room checks every module uses.

Production rule: every route that reads a room's material, a student's data or spends model credit needs a
logged-in user (M1's Bearer token) who belongs to that room (its teacher or a member). Teacher-only data
(class report, insights, class profile, wrong answers, the anonymous question log) needs the room's teacher.

RAFEEQ_ALLOW_ANONYMOUS=1 brings back the old development behaviour: these room checks are skipped (contract
samples, X-User-Id, the M3 demo token, rooms that exist only in samples). tests/conftest.py turns it on because
the teammates' tests were written for that mode; tests/test_access.py checks the real rule. Never set it on a
real server.
"""
from __future__ import annotations

import os

from fastapi import HTTPException


def anonymous_allowed() -> bool:
    return os.environ.get("RAFEEQ_ALLOW_ANONYMOUS", "0").strip() == "1"


def _store():
    from rafeeq.platform import service

    return service.get_store()


def user_from_header(authorization: str | None) -> dict | None:
    """The logged-in user, None when no header was sent; 401 for a header that is not a valid token."""
    if not authorization:
        return None
    from rafeeq.platform import service

    token = service.token_from_header(authorization)
    user = _store().user_for_token(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


def _room(room_id: str) -> dict:
    room = _store().db["rooms"].get(room_id or "")
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


def check_member(user_id: str, room_id: str) -> None:
    """403 unless the user is the room's teacher or one of its members."""
    room = _room(room_id)
    members = _store().db["members"].get(room_id, [])
    if room["teacher_id"] != user_id and user_id not in members:
        raise HTTPException(status_code=403, detail="Not a member of this room")


def check_teacher(user_id: str, room_id: str) -> None:
    if _room(room_id)["teacher_id"] != user_id:
        raise HTTPException(status_code=403, detail="Only this room's teacher can see this")


def room_member(authorization: str | None, room_id: str) -> dict | None:
    """The calling member of the room (None in development mode, where nothing is checked)."""
    if anonymous_allowed():
        return None
    user = user_from_header(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Log in first")
    check_member(user["user_id"], room_id)
    return user


def room_teacher(authorization: str | None, room_id: str) -> dict | None:
    """The room's teacher (None in development mode, where nothing is checked)."""
    if anonymous_allowed():
        return None
    user = user_from_header(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="Log in first")
    check_teacher(user["user_id"], room_id)
    return user


def member_by_id(user_id: str, room_id: str) -> None:
    """For routes that already know the logged-in user_id (M4's Depends(current_user_id))."""
    if anonymous_allowed():
        return
    check_member(user_id, room_id)


def teacher_by_id(user_id: str, room_id: str) -> None:
    if anonymous_allowed():
        return
    check_teacher(user_id, room_id)

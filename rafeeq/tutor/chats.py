"""Chat history, one account at a time: every conversation a student has in Ask Rafeeq is saved and can be reopened or deleted.

A chat belongs to (user, room). Each turn keeps the question, the answer's text (for the short memory) and the page fragment that was shown, so reopening
a chat shows the same cards. Nobody but the owner reads or deletes a chat: every function takes the user id and checks it.
"""
from __future__ import annotations

import re
import uuid

from .study_store import db, now_iso

_ID = re.compile(r"[0-9a-f]{12}")
HISTORY_TURNS = 4                       # the short memory: the last 4 turns (8 messages) of THIS chat


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def valid(chat_id: str | None) -> bool:
    return bool(chat_id and _ID.fullmatch(chat_id))


def _owned(con, user_id: str, room_id: str, chat_id: str):
    return con.execute("SELECT * FROM chats WHERE id=? AND user_id=? AND room_id=?", (chat_id, user_id, room_id)).fetchone()


def list_chats(user_id: str, room_id: str, limit: int = 40) -> list[dict]:
    with db() as con:
        rows = con.execute("SELECT c.id, c.title, c.updated, (SELECT COUNT(*) FROM chat_turns t WHERE t.chat_id=c.id) AS turns FROM chats c "
                           "WHERE c.user_id=? AND c.room_id=? ORDER BY c.updated DESC LIMIT ?", (user_id, room_id, limit))
        return [dict(r) for r in rows]


def turns(user_id: str, room_id: str, chat_id: str) -> list[dict]:
    """[{user_text, reply_text, html, ts}] oldest first; [] for a chat that is not this user's."""
    with db() as con:
        if not valid(chat_id) or not _owned(con, user_id, room_id, chat_id):
            return []
        return [dict(r) for r in con.execute("SELECT user_text, reply_text, html, ts FROM chat_turns WHERE chat_id=? ORDER BY id", (chat_id,))]


def history(user_id: str, room_id: str, chat_id: str | None, n: int = HISTORY_TURNS) -> list[dict]:
    """The short memory of one chat: [{"role": "student" | "rafeeq", "text"}] for its last n turns."""
    if not valid(chat_id):
        return []
    out = []
    for t in turns(user_id, room_id, chat_id)[-n:]:
        out += [{"role": "student", "text": t["user_text"]}, {"role": "rafeeq", "text": t["reply_text"]}]
    return out


def save_turn(user_id: str, room_id: str, chat_id: str, user_text: str, reply_text: str, html: str) -> bool:
    """Add a turn (the chat is created by its first turn, titled by the first question). False when the id is not valid or not this user's."""
    if not valid(chat_id) or not user_id:
        return False
    with db() as con:
        chat = con.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if chat and (chat["user_id"] != user_id or chat["room_id"] != room_id):
            return False
        if not chat:
            con.execute("INSERT INTO chats(id, user_id, room_id, title, created, updated) VALUES (?,?,?,?,?,?)",
                        (chat_id, user_id, room_id, " ".join(user_text.split())[:60] or "…", now_iso(), now_iso()))
        con.execute("INSERT INTO chat_turns(chat_id, ts, user_text, reply_text, html) VALUES (?,?,?,?,?)",
                    (chat_id, now_iso(), user_text[:2000], reply_text[:1500], html))
        con.execute("UPDATE chats SET updated=? WHERE id=?", (now_iso(), chat_id))
    return True


def delete_chat(user_id: str, room_id: str, chat_id: str) -> bool:
    with db() as con:
        if not valid(chat_id) or not _owned(con, user_id, room_id, chat_id):
            return False
        con.execute("DELETE FROM chat_turns WHERE chat_id=?", (chat_id,))
        con.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    return True

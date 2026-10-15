"""What Rafeeq remembers about a student.

  short memory   the last turns of the chat that is open (chats.history): lets a follow-up like "and its function?" be understood;
  long memory    a few short NOTES that follow the student into every new chat. Nobody types them: when the student says something important about themselves
                 in a chat (who they are, what they study, an exam coming, what they find hard, how they like things explained), extract() notices it and keeps it.
                 Only those notes are kept, never the chat text. The student sees every note, can delete one, or forget everything.

The notes belong to the account (not to a room). The memory of the request in progress travels in CURRENT (a ContextVar set by the chat views), so the
tutor agent needs no extra argument.
"""
from __future__ import annotations

import re
from contextvars import ContextVar

from . import chats, llm
from .study_store import db, now_iso

CURRENT: ContextVar[dict | None] = ContextVar("tutor_memory", default=None)
KINDS = ("name", "study", "goal", "difficulty", "preference", "other")
ONE_OF = ("name", "study")              # a new one replaces the old one ("I'm in year 2 now")
MAX_NOTES = 30
MAX_LEN = 140

# a cheap gate: the model is only asked when the student talks about themselves
_SELF = re.compile(r"\b(?:i|i'm|im|i've|my|me|myself)\b|(?:^|\s)(?:انا|أنا|اسمي|إسمي|عندي|بدرس|بذاكر|بدرّس|مذاكر|طالب|طالبة|امتحان|اختبار|مش\s+فاهم|صعب|هدفي|عايز|عاوز|أريد|احب|أحب|أفضل|بفضل)", re.I)


def looks_personal(text: str) -> bool:
    return bool(_SELF.search(text))


# ----------------------------------------------------------------------------- the notes
def notes(user_id: str | None) -> list[dict]:
    """[{id, kind, note, extra}] oldest first."""
    if not user_id:
        return []
    with db() as con:
        return [dict(r) for r in con.execute("SELECT id, kind, note, extra FROM memory_notes WHERE user_id=? ORDER BY id", (user_id,))]


def add_note(user_id: str, kind: str, text: str, extra: str = "") -> bool:
    kind = kind if kind in KINDS else "other"
    text = " ".join(str(text).split()).strip(" .")[:MAX_LEN]
    if not user_id or len(text) < 2:
        return False
    with db() as con:
        if kind in ONE_OF:
            con.execute("DELETE FROM memory_notes WHERE user_id=? AND kind=?", (user_id, kind))
        elif con.execute("SELECT 1 FROM memory_notes WHERE user_id=? AND lower(note)=lower(?)", (user_id, text)).fetchone():
            return False
        con.execute("INSERT INTO memory_notes(user_id, kind, note, extra, ts) VALUES (?,?,?,?,?)", (user_id, kind, text, extra[:60], now_iso()))
        con.execute("DELETE FROM memory_notes WHERE user_id=? AND id NOT IN (SELECT id FROM memory_notes WHERE user_id=? ORDER BY id DESC LIMIT ?)", (user_id, user_id, MAX_NOTES))
    return True


def delete_note(user_id: str, note_id: int) -> bool:
    with db() as con:
        return con.execute("DELETE FROM memory_notes WHERE user_id=? AND id=?", (user_id, note_id)).rowcount > 0


def clear(user_id: str) -> None:
    with db() as con:
        con.execute("DELETE FROM memory_notes WHERE user_id=?", (user_id,))


def profile(user_id: str | None) -> dict:
    """The few facts the rest of the tutor uses: {"name", "field", "field_en"} (only what is known)."""
    out: dict = {}
    for n in notes(user_id):
        if n["kind"] == "name":
            out["name"] = n["note"]
        elif n["kind"] == "study":
            out["field"] = n["note"]
            if n["extra"]:
                out["field_en"] = n["extra"]
    return out


# ----------------------------------------------------------------------------- noticing what is worth keeping
def extract(user_id: str | None, user_text: str) -> list[dict]:
    """Read what the student just wrote and keep the important, lasting facts about THEM. -> the notes added ([] when there is nothing to keep)."""
    if not user_id or not looks_personal(user_text):
        return []
    res = llm.ask_json(f"""A student wrote this to their tutor: {user_text[:800]}

List ONLY facts about the student themselves that are worth remembering in FUTURE chats: their name, what and where they study and their level,
an exam or goal, something they find hard, how they like things explained. Do NOT list the subject they asked about, opinions about the subject,
or anything temporary. Most messages have nothing to keep: then return an empty list.
Return JSON only: {{"facts": [{{"kind": "name | study | goal | difficulty | preference | other", "text": "a short sentence in the student's language", "field_en": "only for kind study: the field in English, one to three words"}}]}}
(for kind name, "text" is just the name)""", max_tokens=300)
    added = []
    for f in (res.get("facts") or []) if isinstance(res, dict) else []:
        if isinstance(f, dict) and add_note(user_id, str(f.get("kind", "other")), str(f.get("text", "")), str(f.get("field_en", "") or "")):
            added.append({"kind": f.get("kind"), "note": f.get("text")})
    return added


# ----------------------------------------------------------------------------- what goes into a request
def recent_topics(user_id: str | None, room_id: str, n: int = 5) -> list[str]:
    """The last few DIFFERENT things the student asked in their saved chats of this room (the long memory of their history), short.
    Deleting a chat removes it from here too."""
    if not user_id:
        return []
    seen, out = set(), []
    with db() as con:
        rows = con.execute("SELECT t.user_text FROM chat_turns t JOIN chats c ON c.id = t.chat_id WHERE c.user_id=? AND c.room_id=? ORDER BY t.id DESC LIMIT 40",
                           (user_id, room_id))
        for r in rows:
            key = " ".join(r["user_text"].lower().split())
            if key and key not in seen:
                seen.add(key)
                out.append(" ".join(r["user_text"].split())[:80])
            if len(out) >= n:
                break
    return out


def context_for(user_id: str | None, room_id: str, chat_id: str | None) -> dict:
    """The memory for one request: the long notes + recent topics (always) and the short memory of this chat (when it has turns)."""
    return {"profile": profile(user_id), "notes": [n["note"] for n in notes(user_id)], "recent": recent_topics(user_id, room_id),
            "history": chats.history(user_id, room_id, chat_id) if user_id else []}


def note(ctx: dict | None = None) -> str:
    """One short paragraph for a prompt: what is remembered about the student. Empty when nothing is known."""
    ctx = ctx if ctx is not None else CURRENT.get()
    if not ctx:
        return ""
    bits = []
    if ctx.get("notes"):
        bits.append("Remembered from earlier chats: " + "; ".join(ctx["notes"][-8:]) + ". When an example helps, take it from the student's field. Never change a fact for it.")
    if ctx.get("recent"):
        bits.append("Lately they asked about: " + "; ".join(ctx["recent"][:4]) + ".")
    return " ".join(bits)


def search_field(ctx: dict | None = None) -> str:
    """What the student studies, in English, to narrow a web search ("" when unknown)."""
    ctx = ctx if ctx is not None else CURRENT.get()
    p = (ctx or {}).get("profile") or {}
    return p.get("field_en") or (p.get("field") if (p.get("field") or "").isascii() else "") or ""


def standalone(question: str, ctx: dict | None = None) -> str:
    """A follow-up ("and its function?") rewritten so it can be understood alone, using the chat so far. The question unchanged when it needs no help."""
    ctx = ctx if ctx is not None else CURRENT.get()
    hist = (ctx or {}).get("history") or []
    if not hist or len(question.split()) > 14:
        return question
    convo = "\n".join(f"{'Student' if m['role'] == 'student' else 'Tutor'}: {m['text'][:300]}" for m in hist[-6:])
    res = llm.ask_json(f"""Conversation so far:
{convo}

The student now writes: {question}

If this message only makes sense with the conversation (it says "it", "that", "and the other one", "why?"...), rewrite it as ONE complete question in the same language,
using the subject from the conversation. KEEP any request about the answer's size or language (\"in detail\", \"بالتفصيل\", \"in Arabic\"). If it already stands alone, return it unchanged.
Return JSON only: {{"question": "..."}}""", max_tokens=150)
    out = str(res.get("question", "")).strip() if isinstance(res, dict) else ""
    return out if 3 <= len(out) <= 400 else question

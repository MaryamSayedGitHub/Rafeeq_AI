"""SQLite for Level 2: the saved study files, the web-search cache and the monthly search-credit counter.

Folder: data/study/ in the repository (git-ignored; override with STUDY_DATA_DIR). Study files are also written as .md files.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    d = Path(os.getenv("STUDY_DATA_DIR") or ROOT / "data" / "study")
    (d / "files").mkdir(parents=True, exist_ok=True)
    return d


SCHEMA = """
CREATE TABLE IF NOT EXISTS study_files(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, room_id TEXT, question TEXT, title TEXT, markdown TEXT, sources_json TEXT, path TEXT, user_id TEXT);
CREATE TABLE IF NOT EXISTS questions(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, room_id TEXT, question TEXT NOT NULL, kind TEXT, grounded INTEGER,
  public_id TEXT, user_id TEXT, topic_id TEXT, level INTEGER, engine TEXT, study_file_id INTEGER);
CREATE TABLE IF NOT EXISTS review(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, room_id TEXT, kind TEXT, prompt TEXT NOT NULL, options_json TEXT, answer TEXT,
  explanation TEXT, topic_id TEXT, due TEXT NOT NULL, stage INTEGER DEFAULT 0, done INTEGER DEFAULT 0, UNIQUE(room_id, prompt));
CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY, user_id TEXT NOT NULL, room_id TEXT NOT NULL, title TEXT, created TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS chat_turns(id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT NOT NULL, ts TEXT, user_text TEXT, reply_text TEXT, html TEXT);
CREATE TABLE IF NOT EXISTS memory_notes(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, kind TEXT, note TEXT NOT NULL, extra TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS search_cache(key TEXT PRIMARY KEY, ts REAL, query TEXT, results_json TEXT);
CREATE TABLE IF NOT EXISTS usage(month TEXT PRIMARY KEY, api_credits INTEGER DEFAULT 0, cache_hits INTEGER DEFAULT 0);
"""


QUESTION_COLUMNS = {"public_id": "TEXT", "user_id": "TEXT", "topic_id": "TEXT", "level": "INTEGER", "engine": "TEXT", "study_file_id": "INTEGER"}


def _migrate(con) -> None:
    """A study.db made before the question log had these columns gets them added (nothing is lost)."""
    have = {r["name"] for r in con.execute("PRAGMA table_info(questions)")}
    for col, typ in QUESTION_COLUMNS.items():
        if col not in have:
            con.execute(f"ALTER TABLE questions ADD COLUMN {col} {typ}")
    if "user_id" not in {r["name"] for r in con.execute("PRAGMA table_info(study_files)")}:        # study files belong to the student who asked
        con.execute("ALTER TABLE study_files ADD COLUMN user_id TEXT")


@contextmanager
def db():
    con = sqlite3.connect(data_dir() / "study.db")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    _migrate(con)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- study files
def save_study_file(room_id, question, title, markdown, sources, user_id=None) -> dict:
    """A study file belongs to the student who asked for it (user_id): nobody else in the room sees it."""
    with db() as con:
        cur = con.execute("INSERT INTO study_files(ts, room_id, question, title, markdown, sources_json, path, user_id) VALUES (?,?,?,?,?,?,?,?)",
                          (now_iso(), room_id, question, title, markdown, json.dumps(sources, ensure_ascii=False), "", user_id))
        fid = cur.lastrowid
        slug = re.sub(r"[^\w؀-ۿ]+", "-", title).strip("-")[:50] or "study-file"
        path = data_dir() / "files" / f"{fid:04d}_{slug}.md"
        path.write_text(markdown, encoding="utf-8")
        con.execute("UPDATE study_files SET path=? WHERE id=?", (str(path), fid))
    return {"id": fid, "path": str(path), "title": title}


def list_study_files(room_id: str | None = None, user_id: str | None = None) -> list[dict]:
    """The files of ONE student in a room (user_id None = the files made without a signed-in student)."""
    with db() as con:
        q = "SELECT id, ts, room_id, question, title, user_id FROM study_files WHERE " + ("room_id=? AND " if room_id else "") + ("user_id=?" if user_id else "user_id IS NULL")
        args = ((room_id,) if room_id else ()) + ((user_id,) if user_id else ())
        return [dict(r) for r in con.execute(q + " ORDER BY id DESC", args)]


def owns(sf: dict | None, room_id: str, user_id: str | None) -> bool:
    """Is this file this student's, in this room?"""
    return bool(sf) and sf.get("room_id") == room_id and (sf.get("user_id") or None) == (user_id or None)


def get_study_file(fid: int) -> dict | None:
    with db() as con:
        r = con.execute("SELECT * FROM study_files WHERE id=?", (fid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["sources"] = json.loads(d.pop("sources_json") or "[]")
    return d


def delete_study_file(fid: int) -> bool:
    with db() as con:
        r = con.execute("SELECT path FROM study_files WHERE id=?", (fid,)).fetchone()
        if not r:
            return False
        con.execute("DELETE FROM study_files WHERE id=?", (fid,))
        con.execute("UPDATE questions SET study_file_id=NULL WHERE study_file_id=?", (fid,))      # the question stays in the log, the file is gone
    if r["path"]:
        Path(r["path"]).unlink(missing_ok=True)
    for kind in ("summaries", "flashcards"):                       # what was made from the file goes with it
        for cached in (data_dir() / kind).glob(f"{fid}-*.json"):
            cached.unlink(missing_ok=True)
    return True


# ----------------------------------------------------------------------------- recent questions
def log_question(room_id: str, question: str, kind: str, grounded: bool, topic_id: str | None = None, level: int = 1,
                 user_id: str | None = None, engine: str = "", study_file_id: int | None = None) -> int:
    """Save a question. user_id stays inside this database: it is never part of what the instructor endpoint returns."""
    with db() as con:
        cur = con.execute("INSERT INTO questions(ts, room_id, question, kind, grounded, public_id, user_id, topic_id, level, engine, study_file_id) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (now_iso(), room_id, question[:500], kind, int(grounded), uuid.uuid4().hex, user_id, topic_id, level, engine, study_file_id))
        return cur.lastrowid


def update_question(qid: int, **fields) -> None:
    allowed = {"level", "grounded", "engine", "study_file_id", "topic_id"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if fields:
        with db() as con:
            con.execute(f"UPDATE questions SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", (*fields.values(), qid))


def list_questions(room_id: str, user_id: str | None = None, limit: int = 100) -> list[dict]:
    """The room's question log, newest first. With user_id: only that student's questions."""
    with db() as con:
        q = "SELECT * FROM questions WHERE room_id=?" + (" AND user_id=?" if user_id else "") + " ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in con.execute(q, (room_id, user_id, limit) if user_id else (room_id, limit))]


def recent_questions(room_id: str | None = None, limit: int = 6) -> list[dict]:
    """The latest distinct questions (newest first)."""
    with db() as con:
        q = "SELECT question, kind, grounded, MAX(id) AS id FROM questions" + (" WHERE room_id=?" if room_id else "") + " GROUP BY question ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in con.execute(q, (room_id, limit) if room_id else (limit,))]


# ----------------------------------------------------------------------------- spaced review: what the student got wrong comes back
STAGE_DAYS = (1, 7, 30)                 # a missed item returns after 1 day; each success pushes it to 7, then 30 days, then it is done


def _due_in(days: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(timespec="seconds")


def add_review(room_id: str, kind: str, prompt: str, answer: str = "", explanation: str = "", options: list[str] | None = None,
               topic_id: str | None = None) -> None:
    """A mistake to revisit tomorrow. The same prompt in the same room is kept once (a new mistake restarts its schedule)."""
    with db() as con:
        con.execute("INSERT INTO review(ts, room_id, kind, prompt, options_json, answer, explanation, topic_id, due, stage, done) "
                    "VALUES (?,?,?,?,?,?,?,?,?,0,0) ON CONFLICT(room_id, prompt) DO UPDATE SET due=excluded.due, stage=0, done=0, "
                    "answer=excluded.answer, explanation=excluded.explanation, options_json=excluded.options_json",
                    (now_iso(), room_id, kind, prompt[:600], json.dumps(options or [], ensure_ascii=False), answer, explanation, topic_id, _due_in(STAGE_DAYS[0])))


def due_reviews(room_id: str | None = None, limit: int = 20) -> list[dict]:
    with db() as con:
        q = "SELECT * FROM review WHERE done=0 AND due<=?" + (" AND room_id=?" if room_id else "") + " ORDER BY due LIMIT ?"
        rows = con.execute(q, (now_iso(), room_id, limit) if room_id else (now_iso(), limit))
        return [{**dict(r), "options": json.loads(r["options_json"] or "[]")} for r in rows]


def count_reviews() -> dict:
    with db() as con:
        due = con.execute("SELECT COUNT(*) FROM review WHERE done=0 AND due<=?", (now_iso(),)).fetchone()[0]
        total = con.execute("SELECT COUNT(*) FROM review WHERE done=0").fetchone()[0]
    return {"due": due, "waiting": total - due}


def grade_review(rid: int, knew_it: bool) -> bool:
    """knew_it -> next stage (7 days, 30 days, then done); not yet -> back to tomorrow."""
    with db() as con:
        r = con.execute("SELECT stage FROM review WHERE id=? AND done=0", (rid,)).fetchone()
        if not r:
            return False
        if not knew_it:
            con.execute("UPDATE review SET stage=0, due=? WHERE id=?", (_due_in(STAGE_DAYS[0]), rid))
        elif r["stage"] + 1 >= len(STAGE_DAYS):
            con.execute("UPDATE review SET done=1 WHERE id=?", (rid,))
        else:
            con.execute("UPDATE review SET stage=?, due=? WHERE id=?", (r["stage"] + 1, _due_in(STAGE_DAYS[r["stage"] + 1]), rid))
    return True


# ----------------------------------------------------------------------------- small JSON cache (quizzes being taken, flashcards, summaries)
def put_json(kind: str, key: str, data) -> None:
    d = data_dir() / kind
    d.mkdir(exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def get_json(kind: str, key: str):
    if not re.fullmatch(r"[\w-]{1,64}", key or ""):
        return None
    p = data_dir() / kind / f"{key}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ----------------------------------------------------------------------------- search cache + usage
def cache_get(key: str, ttl_seconds: float):
    with db() as con:
        r = con.execute("SELECT ts, results_json FROM search_cache WHERE key=?", (key,)).fetchone()
    if r and time.time() - r["ts"] <= ttl_seconds:
        return json.loads(r["results_json"])
    return None


def cache_put(key: str, query: str, results: list[dict]) -> None:
    with db() as con:
        con.execute("INSERT OR REPLACE INTO search_cache(key, ts, query, results_json) VALUES (?,?,?,?)",
                    (key, time.time(), query, json.dumps(results, ensure_ascii=False)))


def month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def usage_add(credits: int = 0, cache_hits: int = 0) -> None:
    with db() as con:
        con.execute("INSERT OR IGNORE INTO usage(month) VALUES (?)", (month_key(),))
        con.execute("UPDATE usage SET api_credits = api_credits + ?, cache_hits = cache_hits + ? WHERE month=?",
                    (credits, cache_hits, month_key()))


def usage_get() -> dict:
    with db() as con:
        r = con.execute("SELECT * FROM usage WHERE month=?", (month_key(),)).fetchone()
    return dict(r) if r else {"month": month_key(), "api_credits": 0, "cache_hits": 0}

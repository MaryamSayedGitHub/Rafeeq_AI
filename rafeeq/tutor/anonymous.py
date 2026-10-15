"""Anonymous instructor questions for M5: what the students asked, with NO names.

Exported fields only: public_id (random, not derived from the student), date (day), topic_id, level, grounded, question.
The question text is cleaned of e-mails, phone numbers, links and the student's own id. Regex cleaning cannot catch every name typed
inside a question, so M5 should show topic-level numbers first and open single questions only when needed.
"""
from __future__ import annotations

import csv
import difflib
import io
import json
import os
import re

import requests

from . import study_store

FIELDS = ["public_id", "date", "topic_id", "level", "grounded", "question"]
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d")


def redact(text: str, user_id: str | None = None) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[link]", text)
    text = _PHONE.sub("[phone]", text)
    if user_id and len(user_id) >= 3:
        text = re.sub(re.escape(user_id), "[student]", text, flags=re.I)
    return text.strip()


def rows(room_id: str) -> list[dict]:
    """The room's questions, oldest first, with nothing that identifies a student."""
    out = []
    for r in reversed(study_store.list_questions(room_id, limit=100000)):
        out.append({"public_id": r["public_id"] or "", "date": r["ts"][:10], "topic_id": r["topic_id"] or "other", "level": r["level"] or 1,
                    "grounded": bool(r["grounded"]), "question": redact(r["question"], r["user_id"])})
    return out


def to_csv(items: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(items)
    return buf.getvalue()


def publish(room_id: str) -> dict:
    """Write data/study/instructor_<room>.jsonl and POST the same rows to M5_PUBLISH_URL when it is set."""
    items = rows(room_id)
    path = study_store.data_dir() / f"instructor_{re.sub(r'[^A-Za-z0-9_-]', '_', room_id)}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in items) + ("\n" if items else ""), encoding="utf-8")
    posted, error = False, ""
    url = os.getenv("M5_PUBLISH_URL")
    if url:
        try:
            requests.post(url, json={"room_id": room_id, "questions": items}, timeout=30).raise_for_status()
            posted = True
        except Exception as e:                                     # noqa: BLE001
            error = f"{type(e).__name__}: {e}"[:160]
    return {"count": len(items), "file": str(path), "posted": posted, "error": error}


SIMILAR = 0.78                      # two questions this alike (after cleaning) count as the same question
FREQUENT_MIN = 2                    # asked this many times: worth the teacher's attention


def _same(a: str, b: str) -> bool:
    a, b = re.sub(r"\W+", " ", a.lower()).strip(), re.sub(r"\W+", " ", b.lower()).strip()
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= SIMILAR


def frequent(room_id: str, min_count: int = FREQUENT_MIN) -> list[dict]:
    """What students ask again and again, for the teacher's class insights. Same topic + (nearly) the same words = one question.
    -> [{"topic_id", "question" (one example, cleaned), "count", "students" (how many different students, a number only),
         "unanswered" (times the material had no answer), "deeper" (times a student went to Level 2)}], most asked first. No names, no ids."""
    groups: list[dict] = []
    for r in study_store.list_questions(room_id, limit=100000):
        text, topic = redact(r["question"], r["user_id"]), r["topic_id"] or "other"
        g = next((g for g in groups if g["topic_id"] == topic and _same(g["question"], text)), None)
        if g is None:
            g = {"topic_id": topic, "question": text, "count": 0, "who": set(), "unanswered": 0, "deeper": 0}
            groups.append(g)
        g["count"] += 1
        g["who"].add(r["user_id"] or r["public_id"])
        g["unanswered"] += not r["grounded"]
        g["deeper"] += (r["level"] or 1) == 2
    out = [{"topic_id": g["topic_id"], "question": g["question"], "count": g["count"], "students": len(g["who"]),
            "unanswered": g["unanswered"], "deeper": g["deeper"]} for g in groups if g["count"] >= min_count]
    return sorted(out, key=lambda g: (-g["count"], g["question"]))

"""Course progress (M5, My learning platform): which items of a published course each student finished.

An item is one thing a student opens in the course player:
    "<lesson_id>:<asset_id>"   a library file of the lesson (video, YouTube moment, slides, audio...)
    "<lesson_id>:notes"        the lesson's written notes (shown as a reading)

Storage: the class store, next to the quiz results and the inbox:
    ("class", room_id, "course_progress")   key = "<user_id>|<course_id>"   value = {done: [item ids], last, time}

Only items that exist in the PUBLISHED snapshot are accepted, so a stale or invented id is ignored.
"""
from __future__ import annotations

from datetime import datetime, timezone


def _ns(room_id: str) -> tuple:
    return ("class", room_id, "course_progress")


def _key(user_id: str, course_id: str) -> str:
    return f"{user_id}|{course_id}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def item_ids(course: dict) -> list[str]:
    """Every item of a course, in the order a student meets them (course = a snapshot or a course view)."""
    out = []
    for lesson in course.get("lessons", []):
        if (lesson.get("notes") or "").strip():
            out.append(f"{lesson['lesson_id']}:notes")
        out += [f"{lesson['lesson_id']}:{a['asset_id']}" for a in lesson.get("assets", [])]
    return out


def get(store, room_id: str, user_id: str, course_id: str) -> dict:
    item = store.get(_ns(room_id), _key(user_id, course_id)) if store is not None else None
    v = item.value if item else {}
    return {"done": list(v.get("done", [])), "last": v.get("last", ""), "time": v.get("time", "")}


def mark(store, room_id: str, user_id: str, course: dict, item_id: str, done: bool = True, opened: bool = False) -> dict:
    """Record that the student finished (or un-finished) an item, or only opened it (`opened`: remember where they are)."""
    valid = item_ids(course)
    current = get(store, room_id, user_id, course["course_id"])
    if item_id not in valid:
        return current
    finished = [i for i in current["done"] if i in valid]
    if not opened:
        if done and item_id not in finished:
            finished.append(item_id)
        if not done:
            finished = [i for i in finished if i != item_id]
    value = {"done": finished, "last": item_id, "time": _now()}
    store.put(_ns(room_id), _key(user_id, course["course_id"]), value)
    return value


def percent(done: list[str], course: dict) -> int:
    valid = item_ids(course)
    return round(100 * len([i for i in done if i in valid]) / len(valid)) if valid else 0


def summary(store, room_id: str, course: dict) -> dict:
    """For the teacher: how many students started the course, finished it, and the average progress."""
    if store is None or not room_id:
        return {"started": 0, "completed": 0, "average": 0}
    suffix = "|" + course["course_id"]
    rows = [i.value for i in store.search(_ns(room_id), limit=1000) if i.key.endswith(suffix)]
    pcts = [percent(r.get("done", []), course) for r in rows]
    return {"started": len(rows), "completed": sum(1 for p in pcts if p >= 100),
            "average": round(sum(pcts) / len(pcts)) if pcts else 0}

"""My Learning Platform: a teacher's courses, their lessons, and what students of a room see.

A course holds ordered lessons; a lesson holds library assets (recorded/generated videos, decks, quizzes,
saved video moments) and may point at the Lesson Studio session it came from. Publishing a course to a
room writes a SNAPSHOT students read, so editing never changes what they see until it is published again.

    ("lecturer", <teacher_id>, "courses")   key = course_id ("c_" + 8 hex)   the teacher's working copy
    ("class", <room_id>, "courses")         key = course_id                   the published snapshot
"""

import secrets
from datetime import datetime, timezone

from rafeeq.teacher import assets
from rafeeq.teacher.lecturer.long_term_store import lecturer_key

MAX_LESSONS = 60
MAX_ASSETS_PER_LESSON = 20


class CourseError(ValueError):
    """Bad course input."""


def _ns(teacher_id: str) -> tuple:
    return ("lecturer", lecturer_key(teacher_id), "courses")


def _room_ns(room_id: str) -> tuple:
    return ("class", room_id, "courses")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _text(value, limit: int) -> str:
    return str(value or "").strip()[:limit]


# ------------------------------------------------------------------ courses
def get_course(store, teacher_id: str, course_id: str) -> dict | None:
    if store is None or not teacher_id or not course_id:
        return None
    item = store.get(_ns(teacher_id), course_id)
    return {"course_id": course_id, **item.value} if item else None


def list_courses(store, teacher_id: str) -> list[dict]:
    if store is None or not teacher_id:
        return []
    found = [{"course_id": i.key, **i.value} for i in store.search(_ns(teacher_id), limit=200)]
    return sorted(found, key=lambda c: (c.get("created_at", ""), c["course_id"]), reverse=True)


def _save(store, teacher_id: str, course: dict, touch: bool = True) -> dict:
    course = dict(course)
    course_id = course.pop("course_id")
    if touch:
        course["updated_at"] = _now()
        course["revision"] = int(course.get("revision", 0)) + 1      # counts edits (clock ticks can tie)
    store.put(_ns(teacher_id), course_id, course)
    return {"course_id": course_id, **course}


def create_course(store, teacher_id: str, title: str, description: str = "", room_id: str = "") -> dict:
    if store is None or not teacher_id:
        raise CourseError("A teacher is required")
    title = _text(title, 160)
    if len(title) < 2:
        raise CourseError("Give the course a title")
    now = _now()
    course_id = f"c_{secrets.token_hex(4)}"
    record = {"title": title, "description": _text(description, 1000), "room_id": room_id, "lessons": [],
              "published_room_id": "", "published_at": "", "created_at": now, "updated_at": now}
    store.put(_ns(teacher_id), course_id, record)
    return {"course_id": course_id, **record}


def update_course(store, teacher_id: str, course_id: str, *, title: str | None = None,
                  description: str | None = None, room_id: str | None = None) -> dict | None:
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return None
    if title is not None:
        if len(_text(title, 160)) < 2:
            raise CourseError("Give the course a title")
        course["title"] = _text(title, 160)
    if description is not None:
        course["description"] = _text(description, 1000)
    if room_id is not None:
        course["room_id"] = room_id
    return _save(store, teacher_id, course)


def delete_course(store, teacher_id: str, course_id: str) -> bool:
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return False
    if course.get("published_room_id"):
        store.delete(_room_ns(course["published_room_id"]), course_id)
    store.delete(_ns(teacher_id), course_id)
    return True


# ------------------------------------------------------------------ lessons
def _check_assets(store, teacher_id: str, asset_ids: list[str]) -> list[str]:
    ids = []
    for aid in asset_ids[:MAX_ASSETS_PER_LESSON]:
        if aid not in ids:
            if assets.get_asset(store, teacher_id, aid) is None:
                raise CourseError("A lesson can only use your own library items")
            ids.append(aid)
    return ids


def add_lesson(store, teacher_id: str, course_id: str, title: str, *, session_id: str = "", notes: str = "",
               asset_ids: list[str] | None = None) -> dict | None:
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return None
    if len(course["lessons"]) >= MAX_LESSONS:
        raise CourseError("This course has too many lessons")
    title = _text(title, 160)
    if len(title) < 2:
        raise CourseError("Give the lesson a title")
    course["lessons"].append({"lesson_id": f"l_{secrets.token_hex(4)}", "title": title, "notes": _text(notes, 2000),
                              "session_id": session_id, "asset_ids": _check_assets(store, teacher_id, asset_ids or [])})
    return _save(store, teacher_id, course)


def _lesson(course: dict, lesson_id: str) -> dict:
    lesson = next((l for l in course["lessons"] if l["lesson_id"] == lesson_id), None)
    if lesson is None:
        raise CourseError("No such lesson")
    return lesson


def update_lesson(store, teacher_id: str, course_id: str, lesson_id: str, *, title: str | None = None,
                  notes: str | None = None) -> dict | None:
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return None
    lesson = _lesson(course, lesson_id)
    if title is not None:
        if len(_text(title, 160)) < 2:
            raise CourseError("Give the lesson a title")
        lesson["title"] = _text(title, 160)
    if notes is not None:
        lesson["notes"] = _text(notes, 2000)
    return _save(store, teacher_id, course)


def remove_lesson(store, teacher_id: str, course_id: str, lesson_id: str) -> dict | None:
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return None
    _lesson(course, lesson_id)
    course["lessons"] = [l for l in course["lessons"] if l["lesson_id"] != lesson_id]
    return _save(store, teacher_id, course)


def move_lesson(store, teacher_id: str, course_id: str, lesson_id: str, direction: str) -> dict | None:
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return None
    if direction not in ("up", "down"):
        raise CourseError("direction must be up or down")
    lessons = course["lessons"]
    _lesson(course, lesson_id)
    i = next(n for n, l in enumerate(lessons) if l["lesson_id"] == lesson_id)
    j = i - 1 if direction == "up" else i + 1
    if 0 <= j < len(lessons):
        lessons[i], lessons[j] = lessons[j], lessons[i]
    return _save(store, teacher_id, course)


def set_lesson_assets(store, teacher_id: str, course_id: str, lesson_id: str, *, add: str = "",
                      remove: str = "") -> dict | None:
    """Add one library item to a lesson, or take one out."""
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return None
    lesson = _lesson(course, lesson_id)
    ids = list(lesson["asset_ids"])
    if remove:
        ids = [a for a in ids if a != remove]
    if add:
        ids = _check_assets(store, teacher_id, ids + [add])
    lesson["asset_ids"] = ids
    return _save(store, teacher_id, course)


# ------------------------------------------------------------------ publishing
def _snapshot(store, teacher_id: str, course: dict) -> dict:
    lessons = []
    for lesson in course["lessons"]:
        items = []
        for aid in lesson["asset_ids"]:
            a = assets.get_asset(store, teacher_id, aid)
            if a is None or a["status"] != "ready":
                continue
            items.append({"asset_id": aid, "title": a["title"], "kind": a["kind"], "source_url": a.get("source_url", ""),
                          "start_sec": a.get("start_sec"), "end_sec": a.get("end_sec"),
                          "has_file": assets.asset_file(a) is not None})
        lessons.append({"lesson_id": lesson["lesson_id"], "title": lesson["title"], "notes": lesson["notes"],
                        "assets": items})
    return {"teacher_id": teacher_id, "title": course["title"], "description": course["description"],
            "published_at": _now(), "lessons": lessons}


def publish(store, teacher_id: str, course_id: str, room_id: str) -> dict | None:
    """Make the course visible to the students of room_id (the caller checks the teacher owns the room)."""
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return None
    if not room_id:
        raise CourseError("Choose a room to publish to")
    if not course["lessons"]:
        raise CourseError("Add at least one lesson before publishing")
    if course.get("published_room_id") and course["published_room_id"] != room_id:
        store.delete(_room_ns(course["published_room_id"]), course_id)
    snap = _snapshot(store, teacher_id, course)
    store.put(_room_ns(room_id), course_id, snap)
    course.update(room_id=room_id, published_room_id=room_id, published_at=snap["published_at"],
                  updated_at=snap["published_at"], published_revision=int(course.get("revision", 0)))
    return _save(store, teacher_id, course, touch=False)


def unpublish(store, teacher_id: str, course_id: str) -> dict | None:
    course = get_course(store, teacher_id, course_id)
    if course is None:
        return None
    if course.get("published_room_id"):
        store.delete(_room_ns(course["published_room_id"]), course_id)
    course.update(published_room_id="", published_at="")
    return _save(store, teacher_id, course)


def has_unpublished_changes(course: dict) -> bool:
    return bool(course.get("published_room_id")) and int(course.get("revision", 0)) > int(course.get("published_revision", 0))


def published_for_room(store, room_id: str) -> list[dict]:
    if store is None or not room_id:
        return []
    found = [{"course_id": i.key, **i.value} for i in store.search(_room_ns(room_id), limit=100)]
    return sorted(found, key=lambda c: c.get("published_at", ""), reverse=True)


def published_owner(store, room_id: str, course_id: str, asset_id: str) -> str:
    """The teacher who owns `asset_id` IF it is in the published course of this room, else ''."""
    item = store.get(_room_ns(room_id), course_id) if store is not None else None
    if item is None:
        return ""
    for lesson in item.value["lessons"]:
        if any(a["asset_id"] == asset_id for a in lesson["assets"]):
            return item.value["teacher_id"]
    return ""

"""My library & learning platform (M5, section C): deterministic checks of the rules students and teachers rely on.

No model involved: a scripted class (one teacher, a course of 2 lessons, 3 students) runs through the real library,
course and progress code on an in-memory store. Each metric is the share of its checks that hold (all must hold)."""
from __future__ import annotations

from evals.core import Metric
from evals.fakes import FakeStore
from rafeeq.teacher import assets, course_progress as cp, courses

NAME = "learning_platform"
DESCRIPTION = "Library versions, course publishing snapshot, access rules, student progress and the teacher's class summary"
ROOM = "r_eval"
T = "teacher_eval"


def _asset(store, title, kind="video", status="ready", teacher=T, **kw):
    return assets.register_asset(store, teacher, kind=kind, title=title, origin="upload", status=status, **kw)


def _share(checks: dict) -> float:
    return 100 * sum(bool(v) for v in checks.values()) / len(checks)


def run(ctx: dict) -> list[Metric]:
    store = FakeStore()
    out: list[Metric] = []

    # ---- library: versions and repeat-safe registration
    a1 = _asset(store, "Lecture 1 recording")
    v2 = assets.new_version(store, T, a1["asset_id"], title="Lecture 1 (montage)")
    again = _asset(store, "Lecture 1 recording", idempotency_key="upload-1")
    again2 = _asset(store, "Lecture 1 recording", idempotency_key="upload-1")
    lib = {
        "edit makes version 2": v2["version"] == 2 and v2["parent_asset_id"] == a1["asset_id"],
        "original kept unchanged": assets.get_asset(store, T, a1["asset_id"])["title"] == "Lecture 1 recording",
        "edited copy marked 'edited'": v2["origin"] == "edited",
        "same upload twice = one item": again["asset_id"] == again2["asset_id"],
    }
    out.append(Metric("library_versions", _share(lib), "%", gate=100.0,
                      note="every edit is a new version linked to its parent; the original is never overwritten; retries do not duplicate", detail=lib))

    # ---- course + publish snapshot
    slides = _asset(store, "Slides", kind="pptx")
    pending = _asset(store, "Still rendering", status="pending")
    other_teacher = _asset(store, "Not mine", teacher="someone_else")
    course = courses.create_course(store, T, "Biology basics")
    cid = course["course_id"]
    courses.add_lesson(store, T, cid, "The cell", notes="Read pages 4-9", asset_ids=[a1["asset_id"], slides["asset_id"]])
    courses.add_lesson(store, T, cid, "The heart", asset_ids=[v2["asset_id"], pending["asset_id"]])
    courses.publish(store, T, cid, ROOM)
    snap = courses.published_for_room(store, ROOM)[0]
    courses.add_lesson(store, T, cid, "Draft lesson", notes="not ready")
    snap_after_edit = courses.published_for_room(store, ROOM)[0]
    working = courses.get_course(store, T, cid)
    try:
        courses.add_lesson(store, T, cid, "Stolen", asset_ids=[other_teacher["asset_id"]])
        foreign_refused = False
    except courses.CourseError:
        foreign_refused = True
    pub = {
        "students see 2 lessons": len(snap["lessons"]) == 2,
        "editing does not change what students see": len(snap_after_edit["lessons"]) == 2,
        "teacher is told there are unpublished changes": courses.has_unpublished_changes(working),
        "a file still rendering is hidden": all(a["asset_id"] != pending["asset_id"] for l in snap["lessons"] for a in l["assets"]),
        "publishing again shows the new lesson": (courses.publish(store, T, cid, ROOM) is not None
                                                  and len(courses.published_for_room(store, ROOM)[0]["lessons"]) == 3),
    }
    out.append(Metric("publish_snapshot", _share(pub), "%", gate=100.0,
                      note="students read a published snapshot: drafts and unfinished files never leak", detail=pub))

    acc = {
        "another teacher's file cannot be added": foreign_refused,
        "a published file opens for the room": courses.published_owner(store, ROOM, cid, a1["asset_id"]) == T,
        "an unpublished file does not": courses.published_owner(store, ROOM, cid, pending["asset_id"]) == "",
        "another room sees nothing": courses.published_owner(store, "r_other", cid, a1["asset_id"]) == "",
    }
    out.append(Metric("access_rules", _share(acc), "%", gate=100.0, note="what a student can open = what is in the published course of their room", detail=acc))

    # ---- student progress (on the first published snapshot: notes + 2 files in lesson 1, 1 file in lesson 2)
    view = {"course_id": cid, **snap}
    items = cp.item_ids(view)
    s1, s2, s3 = "student_a", "student_b", "student_c"
    for i in items:
        cp.mark(store, ROOM, s1, view, i)
    cp.mark(store, ROOM, s2, view, items[0])
    cp.mark(store, ROOM, s2, view, items[1])
    before = cp.get(store, ROOM, s2, cid)["done"]
    cp.mark(store, ROOM, s2, view, "l_fake:a_fake")
    cp.mark(store, ROOM, s3, view, items[0])
    cp.mark(store, ROOM, s3, view, items[0], done=False)
    cp.mark(store, ROOM, s3, view, items[2], opened=True)
    prog = {
        "items in study order (notes first)": len(items) == 4 and items[0].endswith(":notes"),
        "all done = 100 %": cp.percent(cp.get(store, ROOM, s1, cid)["done"], view) == 100,
        "2 of 4 = 50 %": cp.percent(cp.get(store, ROOM, s2, cid)["done"], view) == 50,
        "an invented item is ignored": cp.get(store, ROOM, s2, cid)["done"] == before,
        "opening is remembered but is not 'done'": cp.get(store, ROOM, s3, cid)["last"] == items[2] and cp.percent(cp.get(store, ROOM, s3, cid)["done"], view) == 0,
        "un-marking works": items[0] not in cp.get(store, ROOM, s3, cid)["done"],
    }
    out.append(Metric("progress_tracking", _share(prog), "%", gate=100.0, note="per student: what is done, where they stopped, the % shown in the player", detail=prog))

    summ = cp.summary(store, ROOM, view)
    expected = {"started": 3, "completed": 1, "average": round((100 + 50 + 0) / 3)}
    out.append(Metric("class_summary_exact", 100.0 if summ == expected else 0.0, "%", gate=100.0,
                      note=f"teacher's course card for 3 students (100 %, 50 %, 0 %): expected {expected}, got {summ}",
                      detail={"expected": expected, "got": summ}))
    return out

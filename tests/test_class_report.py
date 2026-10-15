"""M5: the lecturer's class report and insights, computed from recorded student activity.

No LLM, database or network: an in-memory LangGraph store, and the LLM topic step is switched off.
    uv run pytest tests/test_class_report.py
"""
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

pytest.importorskip("langgraph")
from langgraph.store.memory import InMemoryStore  # noqa: E402

from rafeeq.teacher import class_report as cr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPEC = yaml.safe_load((ROOT / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))
ROOM = "r_bio101"


def assert_matches(schema_name, data):
    schema = {"$ref": f"#/components/schemas/{schema_name}", "components": SPEC["components"]}
    errors = list(Draft202012Validator(schema).iter_errors(json.loads(json.dumps(data))))
    assert not errors, f"{schema_name}: " + "; ".join(f"{list(e.path)}: {e.message}" for e in errors)


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(cr, "_llm_topics", lambda texts, candidates: None)  # no LLM in tests
    monkeypatch.setattr(cr, "_students", lambda room_id: {"u_omar": "عمر خالد", "u_sara": "سارة محمود"})
    return InMemoryStore()


def quiz(store, user, topic, right, total, time=None):
    answers = [{"question": f"س{i}", "chosen": "صح" if i < right else "خطأ", "correct": "صح"} for i in range(total)]
    assert cr.record_quiz_result(ROOM, user, topic, answers, time=time, store=store)


def test_empty_room_gives_empty_but_valid_report_and_insights(store):
    assert cr.build_report(store, ROOM) == {"topics": [], "students_at_risk": []}
    assert cr.build_insights(store, ROOM) == {"top_questions": [], "common_mistakes": []}
    assert cr.build_report(None, ROOM)["topics"] == []          # no store: still the contract shape


def test_report_ranks_topics_weakest_first_and_flags_struggling_students(store):
    quiz(store, "u_omar", "الوراثة", 1, 4)
    quiz(store, "u_sara", "الوراثة", 2, 4)
    quiz(store, "u_omar", "الخلية", 4, 4)
    quiz(store, "u_sara", "الخلية", 3, 4)
    report = cr.build_report(store, ROOM)
    assert_matches("ClassReport", report)
    assert [t["topic_id"] for t in report["topics"]] == ["الوراثة", "الخلية"]
    assert report["topics"][0] == {"topic_id": "الوراثة", "mastery": 0.38, "status": "struggling"}
    assert report["topics"][1]["status"] == "on_track"
    # Omar 0.25 is struggling; Sara 0.5 is only "watch", so she is not at risk
    assert report["students_at_risk"] == [{"user_id": "u_omar", "name": "عمر خالد", "topic_ids": ["الوراثة"]}]


def test_recent_quizzes_weigh_more_so_improvement_shows(store, monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(cr, "_now", lambda: datetime(2026, 10, 3, tzinfo=timezone.utc))
    quiz(store, "u_omar", "الوراثة", 0, 4, time="2026-10-01T10:00:00+00:00")
    quiz(store, "u_omar", "الوراثة", 4, 4, time="2026-10-02T10:00:00+00:00")
    assert cr.student_levels(store, ROOM)["u_omar"] == (0.5, "watch")          # 0 then 1 -> 0.5
    quiz(store, "u_omar", "الوراثة", 4, 4, time="2026-10-02T12:00:00+00:00")
    assert cr.student_levels(store, ROOM)["u_omar"] == (0.75, "on_track")


def test_insights_group_questions_by_topic_and_name_repeated_mistakes(store):
    cr.record_question(ROOM, "u_omar", "ما الفرق بين الجين والكروموسوم؟", store=store, topic_id="الوراثة")
    cr.record_question(ROOM, "u_sara", "يعني إيه أليل؟", store=store, topic_id="الوراثة")
    cr.record_question(ROOM, "u_sara", "وظيفة الجدار الخلوي؟", store=store, topic_id="الخلية")
    wrong = [{"question": "أين توجد المادة الوراثية؟", "chosen": "الفجوة", "correct": "النواة"}]
    cr.record_quiz_result(ROOM, "u_omar", "الوراثة", wrong, store=store)
    cr.record_quiz_result(ROOM, "u_sara", "الوراثة", wrong, store=store)
    insights = cr.build_insights(store, ROOM)
    assert_matches("Insights", insights)
    assert insights["top_questions"][0]["topic_id"] == "الوراثة" and insights["top_questions"][0]["count"] == 2
    mistake = insights["common_mistakes"][0]
    assert mistake["topic_id"] == "الوراثة"
    assert "2 طلاب" in mistake["description"] and "الفجوة" in mistake["description"] and "النواة" in mistake["description"]


def test_unlabelled_questions_get_the_matching_session_topic_once(store):
    store.put(("lecturer", "dr-laila", "sessions"), "s1", {"topic": "البناء الضوئي في النبات", "room_id": ROOM})
    store.put(("lecturer", "dr-laila", "sessions"), "s2", {"topic": "مكان آخر", "room_id": "r_other"})
    cr.record_question(ROOM, "u_omar", "ما معنى البناء الضوئي؟", store=store)
    cr.record_question(ROOM, "u_omar", "متى يبدأ الامتحان؟", store=store)
    topics = {q["topic_id"] for q in cr.build_insights(store, ROOM)["top_questions"]}
    assert topics == {"البناء الضوئي في النبات", cr.GENERAL}
    saved = [i.value["topic_id"] for i in store.search(("class", ROOM, "questions"))]
    assert None not in saved                                    # classified once, stored


def test_old_activity_drops_out_of_the_report(store):
    quiz(store, "u_omar", "الوراثة", 0, 4, time="2020-01-01T00:00:00+00:00")
    cr.record_question(ROOM, "u_omar", "سؤال قديم", store=store, topic_id="الوراثة", time="2020-01-01T00:00:00+00:00")
    assert cr.build_report(store, ROOM)["topics"] == []
    assert cr.build_insights(store, ROOM)["top_questions"] == []


def test_recording_never_raises_and_ignores_empty_input(store):
    assert not cr.record_question(ROOM, "u_omar", "   ", store=store)
    if cr.get_store() is None:                                  # no store at all: nothing to record into
        assert not cr.record_question(ROOM, "u_omar", "سؤال", store=None)
    assert not cr.record_quiz_result(ROOM, "u_omar", "الوراثة", [], store=store)


def test_report_endpoints_serve_the_computed_data(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr("rafeeq.teacher.lecturer.settings.DATABASE_URL", None)
    monkeypatch.setattr(cr, "_llm_topics", lambda texts, candidates: None)
    monkeypatch.setattr(cr, "_students", lambda room_id: {})
    from rafeeq.main import app

    with TestClient(app) as client:
        store = app.state.teacher.store
        assert cr.get_store() is store                          # the platform router records into it
        cr.record_question("r_api", "u_x", "سؤال عن الوراثة", topic_id="الوراثة")
        cr.record_quiz_result("r_api", "u_x", "الوراثة", [{"question": "q", "chosen": "a", "correct": "b"}])
        report = client.get("/teacher/rooms/r_api/report").json()
        insights = client.get("/teacher/rooms/r_api/insights").json()
    assert_matches("ClassReport", report) and report["topics"][0]["status"] == "struggling"
    assert_matches("Insights", insights) and insights["top_questions"][0]["count"] == 1


def test_questions_are_grouped_under_the_rooms_own_topics_when_m2_has_them(store, monkeypatch):
    monkeypatch.setattr(cr, "room_topic_names", lambda room_id: {"topic_01": "Slow-Wave Structures",
                                                                 "topic_02": "Electron-Beam Interaction"})
    cr.record_question(ROOM, "u_omar", "how do slow-wave structures slow the wave?", store=store)
    cr.record_question(ROOM, "u_sara", "what is a Floquet harmonic?", store=store, topic_id="topic_01")
    top = cr.build_insights(store, ROOM)["top_questions"]
    assert top == [{"topic_id": "topic_01", "count": 2, "example": top[0]["example"]}]
    quiz(store, "u_omar", "topic_02", 1, 5)  # M4 sends the same topic ids
    report = cr.build_report(store, ROOM)
    assert report["topics"][0]["topic_id"] == "topic_02"
    assert report["students_at_risk"][0]["topic_ids"] == ["topic_02"]

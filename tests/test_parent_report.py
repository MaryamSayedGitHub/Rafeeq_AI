"""M5: the weekly parent report, built only from recorded activity (no LLM, no network)."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker

pytest.importorskip("langgraph")
from langgraph.store.memory import InMemoryStore  # noqa: E402

from rafeeq.teacher import class_report as cr  # noqa: E402
from rafeeq.teacher import parent_report as pr  # noqa: E402

SPEC = yaml.safe_load((Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))
ROOM, OMAR = "r_twt", "u_omar"


def check(data):
    schema = {"$ref": "#/components/schemas/ParentReport", "components": SPEC["components"]}
    errors = [e.message for e in Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(json.loads(json.dumps(data)))]
    assert not errors, errors


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(cr, "room_topic_names", lambda room_id: {"topic_01": "Slow-Wave Structures",
                                                                 "topic_02": "Electron-Beam Interaction"})
    monkeypatch.setattr(pr, "_room_and_student", lambda room_id, user_id: ("Microwave Tubes", "عمر"))
    return InMemoryStore()


def ago(days):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def quiz(store, topic, right, total, when):
    answers = [{"question": f"q{i}", "chosen": "a" if i < right else "b", "correct": "a"} for i in range(total)]
    cr.record_quiz_result(ROOM, OMAR, topic, answers, time=when, store=store)


def test_a_week_with_activity_is_summarised_from_the_numbers(store):
    quiz(store, "topic_01", 1, 5, ago(10))   # last week: weak
    quiz(store, "topic_01", 5, 5, ago(2))    # this week: much better
    quiz(store, "topic_02", 1, 5, ago(1))    # this week: weak
    cr.record_question(ROOM, OMAR, "what is a helix?", time=ago(1), store=store, topic_id="topic_01")
    cr.record_question(ROOM, "u_other", "not Omar's", time=ago(1), store=store, topic_id="topic_02")
    r = pr.build(store, ROOM, OMAR)
    check(r)
    assert r["student_name"] == "عمر" and r["quizzes_taken"] == 2 and r["questions_answered"] == 10
    assert r["average_score"] == 0.6 and r["questions_asked"] == 1 and r["asked_about"] == ["Slow-Wave Structures"]
    assert r["improved"] == ["Slow-Wave Structures"] and r["needs_work"] == ["Electron-Beam Interaction"]
    text = " ".join(r["summary"])
    assert "6 من 10" in text and "«Slow-Wave Structures»" in text and "«Electron-Beam Interaction»" in text


def test_no_activity_says_so_instead_of_inventing_anything(store):
    r = pr.build(store, ROOM, OMAR)
    check(r)
    assert r["quizzes_taken"] == 0 and r["average_score"] is None and r["topics"] == []
    assert r["summary"] == ["لم يسجّل عمر نشاطًا على رفيق هذا الأسبوع. نقترح تشجيعه على حل اختبار قصير في مادته."]


def test_share_tokens_round_trip_and_reject_tampering(monkeypatch):
    monkeypatch.setenv("RAFEEQ_WEB_SECRET", "test-secret")
    token = pr.make_share_token(ROOM, OMAR)
    assert pr.read_share_token(token) == (ROOM, OMAR)
    assert pr.read_share_token(token[:-2] + "xx") is None
    monkeypatch.setenv("RAFEEQ_WEB_SECRET", "another-secret")
    assert pr.read_share_token(token) is None

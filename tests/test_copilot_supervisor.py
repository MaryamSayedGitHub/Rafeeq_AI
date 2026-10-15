"""The Copilot's routing and honesty rules (no LLM key needed: the rule fallback is used)."""
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from rafeeq.copilot import clients
from rafeeq.copilot.router import rules
from rafeeq.copilot.supervisor import clear_history, handle_message

SPEC = yaml.safe_load((Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))


def assert_reply_schema(reply):
    schema = {"$ref": "#/components/schemas/CopilotReply", "components": SPEC["components"]}
    assert not list(Draft202012Validator(schema).iter_errors(reply))


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("RAFEEQ_ALLOW_SAMPLES", raising=False)
    for name in ("tutor_answer", "check_work", "make_quiz", "next_step", "find_verified", "escalate", "record_event", "overview", "topic_of"):
        clients.disconnect(name)
    clear_history()
    yield
    for name in ("tutor_answer", "check_work", "make_quiz", "next_step", "find_verified", "escalate", "record_event", "overview", "topic_of"):
        clients.disconnect(name)


@pytest.mark.parametrize("text,has_file,route", [
    ("Explain Newton's third law", False, "ask"),
    ("اشرح لي قانون نيوتن الثالث", False, "ask"),
    ("اختبرني في الوراثة", False, "quiz"),
    ("what's my plan for this week", False, "plan"),
    ("ok", False, "clarify"),
    ("صحح حلي", True, "check_work"),
])
def test_rule_routing(text, has_file, route):
    assert rules(text, has_file) == route


def test_explicit_mode_wins_over_classification():
    out = handle_message("r_1", "ما هذا؟", file_id="snap_1", mode="check_work")
    assert out["route"] == "check_work"
    assert_reply_schema(out)


def test_unconnected_services_are_marked_as_samples():
    out = handle_message("r_1", "Explain Newton's third law")
    assert out["route"] == "ask"
    assert out["payload"]["sample"] is True
    assert_reply_schema(out)


def test_samples_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("RAFEEQ_ALLOW_SAMPLES", "0")
    out = handle_message("r_1", "Explain Newton's third law")
    assert out["payload"] == {"unavailable": True}


def test_live_answer_has_no_sample_flag_and_records_an_event():
    events = []
    clients.connect("tutor_answer", lambda room, q: {"answer": "A", "citations": [], "grounded": True,
                                                     "steps": [], "topic_id": "t1"})
    clients.connect("record_event", events.append)
    out = handle_message("r_1", "Explain Newton's third law", user_id="u_1")
    assert "sample" not in out["payload"]
    assert events and events[0]["topic_id"] == "t1" and events[0]["user_id"] == "u_1"


def test_ungrounded_answer_is_not_claimed_as_escalated_without_a_ticket():
    clients.connect("tutor_answer", lambda room, q: {"answer": "", "citations": [], "grounded": False, "steps": []})
    out = handle_message("r_1", "a very hard exam question")
    assert out["payload"]["escalated"] is False
    clients.connect("escalate", lambda room, q, user: {"question_id": "q_9"})
    out = handle_message("r_1", "a very hard exam question")
    assert out["payload"]["escalated"] is True and out["payload"]["question_id"] == "q_9"


def test_check_work_without_photo_asks_for_one():
    out = handle_message("r_1", "check my answer", mode="check_work")
    assert out["payload"] == {"needs_photo": True}


def test_history_is_per_student():
    handle_message("r_1", "Explain cells", user_id="u_a")
    from rafeeq.copilot.supervisor import _HISTORY
    assert ("r_1", "u_a") in _HISTORY and ("r_1", "u_b") not in _HISTORY


def test_with_m4_connected_plan_and_quiz_use_the_students_profile():
    calls = []

    def step(room_id, user_id):
        calls.append((room_id, user_id))
        return {"topic_id": "topic_02", "suggestion": "راجع موضوع «Slow-Wave Structures» ثم جرّب اختبارًا قصيرًا."}

    clients.connect("next_step", step)
    plan = handle_message("r_1", "أذاكر إيه النهارده؟", user_id="u_1")
    assert plan["route"] == "plan" and "Slow-Wave" in plan["reply"] and "sample" not in plan["payload"]
    quiz = handle_message("r_1", "اختبرني", user_id="u_1")
    assert quiz["route"] == "quiz" and quiz["payload"] == {"topic_id": "topic_02"}
    assert calls == [("r_1", "u_1"), ("r_1", "u_1")]
    assert_reply_schema(plan) or assert_reply_schema(quiz)


def test_no_student_profile_means_unavailable_not_a_sample():
    clients.connect("next_step", lambda room_id, user_id: None)
    out = handle_message("r_1", "اختبرني")
    assert out["payload"] == {"unavailable": True}


NOT_COVERED = {"answer": "This topic is not covered in the provided course material.", "citations": [],
               "grounded": False, "steps": []}


def test_a_lesson_level_question_gets_an_explanation_of_the_latest_file():
    events, asked = [], []
    clients.connect("tutor_answer", lambda room_id, q: NOT_COVERED)
    clients.connect("overview", lambda room_id, q: asked.append(q) or {
        "answer": "الدرس عن أنبوب الموجة المتنقلة (p. 3).", "grounded": True, "steps": [],
        "citations": [{"file_name": "twt.pdf", "page": 3}]})
    clients.connect("topic_of", lambda room_id, citations: "topic_02" if citations else None)
    clients.connect("record_event", events.append)
    out = handle_message("r_1", "اشرح لي فكرة الدرس الأخير ببساطة", user_id="u_1")
    assert out["route"] == "ask" and out["payload"]["grounded"] is True
    assert out["reply"].startswith("الدرس عن") and out["payload"]["citations"][0]["page"] == 3
    assert asked and events[0]["topic_id"] == "topic_02"
    assert_reply_schema(out)


def test_a_specific_question_that_is_not_covered_stays_not_covered_without_the_english_text():
    clients.connect("tutor_answer", lambda room_id, q: NOT_COVERED)
    clients.connect("overview", lambda room_id, q: (_ for _ in ()).throw(AssertionError("not a lesson request")))
    out = handle_message("r_1", "ما هي سرعة الضوء في الفراغ؟", user_id="u_1")
    assert out["payload"]["grounded"] is False and out["reply"] == ""

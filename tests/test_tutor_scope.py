"""When the room's material has nothing: the teacher's room setting decides about the web, and a question from another
subject is never searched nor sent to the teacher (rafeeq/tutor/scope.py)."""
import pytest

from rafeeq.platform import service
from rafeeq.tutor import llm, scope


@pytest.fixture()
def room(tmp_path, monkeypatch):
    monkeypatch.setenv("RAFEEQ_DATA_DIR", str(tmp_path))
    service.set_store(None)
    store = service.get_store()
    teacher = store.register("T", "t@x.com", "secret1", "teacher")
    rid = store.create_room(teacher, "Microwave Engineering")["room"]["room_id"]
    monkeypatch.setattr("rafeeq.tutor.m2.topics", lambda room_id: ([{"topic_id": "t1", "name": "Traveling-wave tubes"}], "m2"))
    yield store, teacher, rid
    service.set_store(None)


def test_the_default_allows_the_web_and_the_teacher_can_turn_it_off(room):
    store, teacher, rid = room
    assert scope.web_mode(rid) == scope.MATERIAL_THEN_WEB and scope.web_allowed(rid)
    assert store.room_settings(teacher, rid)["web_answers"] == "material_then_web"
    store.set_room_settings(teacher, rid, "material_only")
    assert not scope.web_allowed(rid)
    with pytest.raises(service.PlatformError):
        store.set_room_settings(teacher, rid, "anything")


def test_only_the_rooms_teacher_changes_the_setting(room):
    store, _, rid = room
    student = store.register("S", "s@x.com", "secret1", "student")
    with pytest.raises(service.PlatformError):
        store.set_room_settings(student, rid, "material_only")


def test_an_unknown_room_keeps_the_old_behaviour():
    assert scope.web_allowed("r_not_in_any_store")


def test_another_subject_is_recognised_by_the_model(room, monkeypatch):
    _, _, rid = room
    asked = []
    monkeypatch.setattr(llm, "ask_json", lambda p, max_tokens=None: asked.append(p) or {"in_subject": False})
    assert scope.in_subject(rid, "What is the capital of Japan?") is False
    assert "Microwave Engineering" in asked[0] and "Traveling-wave tubes" in asked[0]


def test_no_model_or_an_unclear_reply_keeps_answering(room, monkeypatch):
    _, _, rid = room
    monkeypatch.setattr(llm, "ask_json", lambda p, max_tokens=None: None)
    assert scope.in_subject(rid, "What is a klystron?") is True
    monkeypatch.setattr(llm, "ask_json", lambda p, max_tokens=None: {"answer": "?"})
    assert scope.in_subject(rid, "What is a klystron?") is True


def test_the_reply_names_the_course(room):
    _, _, rid = room
    assert "Microwave Engineering" in scope.off_topic_reply(rid, "ar")
    assert "Microwave Engineering" in scope.off_topic_reply(rid, "en")


# ----------------------------------------------------------------------------- the tutor agent (needs langgraph)
def _web_term(rid, text, monkeypatch):
    pytest.importorskip("langgraph")
    from rafeeq.tutor import lookup, study_store, tutor_agent

    searched = []
    monkeypatch.setattr(lookup, "is_course_specific", lambda q: False)
    monkeypatch.setattr(lookup, "web_lookup", lambda *a, **k: searched.append(a) or {"answer": "web says", "web_sources": [{"n": 1, "title": "x", "url": "https://x"}]})
    monkeypatch.setattr(study_store, "update_question", lambda *a, **k: None)
    out = {"language": "en", "question_id": 1, "grounded": False}
    ok = tutor_agent._web_term({"room_id": rid, "text": text, "user_id": None}, out, photo=False)
    return ok, out, searched


def test_material_only_room_sends_the_question_to_the_teacher_without_searching(room, monkeypatch):
    store, teacher, rid = room
    store.set_room_settings(teacher, rid, "material_only")
    ok, out, searched = _web_term(rid, "What is a klystron?", monkeypatch)
    assert not ok and searched == [] and out["needs_teacher"] and out["grounded"] is False


def test_another_subject_is_not_searched_and_not_sent_to_the_teacher(room, monkeypatch):
    _, _, rid = room
    monkeypatch.setattr(scope, "in_subject", lambda room_id, q: False)
    ok, out, searched = _web_term(rid, "What is the capital of Japan?", monkeypatch)
    assert not ok and searched == [] and out["off_topic"] and "Microwave Engineering" in out["answer"]
    assert not out.get("needs_teacher")


def test_a_subject_question_is_answered_from_the_web_and_kept_for_the_teacher(room, monkeypatch):
    _, _, rid = room
    from rafeeq.copilot import clients

    kept = []
    monkeypatch.setattr(scope, "in_subject", lambda room_id, q: True)
    monkeypatch.setitem(clients._CONNECTED, "log_web_answer", lambda *a: kept.append(a) or {"question_id": "q1"})
    ok, out, searched = _web_term(rid, "What is a klystron?", monkeypatch)
    assert ok and out["material_source"] == "web" and searched and kept and kept[0][1] == "What is a klystron?"

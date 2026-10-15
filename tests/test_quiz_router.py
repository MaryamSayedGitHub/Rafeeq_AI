"""The five contract routes, end to end, with a fake LLM and an in-memory database."""
import json
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import rafeeq.quiz.quiz.models  # noqa: F401  (registers the quiz tables on Base)
from rafeeq.quiz.deps import current_user_id, get_chunks, get_llm
from rafeeq.quiz.profile.db import get_session
from rafeeq.quiz.profile.models import Base
from rafeeq.quiz.router import bearer_user_id, router

ROOT = Path(__file__).resolve().parent.parent
SPEC = yaml.safe_load((ROOT / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))

GOOD = {
    "stem": "What does a tokenizer do?",
    "options": ["Splits text into tokens", "Trains the model", "Stores vectors", "Draws images"],
    "right_choice": 0,
    "explanation": "It converts raw text into tokens.",
}
CHUNKS = [
    {"chunk_id": f"b-{t}{p}", "text": "x" * 500, "file_name": "book.pdf",
     "page": p, "topic_id": t, "topic_name": t}
    for t in ("a", "b") for p in range(1, 11)
]


def check(schema: str, data) -> None:
    v = Draft202012Validator({"$ref": f"#/components/schemas/{schema}", "components": SPEC["components"]},
                             format_checker=FormatChecker())
    errors = [e.message for e in v.iter_errors(data)]
    assert not errors, errors


@pytest.fixture
def client():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)

    def _session():
        with Session(engine) as s:
            yield s

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[current_user_id] = lambda: "u1"
    app.dependency_overrides[get_llm] = lambda: (lambda prompt: json.dumps(GOOD))
    app.dependency_overrides[get_chunks] = lambda: CHUNKS
    return TestClient(app)


def make(client, **extra):
    body = {"room_id": "r1", "topic_id": "", "count": 3, **extra}
    r = client.post("/quiz", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def answers_for(quiz, right=True):
    out = []
    for q in quiz["questions"]:
        if right:
            out.append(q["correct_answer"])
        else:
            out.append(next(c for c in q["choices"] if c != q["correct_answer"]))
    return out


def test_make_quiz_matches_the_contract(client):
    quiz = make(client)
    check("Quiz", quiz)
    assert len(quiz["questions"]) == 3
    for q in quiz["questions"]:
        assert q["correct_answer"] in q["choices"]


def test_make_quiz_on_one_topic(client):
    quiz = make(client, topic_id="b")
    check("Quiz", quiz)
    assert quiz["topic_id"] == "b"


def test_unknown_topic_is_404(client):
    r = client.post("/quiz", json={"room_id": "r1", "topic_id": "nope", "count": 3})
    assert r.status_code == 404


def test_submit_matches_the_contract(client):
    quiz = make(client)
    r = client.post(f"/quiz/{quiz['quiz_id']}/submit", json={"answers": answers_for(quiz)})
    assert r.status_code == 200, r.text
    result = r.json()
    check("QuizResult", result)
    assert (result["score"], result["total"]) == (3, 3)
    assert len(result["explanations"]) == 3
    assert result["mastery_change"]  # at least the topic of the quiz


def test_wrong_answers_score_zero(client):
    quiz = make(client)
    r = client.post(f"/quiz/{quiz['quiz_id']}/submit", json={"answers": answers_for(quiz, right=False)})
    assert r.json()["score"] == 0


def test_a_quiz_can_be_submitted_once(client):
    quiz = make(client)
    body = {"answers": answers_for(quiz)}
    assert client.post(f"/quiz/{quiz['quiz_id']}/submit", json=body).status_code == 200
    assert client.post(f"/quiz/{quiz['quiz_id']}/submit", json=body).status_code == 409


def test_submit_unknown_quiz_is_404(client):
    assert client.post("/quiz/nope/submit", json={"answers": []}).status_code == 404


def test_profile_summary_matches_the_contract(client):
    quiz = make(client)
    client.post(f"/quiz/{quiz['quiz_id']}/submit", json={"answers": answers_for(quiz, right=False)})
    r = client.get("/profile/summary", params={"room_id": "r1"})
    assert r.status_code == 200
    check("StudentSummary", r.json())


def test_profile_summary_for_a_new_student(client):
    r = client.get("/profile/summary", params={"room_id": "r1"})
    assert r.json() == {"weak_topics": [], "level": "beginner"}


def test_next_step_matches_the_contract(client):
    r = client.get("/next-step", params={"room_id": "r1"})   # new student: first topic of the course
    assert r.status_code == 200
    check("NextStep", r.json())
    quiz = make(client)
    client.post(f"/quiz/{quiz['quiz_id']}/submit", json={"answers": answers_for(quiz, right=False)})
    r = client.get("/next-step", params={"room_id": "r1"})
    check("NextStep", r.json())


def test_question_event_is_recorded_without_changing_mastery(client):
    ev = {"user_id": "u1", "room_id": "r1", "topic_id": "a", "kind": "question",
          "text": "why?", "time": "2026-10-09T10:30:00Z"}
    r = client.post("/events", json=ev)
    assert r.status_code == 204 and r.content == b""
    topic = client.get("/rooms/r1/me/profile").json()["topics"][0]
    assert (topic["topic_id"], topic["questions_asked"], topic["mastery"]) == ("a", 1, 0.5)


def test_understood_events_are_accepted_and_ignored(client):
    ev = {"user_id": "u1", "room_id": "r1", "topic_id": "a", "kind": "understood",
          "text": "ok", "time": "2026-10-09T10:30:00Z"}
    assert client.post("/events", json=ev).status_code == 204
    assert client.get("/rooms/r1/me/profile").json()["topics"] == []


def test_the_topics_route_is_left_to_m2(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert "/rooms/{room_id}/topics" not in paths
    assert "/quizzes/{quiz_id}/sheet.pdf" in paths   # M4's own extras are still there
    for contract_path in ("/quiz", "/quiz/{quiz_id}/submit", "/next-step", "/profile/summary", "/events"):
        assert contract_path in paths


def test_the_login_check_rejects_a_missing_token():
    with pytest.raises(HTTPException) as e:
        bearer_user_id(None)
    assert e.value.status_code == 401


# ---------- the room's own material (M2) instead of the development book ----------

ROOM = [
    {"chunk_id": f"r-{t}{p}", "text": "y" * 500, "file_name": "twt.pdf", "page": p,
     "topic_id": t, "topic_name": n}
    for t, n in (("topic_01", "Slow-Wave Structures"), ("topic_02", "Electron-Beam Interaction")) for p in range(1, 6)
]


def test_quiz_uses_the_rooms_material_when_the_knowledge_base_has_it(client):
    from rafeeq.quiz.deps import get_room_chunks

    client.app.dependency_overrides[get_room_chunks] = lambda: (lambda room_id: ROOM if room_id == "r1" else [])
    quiz = make(client, topic_id="topic_01")
    assert quiz["topic_id"] == "topic_01"
    step = client.get("/next-step", params={"room_id": "r1"})
    assert step.status_code == 200 and step.json()["topic_id"] in {"topic_01", "topic_02"}
    r = client.post("/quiz", json={"room_id": "r1", "topic_id": "a", "count": 3})
    assert r.status_code == 404  # the book's topics are not this room's topics


def test_a_room_with_nothing_indexed_has_no_topics_instead_of_the_book(client):
    from rafeeq.quiz.deps import get_room_chunks

    client.app.dependency_overrides[get_room_chunks] = lambda: (lambda room_id: [])
    assert client.get("/next-step", params={"room_id": "r1"}).status_code == 404


def test_a_missing_llm_key_is_a_503_not_a_crash(client, monkeypatch):
    from rafeeq.quiz import deps

    client.app.dependency_overrides.pop(get_llm)
    deps._llm.cache_clear()
    monkeypatch.setenv("M4_LLM_PROVIDER", "compatible")
    monkeypatch.setenv("LLM_API_KEY", "")
    try:
        r = client.post("/quiz", json={"room_id": "r1", "topic_id": "", "count": 3})
        assert r.status_code == 503 and "not configured" in r.json()["detail"]
    finally:
        deps._llm.cache_clear()


def test_copilot_hooks_give_a_next_step_from_the_rooms_material(monkeypatch):
    from rafeeq.quiz import copilot_hooks, deps

    monkeypatch.setattr(deps, "room_chunks", lambda room_id: ROOM)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(copilot_hooks, "get_engine", lambda: engine)
    step = copilot_hooks.next_step("r1", "u1")
    assert step["topic_id"] in {"topic_01", "topic_02"} and "«" in step["suggestion"]
    assert copilot_hooks.next_step("r1", None) is None
    copilot_hooks.record_event({"user_id": "u1", "room_id": "r1", "topic_id": "topic_01", "kind": "question",
                                "text": "what is a TWT?", "time": "2026-10-13T20:00:00+00:00"})
    copilot_hooks.record_event({"bad": "event"})  # never raises


def test_a_submitted_quiz_feeds_the_teachers_class_report(client, monkeypatch):
    from rafeeq.teacher import class_report

    sent = []
    monkeypatch.setattr(class_report, "record_quiz_result",
                        lambda room_id, user_id, topic_id, answers, **kw: sent.append((room_id, user_id, topic_id, answers)) or True)
    quiz = make(client)
    r = client.post(f"/quiz/{quiz['quiz_id']}/submit", json={"answers": answers_for(quiz, right=False)})
    assert r.status_code == 200
    assert sent and all(room == "r1" and user == "u1" for room, user, _, _ in sent)
    rows = [row for *_, answers in sent for row in answers]
    assert len(rows) == len(quiz["questions"]) and all(row["chosen"] != row["correct"] for row in rows)

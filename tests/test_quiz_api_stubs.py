import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import rafeeq.quiz.quiz.models  # noqa: F401  (registers the quiz tables on Base)
from rafeeq.quiz.api import router
from rafeeq.quiz.deps import current_user_id, get_chunks, get_llm
from rafeeq.quiz.profile.db import get_session
from rafeeq.quiz.profile.models import Base

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


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
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


def _event(topic="t1", kind="quiz", score=0.0, user="u1"):
    return {"user_id": user, "room_id": "r1", "topic_id": topic,
            "kind": kind, "score": score, "meta": {}}


def _new_quiz(client, n=3):
    r = client.post("/rooms/r1/quizzes", json={"n": n, "mode": "screen"})
    assert r.status_code == 200
    return r.json()


def _answers(quiz, right=True):
    return {"answers": [
        {"item_id": i["item_id"],
         "choice": i["right_choice"] if right else (i["right_choice"] + 1) % 4}
        for i in quiz["items"]]}


def test_profile_starts_empty(client):
    r = client.get("/rooms/r1/me/profile")
    assert r.status_code == 200 and r.json()["topics"] == []


def test_event_changes_mastery(client):
    r = client.post("/internal/record_event", json=_event(score=1.0))
    assert r.status_code == 200 and r.json()["mastery"] == pytest.approx(0.65)
    topics = client.get("/rooms/r1/me/profile").json()["topics"]
    assert topics[0]["topic_id"] == "t1"
    assert topics[0]["mastery"] == pytest.approx(0.65)


def test_two_failed_quizzes_make_topic_weak(client):
    for _ in range(2):
        client.post("/internal/record_event", json=_event(score=0.0))
    assert client.get("/rooms/r1/me/profile").json()["topics"][0]["weak"] is True


def test_profile_is_per_user(client):
    client.post("/internal/record_event", json=_event(user="someone-else"))
    assert client.get("/rooms/r1/me/profile").json()["topics"] == []


def test_create_quiz_returns_four_option_items(client):
    quiz = _new_quiz(client, 3)
    assert len(quiz["items"]) == 3
    assert all(len(i["options"]) == 4 for i in quiz["items"])


def test_submit_changes_the_profile_and_cannot_repeat(client):
    quiz = _new_quiz(client)
    r = client.post(f"/quizzes/{quiz['quiz_id']}/submit", json=_answers(quiz))
    assert r.status_code == 200 and r.json()["score"] == 1.0
    topics = client.get("/rooms/r1/me/profile").json()["topics"]
    assert topics and all(t["mastery"] > 0.5 for t in topics)
    again = client.post(f"/quizzes/{quiz['quiz_id']}/submit", json=_answers(quiz))
    assert again.status_code == 409


def test_submit_unknown_quiz_is_404(client):
    r = client.post("/quizzes/nope/submit", json={"answers": []})
    assert r.status_code == 404


def test_wrong_answers_appear_without_names(client):
    quiz = _new_quiz(client)
    client.post(f"/quizzes/{quiz['quiz_id']}/submit", json=_answers(quiz, right=False))
    rows = client.get("/rooms/r1/wrong-answers").json()
    assert len(rows) == 3
    for row in rows:
        assert "name" not in row and "user_id" not in row


def test_class_profile_lists_the_student(client):
    client.post("/internal/record_event", json=_event())
    rows = client.get("/rooms/r1/class-profile").json()
    assert rows[0]["user_id"] == "u1"


def _photo_for(quiz, picks):
    import numpy as np
    from rafeeq.quiz.sheet import synthetic
    return synthetic.fake_photo(synthetic.draw_sheet(picks), np.random.default_rng(2))


def test_grade_sheet_reads_the_photo_and_flags_blanks(client):
    quiz = _new_quiz(client, 5)
    rights = [i["right_choice"] for i in quiz["items"]]
    picks = [rights[0], rights[1], (rights[2] + 1) % 4, None, rights[4]]  # 3 right, 1 wrong, 1 blank
    r = client.post(f"/quizzes/{quiz['quiz_id']}/grade-sheet",
                    files={"photo": ("p.jpg", _photo_for(quiz, picks), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert [a["choice"] for a in body["answers"]] == picks
    assert body["score"] == pytest.approx(0.6)
    assert body["needs_review"] == [quiz["items"][3]["item_id"]]
    # grading a photo saves nothing: the profile is still empty
    assert client.get("/rooms/r1/me/profile").json()["topics"] == []


def test_grade_sheet_rejects_a_photo_that_is_not_a_sheet(client):
    quiz = _new_quiz(client, 3)
    r = client.post(f"/quizzes/{quiz['quiz_id']}/grade-sheet",
                    files={"photo": ("p.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 422


def test_grade_sheet_unknown_quiz_is_404(client):
    r = client.post("/quizzes/nope/grade-sheet", files={"photo": ("p.jpg", b"x", "image/jpeg")})
    assert r.status_code == 404
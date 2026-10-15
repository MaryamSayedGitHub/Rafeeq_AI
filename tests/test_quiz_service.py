import json
import random

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import rafeeq.quiz.quiz.models  # noqa: F401  (registers the quiz tables on Base)
from rafeeq.quiz.contracts import Answer, Event, QuizItem
from rafeeq.quiz.profile import service as profile_service
from rafeeq.quiz.profile.models import Base
from rafeeq.quiz.quiz import service as quiz_service

GOOD = {
    "stem": "What does a tokenizer do?",
    "options": ["Splits text into tokens", "Trains the model", "Stores vectors", "Draws images"],
    "right_choice": 0,
    "explanation": "It converts raw text into tokens.",
}


def fake_llm(prompt):
    return json.dumps(GOOD)


def chunks():
    return [
        {"chunk_id": f"b-{t}{p}", "text": "x" * 500, "file_name": "book.pdf",
         "page": p, "topic_id": t, "topic_name": t}
        for t in ("a", "b") for p in range(1, 11)
    ]


def approved_item(item_id, topic="a"):
    return QuizItem(item_id=item_id, topic_id=topic, stem="Which layer adds context?",
                    options=["w", "x", "y", "z"], right_choice=1, explanation="why",
                    file_name="lecture.pdf", page=3, source="lecturer_approved")


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make(session, n=5, user="u1", approved=None, llm=fake_llm):
    return quiz_service.create_quiz(
        session, user_id=user, room_id="r1", n=n, mode="screen", llm=llm,
        chunks=chunks(), approved_items=approved, rng=random.Random(0))


def answers_for(quiz, right=True):
    return [Answer(item_id=i.item_id, choice=i.right_choice if right else (i.right_choice + 1) % 4)
            for i in quiz.items]


def test_create_quiz_returns_n_unique_generated_items(session):
    quiz = make(session, 5)
    assert len(quiz.items) == 5
    assert len({i.item_id for i in quiz.items}) == 5
    assert all(i.source == "generated" and len(i.options) == 4 for i in quiz.items)


def test_approved_items_come_first_and_the_rest_are_generated(session):
    approved = lambda room, weak: [approved_item("m5-1"), approved_item("m5-2")]
    quiz = make(session, 5, approved=approved)
    assert [i.source for i in quiz.items[:2]] == ["lecturer_approved"] * 2
    assert len(quiz.items) == 5


def test_no_approved_items_falls_back_without_error(session):
    quiz = make(session, 5, approved=lambda room, weak: [])
    assert len(quiz.items) == 5


def test_quiz_fails_cleanly_when_nothing_can_be_generated(session):
    with pytest.raises(quiz_service.QuizGenerationError):
        make(session, 3, llm=lambda prompt: "not json")


def test_submit_right_answers_raises_mastery_and_explains(session):
    quiz = make(session, 4)
    res = quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1",
                                   answers=answers_for(quiz))
    assert res.score == 1.0 and all(r.correct for r in res.items)
    assert all(r.explanation and r.page >= 1 for r in res.items)
    topics = profile_service.get_profile(session, "u1", "r1").topics
    assert topics and all(t.mastery == pytest.approx(0.65) for t in topics)


def test_submit_wrong_answers_lowers_mastery(session):
    quiz = make(session, 4)
    res = quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1",
                                   answers=answers_for(quiz, right=False))
    assert res.score == 0.0
    topics = profile_service.get_profile(session, "u1", "r1").topics
    assert all(t.mastery == pytest.approx(0.35) for t in topics)


def test_skipped_questions_count_as_wrong(session):
    quiz = make(session, 3)
    res = quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1", answers=[])
    assert res.score == 0.0


def test_cannot_submit_twice(session):
    quiz = make(session, 3)
    quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1", answers=answers_for(quiz))
    with pytest.raises(quiz_service.AlreadySubmitted):
        quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1", answers=[])


def test_cannot_submit_someone_elses_quiz(session):
    quiz = make(session, 3, user="u1")
    with pytest.raises(PermissionError):
        quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u2", answers=[])


def test_unknown_quiz(session):
    with pytest.raises(quiz_service.QuizNotFound):
        quiz_service.submit_quiz(session, quiz_id="nope", user_id="u1", answers=[])


def test_failing_a_topic_twice_aims_the_next_quiz_at_it(session):
    for _ in range(2):
        profile_service.record_event(
            session, Event(user_id="u1", room_id="r1", topic_id="a", kind="quiz", score=0.0))
    quiz = make(session, 5)
    assert sum(i.topic_id == "a" for i in quiz.items) == 3  # 60% of 5


def test_wrong_answers_are_anonymous_and_only_wrong_ones(session):
    bad = make(session, 3, user="u1")
    quiz_service.submit_quiz(session, quiz_id=bad.quiz_id, user_id="u1",
                             answers=answers_for(bad, right=False))
    good = make(session, 3, user="u2")
    quiz_service.submit_quiz(session, quiz_id=good.quiz_id, user_id="u2",
                             answers=answers_for(good))
    rows = quiz_service.wrong_answers(session, "r1")
    assert len(rows) == 3
    for row in rows:
        assert set(row.model_dump()) == {"item_id", "student_ref", "topic_id", "stem", "chosen", "right"}
        assert row.student_ref.startswith("s-") and "u1" not in row.student_ref


def test_class_profile_lists_every_student(session):
    for user in ("u1", "u2"):
        profile_service.record_event(
            session, Event(user_id=user, room_id="r1", topic_id="a", kind="quiz", score=1.0))
    rows = quiz_service.class_profile(session, "r1", names={"u1": "Sara"})
    assert [(r.user_id, r.name) for r in rows] == [("u1", "Sara"), ("u2", "u2")]
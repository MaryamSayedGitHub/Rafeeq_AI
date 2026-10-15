import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from rafeeq.quiz.contracts import Event
from rafeeq.quiz.profile import service
from rafeeq.quiz.profile.models import Base, EventRow


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def ev(kind="quiz", score=0.0, topic="t1", user="u1"):
    return Event(user_id=user, room_id="r1", topic_id=topic, kind=kind, score=score)


def test_sending_an_event_changes_mastery(session):
    service.record_event(session, ev(score=1.0))
    p = service.get_profile(session, "u1", "r1")
    assert p.topics[0].mastery == pytest.approx(0.65)


def test_every_event_is_stored(session):
    service.record_event(session, ev())
    service.record_event(session, ev(kind="question"))
    count = session.scalar(select(func.count()).select_from(EventRow))
    assert count == 2


def test_question_does_not_move_mastery(session):
    service.record_event(session, ev(kind="question", score=0.0))
    t = service.get_profile(session, "u1", "r1").topics[0]
    assert t.mastery == 0.5 and t.questions_asked == 1


def test_attendance_does_not_move_mastery(session):
    service.record_event(session, ev(kind="attendance", score=1.0))
    assert service.get_profile(session, "u1", "r1").topics[0].mastery == 0.5


def test_weak_after_two_failed_quizzes(session):
    service.record_event(session, ev(score=0.0))
    service.record_event(session, ev(score=0.0))
    assert service.get_profile(session, "u1", "r1").topics[0].weak is True


def test_weak_after_three_questions_then_cleared_by_quiz(session):
    for _ in range(3):
        service.record_event(session, ev(kind="question", score=1.0))
    assert service.get_profile(session, "u1", "r1").topics[0].weak is True
    service.record_event(session, ev(kind="quiz", score=1.0))
    assert service.get_profile(session, "u1", "r1").topics[0].weak is False


def test_handwriting_updates_the_same_profile(session):
    service.record_event(session, ev(kind="handwriting", score=1.0))
    service.record_event(session, ev(kind="quiz", score=1.0))
    assert service.get_profile(session, "u1", "r1").topics[0].attempts == 2


def test_topics_and_users_are_separate(session):
    service.record_event(session, ev(topic="t1"))
    service.record_event(session, ev(topic="t2"))
    service.record_event(session, ev(user="u2"))
    assert len(service.get_profile(session, "u1", "r1").topics) == 2
    assert len(service.get_profile(session, "u2", "r1").topics) == 1


def test_topic_names_are_used_when_given(session):
    service.record_event(session, ev())
    p = service.get_profile(session, "u1", "r1", topic_names={"t1": "Tokenization"})
    assert p.topics[0].name == "Tokenization"
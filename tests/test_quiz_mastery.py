import pytest

from rafeeq.quiz.profile.mastery import TopicState, apply_event, is_weak, update_mastery


def test_update_formula():
    assert update_mastery(0.5, 1.0) == pytest.approx(0.65)
    assert update_mastery(0.8, 0.0) == pytest.approx(0.56)


def test_score_out_of_range():
    with pytest.raises(ValueError):
        update_mastery(0.5, 1.5)


def test_quiz_changes_mastery():
    s = apply_event(TopicState(), "quiz", 0.0)
    assert s.mastery < 0.5 and s.attempts == 1


def test_question_does_not_change_mastery():
    s = apply_event(TopicState(), "question", 0.0)
    assert s.mastery == 0.5 and s.questions_asked == 1


def test_not_weak_after_one_bad_event():
    s = apply_event(TopicState(), "quiz", 0.0)
    assert not is_weak(s)


def test_weak_after_two_bad_events():
    s = TopicState()
    for _ in range(2):
        s = apply_event(s, "quiz", 0.0)
    assert is_weak(s)


def test_weak_after_three_questions_without_quiz():
    s = TopicState()
    for _ in range(3):
        s = apply_event(s, "question", 1.0)
    assert is_weak(s)


def test_quiz_resets_question_counter():
    s = TopicState()
    for _ in range(3):
        s = apply_event(s, "question", 1.0)
    s = apply_event(s, "quiz", 1.0)
    assert not is_weak(s)


def test_handwriting_moves_mastery():
    s = apply_event(TopicState(), "handwriting", 1.0)
    assert s.mastery > 0.5

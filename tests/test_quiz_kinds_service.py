"""Submit, review and the API for a quiz that mixes question types."""
import json
import random

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

import rafeeq.quiz.quiz.models  # noqa: F401  (registers the quiz tables on Base)
from rafeeq.quiz import adapters
from rafeeq.quiz.contracts import Answer
from rafeeq.quiz.profile import service as profile_service
from rafeeq.quiz.profile.db import _add_missing_columns
from rafeeq.quiz.profile.models import Base
from rafeeq.quiz.quiz import service as quiz_service
from rafeeq.quiz.quiz.models import AnswerRow

MCQ = {"stem": "What does a derivative measure in calculus?",
       "options": ["Rate of change", "Total area", "Average value", "Maximum size"],
       "right_choice": 0, "explanation": "It measures how fast a quantity changes."}
ESSAY = {"stem": "Explain why the derivative measures a rate of change.",
         "answer": "The derivative is the limit of the ratio of the change in output to the change in input. "
                   "It therefore gives the instantaneous rate of change. Geometrically it is the slope of the tangent.",
         "rubric": ["limit of a ratio of changes", "instantaneous rate", "slope of the tangent"],
         "explanation": "Definition of the derivative."}
SHORT = {"stem": "What is the derivative of a constant function equal to?", "answer": "zero | 0", "explanation": "No change."}


def llm(prompt):
    if "ESSAY" in prompt:
        return json.dumps(ESSAY)
    if "SHORT-ANSWER" in prompt:
        return json.dumps(SHORT)
    return json.dumps(MCQ)


def grader(prompt):          # ticks the first two of three rubric points
    return json.dumps({"met": [True, True, False], "feedback": "Missing the tangent."})


def chunks():
    return [{"chunk_id": f"b{p}", "text": "The derivative measures a rate of change. " * 20, "file_name": "calc.pdf",
             "page": p, "topic_id": "t1", "topic_name": "Derivatives"} for p in range(1, 9)]


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make(session, kinds, mode="screen", n=3):
    return quiz_service.create_quiz(session, user_id="u1", room_id="r1", n=n, mode=mode, llm=llm, chunks=chunks(),
                                    rng=random.Random(0), kinds=kinds)


def test_a_mixed_quiz_is_saved_and_returned_with_its_kinds(session):
    quiz = make(session, ("mcq", "essay", "short"))
    assert [i.kind for i in quiz.items] == ["mcq", "essay", "short"]
    again = quiz_service.get_quiz_items(session, quiz_id=quiz.quiz_id, user_id="u1")
    assert [i.kind for i in again] == ["mcq", "essay", "short"] and again[1].rubric == ESSAY["rubric"]


def test_a_paper_quiz_is_always_mcq(session):
    quiz = make(session, ("essay", "short"), mode="paper")
    assert {i.kind for i in quiz.items} == {"mcq"}


def test_submit_grades_each_kind_and_gives_partial_marks(session):
    quiz = make(session, ("mcq", "essay", "short"))
    mcq, essay, short = quiz.items
    result = quiz_service.submit_quiz(
        session, quiz_id=quiz.quiz_id, user_id="u1", grader_llm=grader,
        answers=[Answer(item_id=mcq.item_id, choice=mcq.right_choice),
                 Answer(item_id=essay.item_id, text="It is a limit of ratios and an instant rate."),
                 Answer(item_id=short.item_id, text="Zero")])
    by_id = {r.item_id: r for r in result.items}
    assert by_id[mcq.item_id].score == 1.0 and by_id[mcq.item_id].correct
    assert by_id[essay.item_id].score == pytest.approx(0.67) and not by_id[essay.item_id].correct   # below the 0.7 pass mark
    assert by_id[essay.item_id].feedback == "Missing the tangent."
    assert by_id[short.item_id].score == 1.0
    assert result.score == pytest.approx((1 + 0.67 + 1) / 3)
    row = session.query(AnswerRow).filter_by(item_id=essay.item_id).one()
    assert row.answer_text.startswith("It is a limit") and row.score == pytest.approx(0.67) and row.chosen == -1


def test_mastery_moves_by_the_average_mark_of_the_topic(session):
    quiz = make(session, ("essay", "short"), n=2)
    essay, short = quiz.items
    quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1", grader_llm=grader,
                             answers=[Answer(item_id=essay.item_id, text="x y z"), Answer(item_id=short.item_id, text="zero")])
    mastery = profile_service.get_profile(session, "u1", "r1").topics[0].mastery
    assert mastery == pytest.approx(0.7 * 0.5 + 0.3 * ((0.67 + 1) / 2))   # one quiz event for the topic


def test_a_typed_answer_is_not_a_silent_zero_when_there_is_no_grader(session):
    quiz = make(session, ("essay",), n=1)
    result = quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1", grader_llm=None,
                                      answers=[Answer(item_id=quiz.items[0].item_id, text="my answer")])
    assert result.items[0].needs_review and result.items[0].score == 0.0
    review = quiz_service.quiz_review(session, user_id="u1", quiz_id=quiz.quiz_id)
    assert review["questions"][0]["needs_review"] and review["questions"][0]["student_answer"] == "my answer"


def test_review_returns_the_model_answer_and_the_marks(session):
    quiz = make(session, ("mcq", "essay"), n=2)
    mcq, essay = quiz.items
    quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1", grader_llm=grader,
                             answers=[Answer(item_id=mcq.item_id, choice=mcq.right_choice),
                                      Answer(item_id=essay.item_id, text="an answer")])
    qs = {q["item_id"]: q for q in quiz_service.quiz_review(session, user_id="u1", quiz_id=quiz.quiz_id)["questions"]}
    assert qs[essay.item_id]["kind"] == "essay" and qs[essay.item_id]["model_answer"] == ESSAY["answer"]
    assert qs[essay.item_id]["score"] == pytest.approx(0.67) and qs[mcq.item_id]["correct"]


def test_the_contract_shapes_for_typed_questions(session):
    quiz = make(session, ("mcq", "essay"), n=2)
    contract = adapters.to_contract_quiz(quiz)
    assert contract["questions"][1]["kind"] == "essay" and contract["questions"][1]["choices"] == []
    assert contract["questions"][1]["correct_answer"] == ESSAY["answer"]
    answers = adapters.answers_to_choices(quiz.items, [quiz.items[0].options[0], "my essay"])
    assert answers[0].choice == 0 and answers[1].text == "my essay"
    before = profile_service.get_profile(session, "u1", "r1")
    submit = quiz_service.submit_quiz(session, quiz_id=quiz.quiz_id, user_id="u1", answers=answers, grader_llm=grader)
    after = profile_service.get_profile(session, "u1", "r1")
    out = adapters.to_contract_result(quiz.items, submit, before, after)
    assert out["total"] == 2 and len(out["item_scores"]) == 2 and out["item_scores"][1] == pytest.approx(0.67, abs=0.01)
    assert "Partly right" in out["explanations"][1] and "Model answer" in out["explanations"][1]


def test_old_mcq_rows_without_a_score_still_count(session):
    quiz = make(session, ("mcq",), n=1)
    session.add(AnswerRow(quiz_id=quiz.quiz_id, item_id=quiz.items[0].item_id, user_id="u1", room_id="r1", topic_id="t1",
                          chosen=0, correct=True))     # no score: the way a row looked before typed questions
    session.commit()
    assert quiz_service.quiz_history(session, user_id="u1", room_id="r1")[0]["score"] == 1.0
    assert quiz_service.topic_scores(session, user_id="u1", room_id="r1")[0]["average"] == 1.0


def test_the_new_columns_are_added_to_an_existing_database():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:   # the table as the first release created it
        conn.execute(text("CREATE TABLE m4_answers (id INTEGER PRIMARY KEY, quiz_id VARCHAR(64), item_id VARCHAR(80), "
                          "user_id VARCHAR(64), room_id VARCHAR(64), topic_id VARCHAR(64), chosen INTEGER, correct BOOLEAN, "
                          "created_at DATETIME)"))
        conn.execute(text("INSERT INTO m4_answers (quiz_id, item_id, user_id, room_id, topic_id, chosen, correct) "
                          "VALUES ('q', 'i', 'u', 'r', 't', 1, 1)"))
    _add_missing_columns(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("m4_answers")}
    assert {"answer_text", "score", "feedback", "needs_review"} <= cols
    with Session(engine) as s:
        row = s.query(AnswerRow).one()
        assert row.score is None and row.answer_text == "" and row.needs_review in (False, 0)
    _add_missing_columns(engine)   # running it again changes nothing

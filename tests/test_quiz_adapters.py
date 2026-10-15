"""The translation between M4's shapes and the team contract (no database, no web)."""
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker

from rafeeq.quiz import adapters
from rafeeq.quiz.contracts import (
    Profile,
    QuizCreateResponse,
    QuizItem,
    SubmitResponse,
    TopicProfile,
    ItemResult,
)

ROOT = Path(__file__).resolve().parent.parent
SPEC = yaml.safe_load((ROOT / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))


def check(schema: str, data: dict) -> None:
    v = Draft202012Validator({"$ref": f"#/components/schemas/{schema}", "components": SPEC["components"]},
                             format_checker=FormatChecker())
    errors = [e.message for e in v.iter_errors(data)]
    assert not errors, errors


def item(i, topic="t1", right=1, ar=False):
    return QuizItem(
        item_id=f"q-{i}", topic_id=topic,
        stem="أي عضية تنتج الطاقة في الخلية؟" if ar else f"What does step {i} do?",
        options=["أ", "ب", "ج", "د"] if ar else ["A", "B", "C", "D"],
        right_choice=right, explanation="لأنها كذلك" if ar else "Because.",
        file_name="book.pdf", page=3,
    )


def topic(tid, mastery, attempts=2, weak=False, name=None):
    return TopicProfile(topic_id=tid, name=name or tid, mastery=mastery, attempts=attempts,
                        questions_asked=0, weak=weak)


def test_quiz_matches_the_contract():
    resp = QuizCreateResponse(quiz_id="q", items=[item(1), item(2, topic="t2")])
    out = adapters.to_contract_quiz(resp)
    check("Quiz", out)
    assert out["topic_id"] == "t1"                      # mixed quiz: topic of the first question
    assert out["questions"][0]["correct_answer"] == "B"  # right_choice 1 -> the option text
    assert adapters.to_contract_quiz(resp, "t9")["topic_id"] == "t9"


def test_answers_become_choice_indexes():
    items = [item(1), item(2)]
    got = adapters.answers_to_choices(items, ["B", "  D "])
    assert [(a.item_id, a.choice) for a in got] == [("q-1", 1), ("q-2", 3)]


def test_missing_or_unknown_answers_are_skipped():
    items = [item(1), item(2), item(3)]
    got = adapters.answers_to_choices(items, ["not an option"])
    assert [a.choice for a in got] == [-1, -1, -1]


def test_result_matches_the_contract_and_counts_correct_answers():
    items = [item(1), item(2, topic="t2")]
    submit = SubmitResponse(score=0.5, items=[
        ItemResult(item_id="q-1", correct=True, right_choice=1, explanation="Because.", file_name="b", page=1),
        ItemResult(item_id="q-2", correct=False, right_choice=1, explanation="Because.", file_name="b", page=1),
    ])
    before = Profile(topics=[topic("t1", 0.5)])
    after = Profile(topics=[topic("t1", 0.59), topic("t2", 0.35)])
    out = adapters.to_contract_result(items, submit, before, after)
    check("QuizResult", out)
    assert (out["score"], out["total"]) == (1, 2)
    assert out["explanations"][0].startswith("Correct.")
    assert out["explanations"][1].startswith("The correct answer is B.")
    assert out["mastery_change"] == [{"topic_id": "t1", "change": 0.09},
                                     {"topic_id": "t2", "change": -0.15}]  # t2 started at 0.5


def test_arabic_quiz_gets_an_arabic_verdict():
    items = [item(1, ar=True)]
    submit = SubmitResponse(score=1.0, items=[
        ItemResult(item_id="q-1", correct=True, right_choice=1, explanation="x", file_name="b", page=1)])
    out = adapters.to_contract_result(items, submit, Profile(topics=[]), Profile(topics=[]))
    assert out["explanations"][0].startswith("إجابة صحيحة.")


def test_summary_matches_the_contract():
    profile = Profile(topics=[topic("a", 0.35, weak=True), topic("b", 0.55, weak=True), topic("c", 0.9)])
    out = adapters.to_student_summary(profile)
    check("StudentSummary", out)
    assert [t["topic_id"] for t in out["weak_topics"]] == ["a", "b"]   # weakest first


@pytest.mark.parametrize("mastery,level", [(0.2, "beginner"), (0.5, "intermediate"), (0.85, "advanced")])
def test_level_follows_average_mastery(mastery, level):
    assert adapters.level_of(Profile(topics=[topic("a", mastery)])) == level


def test_a_new_student_is_a_beginner_with_no_weak_topics():
    out = adapters.to_student_summary(Profile(topics=[]))
    check("StudentSummary", out)
    assert out == {"weak_topics": [], "level": "beginner"}


def test_next_step_prefers_the_weakest_weak_topic():
    profile = Profile(topics=[topic("a", 0.55, weak=True), topic("b", 0.3, weak=True), topic("c", 0.1)])
    chosen = adapters.pick_next_topic(profile)
    assert chosen.topic_id == "b"
    out = adapters.next_step(chosen)
    check("NextStep", out)


def test_next_step_with_no_weak_topic_takes_the_lowest():
    profile = Profile(topics=[topic("a", 0.8), topic("b", 0.7)])
    assert adapters.pick_next_topic(profile).topic_id == "b"
    assert adapters.pick_next_topic(Profile(topics=[])) is None
    check("NextStep", adapters.first_topic_step("t1", "Intro"))


def test_only_question_events_are_recorded():
    base = {"user_id": "u", "room_id": "r", "topic_id": "t", "text": "why?", "time": "2026-10-09T10:30:00Z"}
    q = adapters.QuestionEvent(kind="question", **base)
    ev = adapters.question_event_to_event(q)
    assert (ev.kind, ev.score, ev.meta["text"]) == ("question", 0.0, "why?")
    assert adapters.question_event_to_event(adapters.QuestionEvent(kind="understood", **base)) is None

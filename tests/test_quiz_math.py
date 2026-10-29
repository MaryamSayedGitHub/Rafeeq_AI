"""Maths answers: sympy compares the student's expression or equation with the model answer by value."""
import json

import pytest

from rafeeq.quiz.quiz import grading
from rafeeq.quiz.quiz.grading import grade_typed_answer
from rafeeq.quiz.quiz.validator import check_item
from rafeeq.quiz.contracts import QuizItem


def item(kind="math", **over):
    base = dict(item_id="i1", topic_id="t1", stem="A question that is long enough?", explanation="because",
                file_name="b.pdf", page=1, kind=kind, answer="x")
    base.update(over)
    return QuizItem(**base)

sympy = pytest.importorskip("sympy")


@pytest.mark.parametrize("student,expected", [
    ("2x+2", "2(x+1)"), ("x^2 - 1", "(x-1)(x+1)"), ("2*x + 3", "2x + 3"), ("1/2", "0.5"), ("3.14", "pi"),
    ("x = 3", "3"), ("3", "x = 3"), ("x=3", "x = 3"), ("2x+1=7", "x=3"), ("y = 2x + 1", "2x + 1 = y"),
    ("sqrt(4)", "2"), ("√(9)", "3"), ("5x²", "5*x^2"), ("\\frac{1}{4}", "0.25"), ("x = -3 or x = 2", "x=2, x=-3"),
    ("1,000", "1000"), ("50%", "1/2"),
])
def test_equivalent_maths_answers_are_accepted(student, expected):
    assert grading.math_equal(student, expected)


@pytest.mark.parametrize("student,expected", [
    ("2x+3", "2x+2"), ("x = 4", "x = 3"), ("1/3", "0.5"), ("x = 2", "x = 2 or x = 3"), ("y = 2x", "y = 2x + 1"),
    ("x^2", "x^3"),
])
def test_different_maths_answers_are_rejected(student, expected):
    assert not grading.math_equal(student, expected)


@pytest.mark.parametrize("text", [
    "__import__('os').system('ls')", "open('/etc/passwd')", "x.free_symbols", "9**9**9", "2**99999", "eval('1')",
    "lambda: 1", "x" * 200, "[1,2]", "a; b",
])
def test_unsafe_or_unreadable_text_is_never_executed(text):
    with pytest.raises(grading.MathParseError):
        grading.parse_math(text)


def test_maths_item_grading_and_fallback_wording():
    it = item("math", answer="x = 3")
    assert grade_typed_answer(it, "2x+1=7", None).score == 1.0
    wrong = grade_typed_answer(it, "x = 4", None)
    assert wrong.score == 0.0 and "not equivalent" in wrong.feedback
    # "x equals three" is not maths syntax: the LLM compares the meaning, and the student gets a format hint if it says no
    said = grade_typed_answer(it, "x equals three", lambda p: json.dumps({"same": True, "feedback": ""}))
    assert said.score == 1.0
    no_llm = grade_typed_answer(it, "x equals three", None)
    assert no_llm.needs_review and "x^2" in no_llm.feedback


def test_a_generated_maths_item_whose_answer_cannot_be_read_is_rejected():
    assert any("computer can read" in p for p in check_item(item("math", answer="the square root of two")))
    assert check_item(item("math", answer="x = 3 or x = -2")) == []

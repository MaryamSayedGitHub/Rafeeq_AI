"""The evaluation tools for the extra question types: the reference cases are sound and the runners count right.
(The numbers a real model gets come from running `python -m rafeeq.quiz.eval.eval_kinds`, not from these tests.)"""
import csv
import json
import re

import pytest

from rafeeq.quiz.eval import eval_kinds as ev
from rafeeq.quiz.quiz import grading
from rafeeq.quiz.quiz.validator import check_item

GRADER = ev.load_cases("grader_cases")
INJECTION = ev.load_cases("injection_cases")
MATH = ev.load_cases("math_cases")


# ---------------------------------------------------------------- the reference cases
def test_every_reference_item_is_a_valid_question():
    for cases in (GRADER, INJECTION):
        for item_id, spec in cases["items"].items():
            assert check_item(ev.make_item(item_id, spec)) == [], item_id


def test_every_grader_case_points_at_an_item_and_has_one_human_label():
    for c in GRADER["answers"]:
        item = GRADER["items"][c["item"]]
        assert ("human_met" in c) != ("human_same" in c)
        if "human_met" in c:
            assert item["kind"] in ("essay", "problem") and len(c["human_met"]) == len(item["rubric"])
        else:
            assert item["kind"] == "short"
    kinds = {GRADER["items"][c["item"]]["kind"] for c in GRADER["answers"]}
    assert kinds == {"essay", "problem", "short"}


def test_the_set_has_good_and_bad_answers_and_some_arabic():
    verdicts = [sum(c["human_met"]) / len(c["human_met"]) >= 0.7 if "human_met" in c else c["human_same"] for c in GRADER["answers"]]
    assert 0.3 < sum(verdicts) / len(verdicts) < 0.7
    assert any(any("؀" <= ch <= "ۿ" for ch in c["answer"]) for c in GRADER["answers"])


def test_attacks_have_no_real_content_and_the_math_set_is_not_empty():
    assert len(INJECTION["attacks"]) >= 10
    assert len(MATH["same"]) >= 30 and len(MATH["different"]) >= 20 and len(MATH["unreadable"]) >= 15


# ---------------------------------------------------------------- the grader runner
def oracle(cases):
    """A grader that always agrees with the person: replies with the human labels of the case being graded."""
    by_answer = {c["answer"]: c for c in cases["answers"]}

    def llm(prompt):
        for answer, c in by_answer.items():
            if f"<student_answer>\n{answer[:4000]}\n</student_answer>" in prompt or f"<student_answer>\n{answer[:500]}\n</student_answer>" in prompt:
                if "human_met" in c:
                    return json.dumps({"met": c["human_met"], "feedback": "ok"})
                return json.dumps({"same": c["human_same"], "feedback": "ok"})
        raise AssertionError("unknown prompt")
    return llm


def test_a_grader_that_agrees_with_the_person_scores_perfectly():
    r = ev.run_grader(GRADER, oracle(GRADER))
    assert r["graded"] == r["cases"] == 40 and r["needs_review"] == 0
    assert r["pass_agreement"] == 1.0 and r["kappa"] == 1.0
    assert r["rubric_mean_abs_error"] == 0.0 and r["rubric_points_agreement"] == 1.0
    assert r["short_accuracy"] == 1.0 and r["disagreements"] == []
    assert r["short_without_model"] > 0           # exact matches never reach the model


def test_a_generous_grader_is_caught():
    def generous(prompt):
        points = len(re.findall(r"^\d+\. ", prompt, re.M))     # one tick per numbered rubric point
        return json.dumps({"met": [True] * points, "same": True, "feedback": "great"})
    r = ev.run_grader(GRADER, generous)
    assert r["pass_agreement"] < 0.9 and r["disagreements"]


def test_a_model_that_is_down_is_counted_as_needs_review_not_as_wrong():
    def down(prompt):
        raise RuntimeError("503")
    r = ev.run_grader(GRADER, down)
    assert r["needs_review"] > 0 and r["graded"] < r["cases"]


def test_kappa_is_one_for_equal_verdicts_and_zero_for_chance():
    assert ev.kappa([True, False, True], [True, False, True]) == 1.0
    assert ev.kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert ev.kappa([], []) == 0.0


# ---------------------------------------------------------------- the injection runner
def test_a_grader_that_ticks_nothing_cannot_be_fooled():
    def strict(prompt):
        return json.dumps({"met": [False] * len(re.findall(r"^\d+\. ", prompt, re.M)), "same": False, "feedback": "no"})
    r = ev.run_injection(INJECTION, strict)
    assert r["judged"] == r["attacks"] and r["fooled"] == [] and r["some_marks"] == []


def test_a_gullible_grader_is_reported():
    def gullible(prompt):
        return json.dumps({"met": [True] * len(re.findall(r"^\d+\. ", prompt, re.M)), "same": True, "feedback": "ok"})
    r = ev.run_injection(INJECTION, gullible)
    assert len(r["fooled"]) >= 5


# ---------------------------------------------------------------- the maths runner (the real checker needs sympy)
def test_math_runner_counts_wrong_accepts_and_wrong_rejects():
    cases = {"same": [["a", "a"], ["b", "b"]], "different": [["a", "b"], ["c", "d"]], "unreadable": ["x equals 3"], "known_gaps": []}
    def refuse(text):
        raise grading.MathParseError("no")
    r = ev.run_math(cases, equal=lambda s, e: True, parse=refuse)           # says "same" to everything
    assert len(r["wrong_accepts"]) == 2 and r["wrong_rejects"] == [] and r["accuracy"] == 0.5
    r = ev.run_math(cases, equal=lambda s, e: False, parse=refuse)
    assert len(r["wrong_rejects"]) == 2 and r["wrong_accepts"] == []


def test_the_real_maths_checker_meets_the_bar():
    pytest.importorskip("sympy")
    r = ev.run_math(MATH)
    assert r["unsafe_failures"] == [], r["unsafe_failures"]      # nothing unsafe is ever parsed
    assert r["wrong_accepts"] == [], r["wrong_accepts"]          # a wrong answer must never get the marks
    assert r["accuracy"] >= 0.9, (r["wrong_rejects"], r["unreadable_but_should_be_judged"])
    assert r["slowest_seconds"] < 2.0


# ---------------------------------------------------------------- generation by type + the review sheet
CHUNK = {"chunk_id": "c1", "text": "The derivative measures a rate of change. " * 20, "file_name": "calc.pdf", "page": 1,
         "topic_id": "t1", "topic_name": "Derivatives"}


def fake_generator(prompt):
    replies = {
        "ESSAY": {"stem": "Explain why the derivative measures a rate of change.",
                  "answer": "The derivative is the limit of the ratio of the change in output to the change in input, so it is the instantaneous rate of change.",
                  "rubric": ["limit of a ratio", "instantaneous rate", "slope of the tangent"], "explanation": "Definition."},
        "SHORT-ANSWER": {"stem": "What is the derivative of a constant function equal to?", "answer": "zero | 0", "explanation": "No change."},
    }
    for marker, reply in replies.items():
        if marker in prompt:
            return json.dumps(reply)
    return "not json"


def test_generation_is_measured_per_type_and_failures_are_counted():
    chunks = [{**CHUNK, "chunk_id": f"c{p}", "page": p} for p in range(1, 7)]
    r = ev.run_generation(fake_generator, chunks, ["essay", "short", "math"], n=4, seed=1)
    assert r["kinds"]["essay"]["produced"] == 1.0 and r["kinds"]["essay"]["first_try"] == 1.0
    assert r["kinds"]["short"]["produced"] == 1.0
    assert r["kinds"]["math"]["produced"] == 0.0           # the fake model cannot write maths
    assert {s["kind"] for s in r["samples"]} == {"essay", "short"}


def test_the_review_sheet_shows_the_right_option_of_choice_questions(tmp_path):
    samples = [{"kind": "mcq", "item_id": "m-1", "page": 1,
                "item": {"stem": "Q?", "options": ["a", "b", "c", "d"], "right_choice": 2, "answer": "", "rubric": []}}]
    path = str(tmp_path / "review.csv")
    ev.write_review_csv(samples, path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        assert list(csv.DictReader(f))[0]["answer"] == "c"


def test_the_review_sheet_round_trips(tmp_path):
    samples = [{"kind": "essay", "item_id": "e-1", "page": 1, "item": {"stem": "Q?", "answer": "A", "rubric": ["r1", "r2"], "options": []}},
               {"kind": "short", "item_id": "s-1", "page": 2, "item": {"stem": "Q2?", "answer": "zero", "options": []}},
               {"kind": "short", "item_id": "s-2", "page": 3, "item": {"stem": "Q3?", "answer": "one", "options": []}}]
    path = str(tmp_path / "review.csv")
    ev.write_review_csv(samples, path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["item_id"] for r in rows] == ["e-1", "s-1", "s-2"] and rows[0]["rubric"] == "r1 | r2"
    rows[0]["human_ok"], rows[1]["human_ok"] = "y", "n"                  # a person fills two rows, leaves the third
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = ev.summarize_review(path)
    assert s["essay"] == {"reviewed": 1, "good": 1, "good_rate": 1.0}
    assert s["short"]["reviewed"] == 1 and s["short"]["good"] == 0


def test_bad_ids_are_the_only_ones_counted_as_bad_and_the_text_file_marks_the_key(tmp_path):
    samples = [{"kind": "mcq", "item_id": "m-1", "page": 1, "item": {"kind": "mcq", "stem": "Q?", "options": ["a", "b"], "right_choice": 1, "answer": "", "rubric": []}},
               {"kind": "math", "item_id": "x-1", "page": 2, "item": {"kind": "math", "stem": "2+2?", "options": [], "answer": "4", "rubric": []}},
               {"kind": "math", "item_id": "x-2", "page": 3, "item": {"kind": "math", "stem": "3+3?", "options": [], "answer": "6", "rubric": []}}]
    s = ev.summarize_bad(samples, ["x-2"])
    assert s["mcq"]["good"] == 1 and s["math"] == {"reviewed": 2, "good": 1, "good_rate": 0.5}
    with pytest.raises(SystemExit):
        ev.summarize_bad(samples, ["nope"])
    path = str(tmp_path / "r.txt")
    ev.write_reading_text(samples, path)
    text = open(path, encoding="utf-8").read()
    assert ">> b" in text and "ANSWER KEY:\n4" in text

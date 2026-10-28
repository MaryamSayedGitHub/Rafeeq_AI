"""Semantic checks for a quiz item. QuizItem itself already enforces the shape of each kind
(mcq: 4 options and right_choice in 0..3; true_false: 2 options; typed kinds: no options and a model answer)."""
from __future__ import annotations

import re

from rafeeq.quiz.contracts import QuizItem

# A question must make sense on its own, so it cannot point at the page it came from.
CONTEXT_PHRASES = re.compile(
    r"\b(the|this|that) (text|passage|excerpt|page|section|figure)\b"
    r"|\bprovided\b|\baccording to\b"
    # "as referenced in", "described in relation to", "described using" ...
    r"|\b(referenced|described|mentioned|discussed|stated|outlined|presented|shown)"
    r"\s+(in|by|above|earlier|previously|here|using|with)\b"
    # "the described architecture", "the mentioned model"
    r"|\b(the|this|these|those) (described|mentioned|discussed|aforementioned|above|given)\b"
    # "in the context" / "in the given context", but not "in the context of NLP"
    r"|\bin the (given )?context\b(?! of)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _check_typed(item: QuizItem) -> list[str]:
    """true_false and the typed kinds."""
    problems: list[str] = []
    answer = item.answer.strip()
    if item.kind == "short":
        if len(answer.split()) > 6 or len(answer) > 80:
            problems.append("a short answer is at most 6 words")
        if _norm(answer) and _norm(answer) in _norm(item.stem):
            problems.append("the question already contains its own answer")
    elif item.kind == "math":
        from rafeeq.quiz.quiz.grading import model_answer_parses

        if not model_answer_parses(answer):
            problems.append("the model answer is not a plain expression or equation a computer can read "
                            "(write x^2 + 1, sqrt(2), 1/2, x = 3; no words or LaTeX)")
    elif item.kind in ("essay", "problem"):
        if not 2 <= len(item.rubric) <= 6:
            problems.append("the rubric needs 3 to 5 points")
        if len(set(_norm(r) for r in item.rubric)) != len(item.rubric):
            problems.append("two rubric points are the same")
        if len(answer) < 40:
            problems.append("the model answer is too short")
    return problems


def check_item(item: QuizItem) -> list[str]:
    """Return a list of problems. Empty list means the item is valid."""
    problems: list[str] = []

    if len(item.stem.strip()) < 10:
        problems.append("the question is too short")
    if CONTEXT_PHRASES.search(item.stem):
        problems.append(
            "the question refers to a text, passage, context or something "
            "'described', 'mentioned' or 'referenced' elsewhere; it must make "
            "sense on its own, so name the concept directly"
        )

    if item.kind == "mcq":
        options = [_norm(o) for o in item.options]
        if any(not o for o in options):
            problems.append("an option is empty")
        if len(set(options)) != len(options):
            problems.append("two options are duplicates")

        # an option that repeats the question gives the answer away
        stem = _norm(item.stem)
        if any(len(o) > 25 and (o in stem or stem in o) for o in options):
            problems.append("an option repeats the question; write a real question and four different answers")
    else:
        problems += _check_typed(item)

    if not item.explanation.strip():
        problems.append("the explanation is empty")
    if not item.file_name.strip() or item.page < 1:
        problems.append("the item has no source page")

    return problems


def is_valid(item: QuizItem) -> bool:
    return not check_item(item)
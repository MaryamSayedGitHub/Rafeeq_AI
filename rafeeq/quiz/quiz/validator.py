"""Semantic checks for a quiz item. QuizItem itself already enforces
exactly 4 options and right_choice in 0..3."""
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

    options = [_norm(o) for o in item.options]
    if any(not o for o in options):
        problems.append("an option is empty")
    if len(set(options)) != len(options):
        problems.append("two options are duplicates")

    # an option that repeats the question gives the answer away
    stem = _norm(item.stem)
    if any(len(o) > 25 and (o in stem or stem in o) for o in options):
        problems.append("an option repeats the question; write a real question and four different answers")

    if not item.explanation.strip():
        problems.append("the explanation is empty")
    if not item.file_name.strip() or item.page < 1:
        problems.append("the item has no source page")

    return problems


def is_valid(item: QuizItem) -> bool:
    return not check_item(item)
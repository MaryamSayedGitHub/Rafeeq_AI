"""How hard is a question: easy, medium or hard.

One LLM call rates a whole batch of questions. A question the LLM does not rate
(or rates with a word we do not know) gets a level from its type, so every question
always has a level.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from rafeeq.quiz.contracts import QuizItem
from rafeeq.teacher.question_bank.jsonreply import first_json_object

log = logging.getLogger(__name__)

LLM = Callable[[str], str]

LEVELS = ("easy", "medium", "hard")
BATCH = 10

# used only when the LLM gives no usable answer
BY_KIND = {"true_false": "easy", "short": "easy", "mcq": "medium", "math": "medium", "problem": "hard", "essay": "hard"}


def default_level(item: QuizItem) -> str:
    return BY_KIND.get(item.kind, "medium")


def _prompt(items: list[QuizItem]) -> str:
    lines = "\n".join(f"{i}. [{it.kind}] {it.stem}" for i, it in enumerate(items, 1))
    return (
        "Rate how hard each question is for a student who has studied the topic.\n"
        "- easy: remember a definition or a single fact.\n"
        "- medium: apply an idea, or connect two ideas.\n"
        "- hard: several steps of reasoning, analysis, or a longer calculation.\n\n"
        f"Questions:\n{lines}\n\n"
        f'Return JSON only, one level per question in the same order: {{"levels": ["easy", "medium", ...]}} '
        f"({len(items)} levels)."
    )


def _rate_batch(items: list[QuizItem], llm: Optional[LLM]) -> list[str]:
    fallback = [default_level(it) for it in items]
    if llm is None:
        return fallback
    try:
        data = first_json_object(llm(_prompt(items)))
    except Exception:
        log.warning("difficulty: LLM call failed, rating by question type", exc_info=True)
        return fallback
    levels = data.get("levels") if data else None
    if not isinstance(levels, list):
        return fallback
    out = []
    for i, default in enumerate(fallback):
        word = str(levels[i]).strip().lower() if i < len(levels) else ""
        out.append(word if word in LEVELS else default)
    return out


def classify_difficulties(items: list[QuizItem], llm: Optional[LLM]) -> dict[str, str]:
    """{item_id: "easy" | "medium" | "hard"} for every item."""
    result: dict[str, str] = {}
    for start in range(0, len(items), BATCH):
        batch = items[start : start + BATCH]
        for item, level in zip(batch, _rate_batch(batch, llm)):
            result[item.item_id] = level
    return result
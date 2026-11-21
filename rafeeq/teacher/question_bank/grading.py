
"""Grading guide for a question bank item.

Rubric is used only for essay and problem questions.
Choice questions are graded by code.
Short and math questions use the model answer without a rubric.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from pydantic import ValidationError

from rafeeq.quiz.contracts import QuizItem
from rafeeq.teacher.question_bank.contracts import Grading

log = logging.getLogger(__name__)

LLM = Callable[[str], str]

CHOICE_KINDS = ("mcq", "true_false")
RUBRIC_KINDS = ("essay", "problem")
MAX_LEN = 600

_ARABIC = re.compile("[؀-ۿ]")


def _is_arabic(item: QuizItem) -> bool:
    return bool(_ARABIC.search(item.stem) or _ARABIC.search(item.answer))


def _uses_rubric(item: QuizItem) -> bool:
    return item.kind in RUBRIC_KINDS


def _parse_json(raw: str) -> Optional[dict]:
    """Take the first JSON object, allowing code fences or extra text."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, dict) else None


# ------------------------------------------------------------------
# MCQ and true/false: grading by code
# ------------------------------------------------------------------

def choice_grading(item: QuizItem) -> Grading:
    """The correct option is right; all other options are wrong."""
    right = item.options[item.right_choice] if item.options else ""

    if _is_arabic(item):
        return Grading(
            correct=f'يختار الطالب "{right}".',
            partial="لا توجد درجة جزئية: الإجابة إما صحيحة أو خاطئة.",
            wrong="يختار الطالب أي اختيار آخر.",
        )

    return Grading(
        correct=f'The student picks "{right}".',
        partial="No partial credit: the answer is either right or wrong.",
        wrong="The student picks any other option.",
    )


# ------------------------------------------------------------------
# Fallback when the LLM cannot help
# ------------------------------------------------------------------

def fallback_grading(item: QuizItem) -> Grading:
    """Build a basic grading guide from the model answer and, when relevant, rubric."""
    ar = _is_arabic(item)

    # Rubric is relevant only to essay and problem questions.
    if _uses_rubric(item) and item.rubric:
        points = "; ".join(item.rubric)

        if ar:
            return Grading(
                correct=f"تتضمن الإجابة كل المعايير التالية: {points}",
                partial=f"تتضمن الإجابة بعض المعايير التالية وليس كلها: {points}",
                wrong="لا تتضمن الإجابة أيًا من المعايير المطلوبة، أو تقدم إجابة مختلفة.",
            )

        return Grading(
            correct=f"Covers all of these criteria: {points}",
            partial=f"Covers some, but not all, of these criteria: {points}",
            wrong="Does not cover the required criteria, or gives a different answer.",
        )

    answer = item.answer.strip()

    if ar:
        return Grading(
            correct=f'الإجابة تطابق "{answer}" أو تقدم صيغة مكافئة صحيحة.',
            partial="الفكرة صحيحة لكن الإجابة ناقصة، أو تحتوي على خطأ بسيط.",
            wrong="إجابة مختلفة أو خارج الموضوع.",
        )

    return Grading(
        correct=f'The answer matches "{answer}" or gives an equivalent correct form.',
        partial="The idea is right but incomplete, or contains a minor error.",
        wrong="A different answer, or an answer that is off topic.",
    )


# ------------------------------------------------------------------
# LLM prompt
# ------------------------------------------------------------------

def build_prompt(item: QuizItem) -> str:
    language = (
        "Write the three descriptions in Arabic."
        if _is_arabic(item)
        else "Write the three descriptions in English."
    )

    rubric_section = ""

    if _uses_rubric(item) and item.rubric:
        criteria = []

        for index, criterion in enumerate(item.rubric):
            weight_text = ""

            if item.rubric_weights and index < len(item.rubric_weights):
                weight_text = f" (weight: {item.rubric_weights[index]}%)"

            criteria.append(f"- {criterion}{weight_text}")

        rubric_section = (
            "Evaluation criteria:\n"
            + "\n".join(criteria)
            + "\n"
            "Use these criteria when describing correct and partially correct answers.\n"
        )

    return (
        "You write a grading guide for a teacher.\n"
        f"Question type: {item.kind}\n"
        f"Question: {item.stem}\n"
        f"Model answer: {item.answer}\n"
        f"{rubric_section}\n"
        "Describe three kinds of student answer:\n"
        '- "correct": what a fully correct answer says or contains.\n'
        '- "partial": what makes an answer partly correct and what is missing.\n'
        '- "wrong": what makes an answer wrong.\n'
        "Each description must be one or two sentences and specific to THIS question.\n"
        "Do not use generic descriptions such as 'good answer' or 'bad answer'.\n"
        f"{language}\n"
        'Return JSON only: {"correct": str, "partial": str, "wrong": str}'
    )


def _from_llm(item: QuizItem, llm: LLM) -> Optional[Grading]:
    data = _parse_json(llm(build_prompt(item)))

    if data is None:
        return None

    try:
        parts = {
            key: str(data.get(key, "")).strip()[:MAX_LEN]
            for key in ("correct", "partial", "wrong")
        }
        return Grading(**parts)

    except ValidationError:
        return None


# ------------------------------------------------------------------
# Public function
# ------------------------------------------------------------------

def generate_grading(
    item: QuizItem,
    llm: Optional[LLM],
) -> Grading:
    """Always return a grading guide for the question."""

    if item.kind in CHOICE_KINDS:
        return choice_grading(item)

    if llm is not None:
        try:
            guide = _from_llm(item, llm)
        except Exception:
            log.warning(
                "grading: LLM call failed for item=%s, using fallback",
                item.item_id,
                exc_info=True,
            )
            guide = None

        if guide is not None:
            return guide

    return fallback_grading(item)

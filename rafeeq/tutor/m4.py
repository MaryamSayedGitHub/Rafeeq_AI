"""The student pages' connection to M4 (quiz and profile): the M4 routes of contracts/openapi.yaml, called exactly as agreed.

  POST /quiz                    {room_id, topic_id, count}  -> {quiz_id, topic_id, questions: [{text, choices, correct_answer, explanation}]}
  POST /quiz/{quiz_id}/submit   {answers: [str]}            -> {score, total, explanations, mastery_change: [{topic_id, change}]}
  GET  /next-step?room_id                                   -> {topic_id, suggestion}
  GET  /profile/summary?room_id                             -> {weak_topics: [{topic_id, score}], level}
  (questions asked / solutions understood go to POST /events, see events.py)

M4_BASE_URL  default http://127.0.0.1:8000 (M4's routes live in this same app).
Until M4's routes exist every call falls back (sample data, or scoring here) and says so ("sample" / "local" instead of "m4").
"""
from __future__ import annotations

import logging
import os

import requests

from rafeeq.samples import load_sample

from . import quiz as quiz_mod

log = logging.getLogger(__name__)
TIMEOUT = 10


def _base() -> str:
    return os.getenv("M4_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _fallback_on() -> bool:
    return os.getenv("TUTOR_SAMPLE_FALLBACK", "1") != "0"


def _headers(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def make_quiz(room_id: str, topic_id: str, count: int = 5, token: str | None = None) -> tuple[dict | None, str]:
    """-> (quiz in the contract's shape, "m4" | "sample" | "none")"""
    try:
        r = requests.post(f"{_base()}/quiz", json={"room_id": room_id, "topic_id": topic_id, "count": count}, headers=_headers(token), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if quiz_mod.normalize(data.get("questions", []), count):
            return data, "m4"
    except Exception as e:                                         # noqa: BLE001
        log.warning("M4 /quiz not reachable: %s", type(e).__name__)
    if _fallback_on():
        return load_sample("make_quiz"), "sample"
    return None, "none"


def submit_quiz(quiz: dict, answers: list[str], token: str | None = None) -> tuple[dict, str]:
    """-> (QuizResult, "m4" | "local"). `quiz` is the quiz the student saw; without M4 it is scored here."""
    try:
        r = requests.post(f"{_base()}/quiz/{quiz['quiz_id']}/submit", json={"answers": answers}, headers=_headers(token), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if {"score", "total", "explanations", "mastery_change"} <= set(data):
            return data, "m4"
    except Exception as e:                                         # noqa: BLE001
        log.warning("M4 quiz submit not reachable: %s", type(e).__name__)
    qs = quiz["questions"]
    right = [i < len(answers) and answers[i] == q["correct_answer"] for i, q in enumerate(qs)]
    return {"score": sum(right), "total": len(qs), "mastery_change": [],
            "explanations": [(("✓ " if ok else "✗ ") + (q.get("explanation") or q["correct_answer"])) for ok, q in zip(right, qs)]}, "local"


def next_step(room_id: str, token: str | None = None) -> tuple[dict, str]:
    try:
        r = requests.get(f"{_base()}/next-step", params={"room_id": room_id}, headers=_headers(token), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data.get("suggestion"):
            return data, "m4"
    except Exception as e:                                         # noqa: BLE001
        log.warning("M4 /next-step not reachable: %s", type(e).__name__)
    return load_sample("next_step"), "sample"


def profile_summary(room_id: str, token: str | None = None) -> tuple[dict, str]:
    try:
        r = requests.get(f"{_base()}/profile/summary", params={"room_id": room_id}, headers=_headers(token), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if "weak_topics" in data:
            return data, "m4"
    except Exception as e:                                         # noqa: BLE001
        log.warning("M4 /profile/summary not reachable: %s", type(e).__name__)
    return load_sample("student_summary"), "sample"

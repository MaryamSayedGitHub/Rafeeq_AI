"""The only place the Copilot reaches other members' work.

Each member connects their real function once at startup, for example in their module or
in the app's main file:

    from rafeeq.copilot import clients
    clients.connect("tutor_answer", my_answer_function)   # M3

Until a function is connected, the Copilot uses the contract sample and marks the result
as a sample, so the UI can label it. Set RAFEEQ_ALLOW_SAMPLES=0 to turn samples off; the
Copilot then says the capability is not available instead of showing sample content.

Expected signatures (all return the contract's JSON shape as a dict):
  tutor_answer(room_id, question)            -> TutorAnswer          (M3)
  check_work(room_id, file_id, text)         -> HandwritingCheck     (M3)
  make_quiz(room_id, topic_id | None)        -> Quiz                 (M4, optional: the chat now uses next_step)
  next_step(room_id, user_id | None)         -> NextStep | None      (M4, rafeeq/quiz/copilot_hooks.py)
  read_image(image_bytes)                    -> {type, language, text, confidence}  (M3 OCR)
  overview(room_id, question)                -> TutorAnswer | None   (M2: "explain the last lesson")
  topic_of(room_id, citations)               -> topic_id | None      (M2: which topic a cited answer is about)
  find_verified(room_id, question)           -> {answer, ...} | None (teacher-approved answers, not in contract yet)
  escalate(room_id, question, user_id)       -> {question_id}        (teacher inbox, not in contract yet)
  record_event(event_dict)                   -> None                 (M4 /events, QuestionEvent)
"""
from __future__ import annotations

import os
from typing import Any, Callable

from rafeeq.samples import load_sample

_CONNECTED: dict[str, Callable[..., Any]] = {}

_SAMPLES = {
    "tutor_answer": "tutor_answer",
    "check_work": "handwriting_check",
    "make_quiz": "make_quiz",
    "next_step": "next_step",
}


def connect(name: str, fn: Callable[..., Any]) -> None:
    _CONNECTED[name] = fn


def disconnect(name: str) -> None:
    _CONNECTED.pop(name, None)


def is_connected(name: str) -> bool:
    return name in _CONNECTED


def samples_allowed() -> bool:
    return os.environ.get("RAFEEQ_ALLOW_SAMPLES", "1") != "0"


def call(name: str, *args: Any) -> tuple[Any, str]:
    """Returns (result, source) where source is "live", "sample" or "unavailable"."""
    fn = _CONNECTED.get(name)
    if fn is not None:
        return fn(*args), "live"
    sample = _SAMPLES.get(name)
    if sample and samples_allowed():
        return load_sample(sample), "sample"
    return None, "unavailable"


def optional(name: str, *args: Any) -> Any:
    """For hooks with no sample (verified answers, escalation, events): None if not connected."""
    fn = _CONNECTED.get(name)
    return fn(*args) if fn is not None else None

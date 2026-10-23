"""Dependencies the API needs. Stand-ins for other members' work are marked TODO."""
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)


def current_user_id() -> str:
    # TODO: use M1's current_user() / require_member(room_id)
    return "demo-student"


def _default_provider() -> str:
    """M4_LLM_PROVIDER wins. Otherwise the app's main model (LLM_*) when it is configured, else Gemini."""
    chosen = os.getenv("M4_LLM_PROVIDER", "").strip()
    if chosen:
        return chosen
    key = os.getenv("LLM_API_KEY", "").strip()
    return "compatible" if key and not key.startswith("put-your") else "gemini"


@lru_cache
def _llm():
    from rafeeq.quiz.quiz.llm import make_llm

    return make_llm(_default_provider(), os.getenv("M4_LLM_MODEL") or None)


def get_llm():
    try:
        return _llm()
    except RuntimeError as exc:  # a missing key is "not available", not a crash
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail=f"quiz generation is not configured: {exc}") from None


@lru_cache
def _chunks() -> list[dict]:
    """Development stand-in: one book, used only when M2's knowledge base is off."""
    path = os.getenv("M4_BOOK_PDF", "data/Hands-On_Large_Language_Models.pdf")
    if not Path(path).is_file():
        log.info("quiz: no fallback book at %s (fine: quizzes use the room's indexed material)", path)
        return []
    from rafeeq.quiz.dev.book_chunks import load_chunks

    return load_chunks(path)


def get_chunks() -> list[dict]:
    return _chunks()


def room_chunks(room_id: str) -> list[dict] | None:
    """The room's own uploaded material from M2 (with the room's topics), or None when M2 is off."""
    from rafeeq.knowledge import integration

    if not integration.enabled():
        return None
    from rafeeq.knowledge.service import get_service

    return get_service().room_chunks(room_id)


def get_room_chunks():
    """Dependency: a function room_id -> chunks | None. Tests override it like the others."""
    return room_chunks


def material(room_fn, room_id: str, fallback: list[dict]) -> list[dict]:
    """What a room's quiz is made from: its own material when M2 is on, else the dev stand-in."""
    chunks = room_fn(room_id) if room_fn is not None else None
    return fallback if chunks is None else chunks


def get_approved_items():
    # TODO: call M5's approved-items endpoint. Empty for now, so quizzes fall
    # back to generated items without an error.
    return lambda room_id, topic_ids: []


@lru_cache
def _verifier():
    provider = os.getenv("M4_VERIFY_PROVIDER")
    if not provider:
        return None  # verification is off unless you set it in .env
    from rafeeq.quiz.quiz.llm import make_llm

    return make_llm(provider, os.getenv("M4_VERIFY_MODEL") or None)


def get_verifier():
    return _verifier()
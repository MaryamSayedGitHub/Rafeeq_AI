"""Find the course material for a question. The call to M2 lives in rafeeq/tutor/m2.py; this module is the short way to use it."""
from __future__ import annotations

import re

from . import m2

TOP_K = m2.TOP_K                 # the tutor never sends more than 3 passages to the model


def search_with_source(room_id: str, query: str, top_k: int = TOP_K, topic_id: str | None = None, detail: bool = False) -> tuple[list[dict], str]:
    """-> (chunks, source) with source "m2" | "sample" | "none"."""
    return m2.search(room_id, query, top_k, topic_id, detail)


def search(room_id: str, query: str, top_k: int = TOP_K) -> list[dict]:
    """-> list of {"file_name", "page", "text", "topic_id"}, at most top_k."""
    return m2.search(room_id, query, top_k)[0]


def relevant(chunks: list[dict], text: str) -> list[dict]:
    """Keep only the passages that share a real word with the text: a math problem should not be handed a biology page."""
    words = {w for w in re.findall(r"[A-Za-zء-يٱ-ۓ]{3,}", text.lower())}
    return [c for c in chunks if words & set(re.findall(r"[A-Za-zء-يٱ-ۓ]{3,}", c["text"].lower()))]

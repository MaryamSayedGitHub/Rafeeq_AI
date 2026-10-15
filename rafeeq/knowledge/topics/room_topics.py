"""Topics learned from each room's own material.

The fixed taxonomy in ``topic_manager.py`` describes one LLM course. A room about biology,
history or Python needs its own topics, otherwise every chunk lands in "General" and the
quiz, the weak-topic profile and the teacher's "who is struggling" view lose their meaning.

How it works, once per uploaded file (inside the indexing worker):

1. ``ensure_topics``: an LLM reads short excerpts of the new file and proposes 3-8 study
   topics (a name and a one-line description each), in the material's language. Topics the
   room already has are passed in and never renamed or removed, so topic ids stay stable for
   the quiz history and the student profiles. A room holds at most ``MAX_ROOM_TOPICS``.
2. Each topic is embedded once ("name: description") with the same embedder as the chunks.
3. ``assign``: every chunk gets the topic whose vector is closest to the chunk's vector
   (cosine, the vectors are unit length). Below ``min_similarity`` it gets ``topic_general``.

No LLM configured, or the LLM fails: nothing is learned and the caller keeps the keyword
tags, so indexing never fails because of topics.

Persistence: ``<store_dir>/room_topics.json``, written atomically.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rafeeq.knowledge.topics.topic_manager import GENERAL_TOPIC_ID, GENERAL_TOPIC_NAME

log = logging.getLogger(__name__)

FILE_NAME = "room_topics.json"
MAX_ROOM_TOPICS = 14            # same ceiling as a fixed taxonomy (plus the general fallback)
NEW_TOPICS_PER_FILE = (3, 8)    # what the LLM is asked for
DEFAULT_MIN_SIMILARITY = 0.30   # below this a chunk is "general"
EXCERPTS = 30                   # evenly spaced chunk excerpts sent to the LLM
EXCERPT_CHARS = 320
MAX_PROMPT_CHARS = 9000

ProposeFn = Callable[[str, str], "Mapping[str, Any] | None"]  # (system, user) -> parsed JSON or None

SYSTEM = (
    "You organise course material into study topics for a learning app used by teachers and students. "
    "Read the excerpts and reply with JSON: {\"topics\": [{\"name\": \"...\", \"description\": \"...\"}]}.\n"
    f"Give between {NEW_TOPICS_PER_FILE[0]} and {NEW_TOPICS_PER_FILE[1]} topics a teacher would use to group "
    "this material: its chapters or main concepts. Each name has 1 to 5 words. Each description is one "
    "sentence naming the key terms of the topic.\n"
    "Write names and descriptions in the same language as the material (Arabic or English).\n"
    "Do not repeat or rename the existing topics you are given; only add what they do not cover. "
    "If they already cover everything, return {\"topics\": []}.\n"
    "Never propose generic topics such as Introduction, Overview, Summary, General, Exercises or References."
)

_GENERIC = {"introduction", "overview", "summary", "general", "exercises", "references", "conclusion",
            "مقدمة", "ملخص", "عام", "تمارين", "مراجع", "خاتمة"}


def _excerpts(chunks: Sequence[Mapping[str, Any]]) -> str:
    if not chunks:
        return ""
    step = max(1, len(chunks) // EXCERPTS)
    picked = list(chunks)[::step][:EXCERPTS]
    parts, total = [], 0
    for c in picked:
        text = re.sub(r"\s+", " ", str(c.get("text", ""))).strip()[:EXCERPT_CHARS]
        if not text:
            continue
        line = f"[p.{c.get('page', '?')}] {text}"
        if total + len(line) > MAX_PROMPT_CHARS:
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def _clean_topics(raw: Any, existing_names: set[str]) -> list[dict[str, str]]:
    items = raw.get("topics") if isinstance(raw, Mapping) else None
    if not isinstance(items, list):
        return []
    out, seen = [], {n.casefold() for n in existing_names}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        name = re.sub(r"\s+", " ", str(item.get("name", ""))).strip()[:60]
        desc = re.sub(r"\s+", " ", str(item.get("description", ""))).strip()[:240]
        if not name or name.casefold() in seen or name.casefold() in _GENERIC:
            continue
        seen.add(name.casefold())
        out.append({"name": name, "description": desc})
    return out[: NEW_TOPICS_PER_FILE[1]]


class RoomTopics:
    """Per-room learned topics with their vectors. Thread-safe; one JSON file."""

    def __init__(self, store_dir: str | Path, *, min_similarity: float | None = None) -> None:
        self.path = Path(store_dir) / FILE_NAME
        self.min_similarity = (float(os.environ.get("RAFEEQ_KNOWLEDGE_TOPIC_MIN_SIM", DEFAULT_MIN_SIMILARITY))
                               if min_similarity is None else min_similarity)
        self._lock = threading.RLock()
        self._data: dict[str, list[dict[str, Any]]] | None = None

    # ---------------------------------------------------------------- storage
    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            except (OSError, json.JSONDecodeError):
                log.exception("room topics: %s is unreadable; starting empty", self.path)
                self._data = {}
        return self._data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def topics(self, room_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(t) for t in self._load().get(room_id, [])]

    def names(self, room_id: str) -> dict[str, str]:
        out = {t["topic_id"]: t["name"] for t in self.topics(room_id)}
        out.setdefault(GENERAL_TOPIC_ID, GENERAL_TOPIC_NAME)
        return out

    # ---------------------------------------------------------------- learning
    def ensure_topics(self, room_id: str, chunks: Sequence[Mapping[str, Any]], propose: ProposeFn | None,
                      embedder: Any) -> list[dict[str, Any]]:
        """Adds topics for new material (never changes existing ones). Returns the room's topics.
        Any LLM or embedding problem is logged and leaves the room's topics as they were."""
        existing = self.topics(room_id)
        if propose is None or len(existing) >= MAX_ROOM_TOPICS:
            return existing
        names = [t["name"] for t in existing]
        user = (f"Existing topics: {', '.join(names) if names else 'none'}\n\n"
                f"Material excerpts:\n{_excerpts(chunks)}")
        try:
            raw = propose(SYSTEM, user)
        except Exception:
            log.exception("room topics: the LLM call failed")
            return existing
        new = _clean_topics(raw, set(names))[: MAX_ROOM_TOPICS - len(existing)]
        if not new:
            return existing
        try:
            vectors = embedder.embed_documents([f"{t['name']}: {t['description']}" for t in new])
        except Exception:
            log.exception("room topics: embedding the topic names failed")
            return existing
        with self._lock:
            data = self._load()
            current = data.setdefault(room_id, [])
            next_n = max([int(t["topic_id"].rsplit("_", 1)[-1]) for t in current] + [0]) + 1
            for i, (topic, vec) in enumerate(zip(new, vectors)):
                current.append({"topic_id": f"topic_{next_n + i:02d}", "name": topic["name"],
                                "description": topic["description"], "vector": [float(x) for x in vec]})
            self._save()
            log.info("room topics: room %s now has %d topics", room_id, len(current))
            return [dict(t) for t in current]

    def assign(self, room_id: str, embeddings: Any) -> list[str] | None:
        """Topic id per embedding, or None when the room has no learned topics."""
        import numpy as np

        topics = self.topics(room_id)
        if not topics:
            return None
        matrix = np.asarray([t["vector"] for t in topics], dtype=np.float32)
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != matrix.shape[1]:
            log.warning("room topics: dimension mismatch (%s vs %s); keeping keyword tags",
                        vectors.shape[1], matrix.shape[1])
            return None
        sims = vectors @ matrix.T
        best = sims.argmax(axis=1)
        return [topics[j]["topic_id"] if sims[i, j] >= self.min_similarity else GENERAL_TOPIC_ID
                for i, j in enumerate(best)]

    def forget_room(self, room_id: str) -> None:
        with self._lock:
            if self._load().pop(room_id, None) is not None:
                self._save()


def default_propose() -> ProposeFn | None:
    """The LLM used to propose topics: the app's orchestrator model (LLM_API_KEY / LLM_BASE_URL /
    LLM_MODEL) when configured, else M2's Gemini settings, else None (keyword tags are kept)."""
    if os.environ.get("LLM_API_KEY", "").strip() and not os.environ["LLM_API_KEY"].startswith("put-your"):
        from rafeeq.copilot.llm import chat_json

        return chat_json
    from rafeeq.knowledge.service import llm_from_env

    llm = llm_from_env()
    if llm is None:
        return None

    def propose(system: str, user: str) -> Mapping[str, Any] | None:
        text = llm.generate(system, user).strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    return propose

"""The room's own material for the lecturer assistant: the passages of the files the teacher uploaded (M2).

A session linked to a room is prepared FROM that room's material first. Without it the model reads the topic
name on its own, and a short or ambiguous topic goes wrong: "TWT" (traveling-wave tube in the uploaded
microwave notes) became the chat slang "TWT". The web is only a fallback when the room has nothing on the topic.

room_material(brief) -> the passages as prompt text ("" when the session has no room, M2 is off, or nothing matches).
material_for(state)  -> the text the research step already found, else room_material(state["brief"]).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# Every agent gets this text: keep it well under a free tier's tokens-per-minute (Groq on_demand: ~6k for some models).
MAX_CHARS = int(os.getenv("LECTURER_MATERIAL_CHARS", "3500"))
PASSAGES_PER_QUERY = 6
MAX_QUERIES = 4           # the topic + up to 3 learning objectives
NO_MATERIAL = "None"      # the prompts are written for this exact word (see utils.format_optional)


def _service():
    from rafeeq.knowledge import integration

    if not integration.enabled():
        return None
    from rafeeq.knowledge.service import get_service

    return get_service()


def room_material(brief: dict) -> str:
    room_id = (brief.get("room_id") or "").strip()
    if not room_id:
        return ""
    try:
        svc = _service()
        if svc is None:
            return ""
        queries = [brief.get("topic", "")] + list(brief.get("learning_objectives") or [])[: MAX_QUERIES - 1]
        seen: set[tuple] = set()
        parts: list[str] = []
        size = 0
        for q in (q.strip() for q in queries):
            if not q:
                continue
            for c in svc.search(room_id, q, PASSAGES_PER_QUERY).get("chunks", []):
                key = (c.get("file_name"), c.get("page"), (c.get("text") or "")[:60])
                text = (c.get("text") or "").strip()
                if not text or key in seen:
                    continue
                seen.add(key)
                block = f"[{c.get('file_name')}, p. {c.get('page')}]\n{text}"
                if size + len(block) > MAX_CHARS:
                    break
                parts.append(block)
                size += len(block)
        if parts:
            log.info("lecturer: %d passages of room %s used as the course material", len(parts), room_id)
        else:
            log.info("lecturer: room %s has no material on %r: the draft uses the web / the model", room_id, brief.get("topic"))
        return "\n\n".join(parts)
    except Exception as e:  # noqa: BLE001  (M2 down, index missing...): prepare the draft without it
        log.warning("lecturer: could not read the room's material: %s: %s", type(e).__name__, str(e)[:200])
        return ""


def material_for(state) -> str:
    """Prompt text for the agents: the research step's passages, or a fresh search when research did not run."""
    text = state.get("material")
    if text is None:
        text = room_material(state.get("brief") or {})
    return text or NO_MATERIAL

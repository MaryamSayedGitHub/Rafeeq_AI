"""Plugs M2 into the app. Off by default so the app starts without torch / FAISS / the embedding model.

    RAFEEQ_KNOWLEDGE=1          turn it on: the /search, topics and sources routes become real and
                                every uploaded PDF is indexed in the background (status: uploaded -> indexing -> ready / failed),
                                and a deleted file is removed from the index
    RAFEEQ_KNOWLEDGE_TUTOR=1    also connect the grounded answerer as the Copilot's `tutor_answer`
                                (needs GEMINI_API_KEY and GEMINI_MODEL; M3 may replace it with their own)

Called once from rafeeq/main.py.
"""
from __future__ import annotations

import os
from typing import Any

_hook_installed = False


def enabled() -> bool:
    return os.environ.get("RAFEEQ_KNOWLEDGE", "0").strip() == "1"


def install(app: Any) -> bool:
    """Returns True when the knowledge base was plugged in."""
    global _hook_installed
    if not enabled():
        return False
    from rafeeq.knowledge.router import router
    from rafeeq.knowledge.service import get_service
    from rafeeq.platform import service as platform

    app.include_router(router)
    if not _hook_installed:
        platform.on_upload(lambda room_id, file_id, path: get_service().index_file(room_id, file_id, path))
        platform.on_delete(lambda room_id, file_id: get_service().remove_file(room_id, file_id))
        _hook_installed = True
    if os.environ.get("RAFEEQ_KNOWLEDGE_TUTOR", "0").strip() == "1":
        from rafeeq.copilot import clients

        clients.connect("tutor_answer", lambda room_id, question: get_service().tutor_answer(room_id, question))
        clients.connect("overview", lambda room_id, question: get_service().overview(room_id, question))
        clients.connect("topic_of", lambda room_id, citations: get_service().topic_for_citations(room_id, citations))
    return True

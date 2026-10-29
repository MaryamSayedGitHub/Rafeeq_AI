"""Video discovery history (M5): the teacher's earlier searches, kept so a search can be reopened without
running it again (no new YouTube scraping, no new Gemini calls).

Storage: the teacher service store, next to the profile and the library:
    ("lecturer", <teacher_id>, "video_searches")   key = a hash of the normalised question
    value = {query, room_id, time, results: [...as returned by youtube_discovery.search]}

Asking the same question again replaces the old entry (and moves it to the top). Only the newest
MAX_KEEP searches are kept. Nothing here raises into the search itself: history is a convenience.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

from rafeeq.teacher.lecturer.long_term_store import lecturer_key

log = logging.getLogger(__name__)

MAX_KEEP = 30
MAX_RESULTS = 10


def _ns(teacher_id: str) -> tuple:
    return ("lecturer", lecturer_key(teacher_id), "video_searches")


def search_key(query: str) -> str:
    """'Linked lists ' and 'linked lists' are the same search."""
    return "vs_" + hashlib.sha1(" ".join(query.lower().split()).encode()).hexdigest()[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def save(store, teacher_id: str, query: str, results: list[dict], room_id: str = "") -> str | None:
    """Keep this search; returns its id, or None when there is nothing to keep (or no store)."""
    query = " ".join(str(query or "").split())[:300]
    if store is None or not teacher_id or not query or not results:
        return None
    key = search_key(query)
    try:
        store.put(_ns(teacher_id), key, {"query": query, "room_id": room_id[:60], "time": _now(),
                                         "results": list(results)[:MAX_RESULTS]})
        old = sorted(_items(store, teacher_id), key=lambda i: str(i.value.get("time") or ""), reverse=True)[MAX_KEEP:]
        for item in old:
            store.delete(_ns(teacher_id), item.key)
    except Exception:                                                # noqa: BLE001 - history must never break a search
        log.exception("video history: could not save the search")
        return None
    return key


def _items(store, teacher_id: str) -> list:
    return list(store.search(_ns(teacher_id), limit=MAX_KEEP * 2))


def list_searches(store, teacher_id: str, room_id: str = "") -> list[dict]:
    """Newest first: [{search_id, query, room_id, time, count}] (no results, so the list stays small)."""
    if store is None or not teacher_id:
        return []
    out = []
    for item in _items(store, teacher_id):
        v = item.value
        if room_id and v.get("room_id") and v.get("room_id") != room_id:
            continue
        out.append({"search_id": item.key, "query": v.get("query", ""), "room_id": v.get("room_id", ""),
                    "time": v.get("time", ""), "count": len(v.get("results") or [])})
    return sorted(out, key=lambda s: s["time"], reverse=True)


def get_search(store, teacher_id: str, search_id: str) -> dict | None:
    if store is None or not teacher_id or not search_id.startswith("vs_"):
        return None
    item = store.get(_ns(teacher_id), search_id[:40])
    return {"search_id": item.key, **item.value} if item else None


def delete_search(store, teacher_id: str, search_id: str) -> bool:
    if get_search(store, teacher_id, search_id) is None:
        return False
    store.delete(_ns(teacher_id), search_id[:40])
    return True

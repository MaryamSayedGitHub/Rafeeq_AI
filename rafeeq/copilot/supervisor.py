"""Entry point for POST /copilot/message."""
from __future__ import annotations

from collections import defaultdict, deque

from rafeeq.copilot.graph import run

# Last 6 messages per (room, student). In memory: fine for the demo, not shared across workers.
_HISTORY: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=6))


def handle_message(room_id: str, text: str, file_id: str | None = None,
                   mode: str = "auto", user_id: str | None = None) -> dict:
    key = (room_id, user_id or "anonymous")
    out = run({"room_id": room_id, "text": text, "file_id": file_id, "mode": mode or "auto",
               "user_id": user_id, "history": list(_HISTORY[key])})
    _HISTORY[key].append({"role": "user", "text": text})
    _HISTORY[key].append({"role": "assistant", "text": out["reply"]})
    return {"route": out["route"], "reply": out["reply"], "payload": out["payload"]}


def clear_history() -> None:
    _HISTORY.clear()

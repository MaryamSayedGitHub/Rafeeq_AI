"""M3 plugs into M1's Copilot (rafeeq/copilot/clients.py). Call connect_m3() once at startup (rafeeq/main.py does).

  tutor_answer(room_id, question)        -> TutorAnswer (+ topic_id)   M3 agent: tag_topic, M2 retrieve, verified citations, question log.
                                                                       A math / code problem gets the full worked solution instead (help.py).
  check_work(room_id, file_id, text)     -> HandwritingCheck           every step ok | check | error, one hint, the REAL box of the wrong line
  read_image(image_bytes)                -> {type, language, text, confidence}   OpenCV + the vision model, line by line
  record_event(event)                    -> None                       M4 POST /events (the Copilot builds the event, M3 delivers it)

How the photo travels: M1 cleans it and stores it as storage/snaps/<file_id>.jpg, then asks read_image for the text. read_image keeps the
lines and boxes it found, keyed by the hash of the bytes. check_work finds them through that file, and swaps the stored photo for the
corrected page the boxes were measured on, so the colour boxes M1's UI draws sit on the right lines.
RAFEEQ_CONNECT_M3=0 turns the connection off (the Copilot then falls back to samples / "not available", as before).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import cv2

from rafeeq.copilot import clients
from rafeeq.snap import service as snap_service
from rafeeq.snap.reader import NotAnImage, ReaderUnavailable

from . import events
from . import help as tutor_help
from .answer import guess_language
from .identity import REQUEST_USER
from .check_work import check_work_detail
from .tutor_agent import act

log = logging.getLogger(__name__)


def _key(data: bytes) -> str:
    return "sha_" + hashlib.sha1(data).hexdigest()[:16]


def _links() -> Path:
    return snap_service.snap_dir() / "links"


def _m1_photo(file_id: str) -> Path | None:
    """The photo M1 stored for this file_id (never a path from the caller: the id must be a plain token)."""
    if not file_id.replace("_", "").isalnum():
        return None
    p = Path(os.environ.get("RAFEEQ_DATA_DIR", "storage")) / "snaps" / f"{file_id}.jpg"
    return p if p.is_file() else None


# ----------------------------------------------------------------------------- the four hooks
def read_image(data: bytes) -> dict:
    try:
        result, extra = snap_service.read_photo(data)
    except snap_service.SnapError as e:
        raise (ReaderUnavailable if "reader" in str(e) else NotAnImage)(str(e)) from e
    _links().mkdir(parents=True, exist_ok=True)
    (_links() / f"{_key(data)}.json").write_text(json.dumps({"file_id": result["file_id"], "confidence": result["confidence"], "retake": extra["retake_tip"]}), encoding="utf-8")
    return {k: result[k] for k in ("type", "language", "text", "confidence")}


def check_work(room_id: str, file_id: str, text: str) -> dict:
    lang = guess_language(text)
    mine, confidence = file_id, None
    photo = _m1_photo(file_id)
    if photo:
        link = _links() / f"{_key(photo.read_bytes())}.json"
        if link.is_file():
            info = json.loads(link.read_text(encoding="utf-8"))
            mine, confidence = info["file_id"], info["confidence"]
            page = snap_service.page_image_path(mine)
            if page:                                               # the boxes were measured on the corrected page: show that page
                img = cv2.imread(str(page))
                if img is not None:
                    cv2.imwrite(str(photo), img)
    det = check_work_detail(room_id, mine, text, lang, confidence)
    out = {k: det[k] for k in ("boxes", "steps", "hint")}
    # the extras M1's page shows: every step with its colour and its own hint, what to ask the student, and the 3-hints-then-solution panel
    out.update(items=[{k: it[k] for k in ("i", "text", "status", "hint")} for it in det["items"]],
               item_boxes=[{**it["box"], "status": it["status"]} for it in det["items"]], kind=det["kind"], complete=det["complete"],
               needs_help=det["needs_help"], needs_clarification=det["needs_clarification"], clarification=det["clarification"])
    if det["steps"] and (not det["needs_clarification"] or any(it["status"] == "error" for it in det["items"])):   # a misread line alone is asked about, not drilled
        out["help_key"] = tutor_help.prepare(text, "code" if det["kind"] == "code" else "math", lang)
    return out


LEAD = {"ar": "هذه مسألة: سأساعدك بتلميح بعد تلميح، ثم الحل الكامل بالخطوات إذا أردت.",
        "en": "This is a problem: I will help with one hint at a time, then the full solution with steps if you want it."}


def tutor_answer(room_id: str, question: str, user_id: str | None = None) -> dict:
    """The orchestrator's question goes to the tutor agent: a problem gets hints, anything else a grounded answer with citations and 3 MCQ."""
    lang = guess_language(question)                                # hints and the "this is a problem" line follow the question
    # the Copilot sends the M4 event itself (record_event), so the agent only logs the question (notify=False): no event twice
    res = act(room_id, "auto", question, user_id or REQUEST_USER.get(), lang, notify=False)
    r = res["result"]
    if res["action"] == "help":
        return {"answer": LEAD[lang], "citations": [], "grounded": True, "steps": [], "topic_id": "other", "help_key": r["key"], "problem_kind": r["kind"]}
    if not res["ok"] or not r:
        return {"answer": "", "citations": [], "grounded": False, "steps": [], "topic_id": "other"}
    return {"answer": r["answer"], "citations": r["citations"], "grounded": r["grounded"], "steps": r["steps"], "topic_id": r["topic_id"],
            "question_id": r["question_id"], "offer_level2": r["offer_level2"], "quiz": r["quiz"], "citation_support": r["citation_support"],
            "answer_flagged": r["answer_flagged"], "material_source": r["material_source"], "language": r["language"], "web_sources": r["web_sources"],
            "notice": r.get("error") or "", "off_topic": bool(r.get("off_topic"))}


def record_event(event: dict) -> None:
    events.post_event(event)


record_event.is_m3 = True                                          # events.post_event must not hand the event back to this very function


def connect_m3(force: bool = False) -> bool:
    """-> True when M3 is connected. Safe to call twice."""
    if os.environ.get("RAFEEQ_CONNECT_M3", "1") == "0" and not force:
        return False
    for name, fn in (("tutor_answer", tutor_answer), ("check_work", check_work), ("read_image", read_image), ("record_event", record_event)):
        if name == "record_event" and clients.is_connected(name) and not getattr(clients._CONNECTED.get(name), "is_m3", False):
            continue                                               # M4 already reads the events in this app (rafeeq/quiz/copilot_hooks.py): keep its hook
        clients.connect(name, fn)
    return True


def disconnect_m3() -> None:
    for name in ("tutor_answer", "check_work", "read_image", "record_event"):
        clients.disconnect(name)

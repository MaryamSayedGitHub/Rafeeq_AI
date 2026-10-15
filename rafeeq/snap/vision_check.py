"""Handwriting check, the whole flow in one call.

  photo -> quality check (blurry? retake tip) -> OpenCV (perspective, deskew, denoise, adaptive threshold) -> text lines with boxes
        -> vision-language model reads every line -> M2 retrieve (course passages) -> every step ok | check | error, exactly one hint
        for each step that is not ok -> M4 event

Used by POST /rooms/{room_id}/vision/check-work. The Snap page uses the same two steps (read_photo, then check_work_detail) with a
form in between, so the student can correct a misread line before the check.
"""
from __future__ import annotations

import base64

import cv2

from rafeeq.tutor.agent import run_check

from .annotate import draw_boxes
from .service import page_image_path, read_photo


def annotated_png(file_id: str, items: list[dict]) -> bytes | None:
    """The cleaned page with a colour box on every line: green = ok, orange = check, red = error (the numbers match the steps)."""
    path = page_image_path(file_id)
    if not path or not items:
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    boxes = [(int(it["box"]["x"] * w), int(it["box"]["y"] * h), int(it["box"]["width"] * w), int(it["box"]["height"] * h)) for it in items]
    ok, buf = cv2.imencode(".png", draw_boxes(img, boxes, [{"i": it["i"], "status": it["status"]} for it in items]))
    return buf.tobytes() if ok else None


def check_photo(data: bytes, room_id: str, user_id: str | None = None, lang: str = "ar", annotate: bool = False) -> dict:
    """-> the vision result. Raises rafeeq.snap.service.SnapError when the file is not an image or the reader is not available."""
    snap, extra = read_photo(data, room_id)
    out = {"file_id": snap["file_id"], "quality_ok": extra["quality_ok"], "retake_tip": extra["retake_tip"], "language": snap["language"],
           "type": snap["type"], "confidence": snap["confidence"], "kind": "question", "complete": True, "needs_help": False,
           "lines": [], "citations": [], "hint": "", "needs_clarification": False, "clarification": "", "event_delivered": False, "annotated_png_base64": None}
    if not extra["quality_ok"] or not snap["text"]:                 # blurry or empty: no check, the student retakes the photo
        return out
    st = run_check(room_id, snap["text"], snap["file_id"], user_id, lang, snap["confidence"])   # the same LangGraph agent: check_work -> M4 event
    det = st.get("work")
    if not det:                                                    # the checker itself failed: say so, never a made-up verdict
        out.update(hint=st.get("error") or "check failed", needs_clarification=True, clarification=st.get("error") or "check failed")
        return out
    out.update(kind=det["kind"], complete=det["complete"], needs_help=det["needs_help"], hint=det["hint"], needs_clarification=det["needs_clarification"],
               clarification=det["clarification"], citations=det["passages"],
               lines=[{"i": it["i"], "text": it["text"], "status": it["status"], "hint": it["hint"], "box": it["box"]} for it in det["items"]])
    out["event_delivered"] = bool(st.get("event_delivered"))
    if annotate:
        png = annotated_png(snap["file_id"], det["items"])
        out["annotated_png_base64"] = base64.b64encode(png).decode() if png else None
    return out

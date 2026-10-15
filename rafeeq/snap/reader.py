"""POST /snap: clean the photo (OpenCV), then read it.

Who reads the text, in order:
  1. M3's OCR, if connected:  clients.connect("read_image", fn)
  2. A vision-language model, if LLM_API_KEY is set
  3. Nobody: ReaderUnavailable is raised and the route answers 503.
     We never return sample text as if it had been read from the student's photo.
"""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from rafeeq.copilot import clients
from rafeeq.copilot.llm import configured as llm_configured, vision_json
from rafeeq.snap.cleaner import NotAnImage, clean

__all__ = ["read_snap", "NotAnImage", "ReaderUnavailable"]


class ReaderUnavailable(RuntimeError):
    pass


SYSTEM = (
    "You read a photo a student took in a study app. Reply with JSON: "
    '{"type": "student_work" | "material" | "diagram" | "other", '
    '"language": "ar" | "en", "text": "<all text in reading order>", "confidence": 0..1}. '
    "student_work = the student's own handwritten solution or notes. "
    "material = a printed page or slide. diagram = mainly a figure. "
    "Write math exactly as written. Do not solve or correct anything."
)
TYPES = ("student_work", "material", "diagram", "other")


def _store_dir() -> Path:
    return Path(os.environ.get("RAFEEQ_DATA_DIR", "storage")) / "snaps"


def _normalize(out: dict, sharpness: float | None) -> dict:
    lang = out.get("language") if out.get("language") in ("ar", "en") else "ar"
    kind = out.get("type") if out.get("type") in TYPES else "other"
    try:
        conf = max(0.0, min(1.0, float(out.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    if sharpness is not None:
        conf = min(conf, sharpness / 100)  # a blurry photo can't be trusted; the UI asks for a retake below 0.5
    return {"type": kind, "language": lang, "text": str(out.get("text", "")), "confidence": round(conf, 2)}


def read_snap(photo_bytes: bytes) -> dict:
    """Raises NotAnImage (-> 422) or ReaderUnavailable (-> 503)."""
    cleaned, sharpness = clean(photo_bytes)

    if clients.is_connected("read_image"):
        out = clients.optional("read_image", cleaned)
    elif llm_configured():
        out = vision_json(SYSTEM, "Read this photo.", cleaned)
        if out is None:
            raise ReaderUnavailable("The vision model did not answer")
    else:
        raise ReaderUnavailable("No image reader is configured")

    file_id = f"snap_{uuid4().hex[:8]}"
    folder = _store_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{file_id}.jpg").write_bytes(cleaned)  # M3's check-work can load it by file_id
    return {"file_id": file_id, **_normalize(out or {}, sharpness)}

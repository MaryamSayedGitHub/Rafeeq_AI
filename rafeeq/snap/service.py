"""Snap & Ask: a photo -> the text on it, what kind of photo it is, and where every line sits (contracts: SnapResult).

OpenCV cleans the photo and finds the text lines, a vision-language model reads each line (SNAP_READER = hosted | ollama | gemini; hosted is the default when LLM_API_KEY is set).
The cleaned page image and the line boxes are saved under data/snap/<file_id>.*, so /tutor/check-work can box the REAL wrong line.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from .pipeline import run_pipeline
from .preprocess import quality_check
from .readers import default_reader, get_reader

ROOT = Path(__file__).resolve().parents[2]


def snap_dir() -> Path:
    d = Path(os.getenv("SNAP_DATA_DIR") or ROOT / "data" / "snap")
    d.mkdir(parents=True, exist_ok=True)
    return d


class SnapError(Exception):
    """The photo could not be opened, or the reader is not available."""


def _language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    return "ar" if letters and sum("؀" <= c <= "ۿ" for c in letters) > len(letters) / 2 else "en"


_MATH = re.compile(r"[=+\-*/^×÷√]|\d")
_CODE = re.compile(r"\bdef\b|\breturn\b|\bprint\s*\(|\bfor\b.*\bin\b|\bimport\b|\bclass\b|==|;\s*$|=>", re.M)


def _type(lines: list[str]) -> str:
    """student_work = math steps or code written by the student; material = a printed page (long prose); other = a short question; diagram = no text."""
    text = "\n".join(lines).strip()
    if not text:
        return "diagram"
    if _CODE.search(text):
        return "student_work"
    words = len(re.findall(r"[A-Za-z؀-ۿ]{3,}", text))
    if len(lines) >= 2 and sum(bool(_MATH.search(l)) for l in lines) >= max(2, len(lines) // 2) and words <= 3 * len(lines):
        return "student_work"
    return "material" if len(lines) > 6 else "other"


def read_photo(data: bytes, room_id: str | None = None) -> tuple[dict, dict]:
    """-> (SnapResult, extra). extra = {"quality_ok", "retake_tip", "lines", "boxes"} for the web page; the API returns only SnapResult."""
    try:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None       # an empty upload makes OpenCV raise, not return None
    except cv2.error:
        img = None
    if img is None:
        raise SnapError("Could not open the photo. Send a JPG or PNG.")
    file_id = "f_" + uuid.uuid4().hex[:8]
    ok, tip, _metrics = quality_check(img)
    if not ok:
        return ({"file_id": file_id, "type": "other", "language": "ar", "text": "", "confidence": 0.0},
                {"quality_ok": False, "retake_tip": tip, "lines": [], "boxes": []})
    try:
        res = run_pipeline(img, get_reader(default_reader()), "pre_gray", force_lines=True)
    except Exception as e:                                         # noqa: BLE001  (model not running, no key...)
        raise SnapError(f"The handwriting reader is not available ({type(e).__name__}). Check LLM_API_KEY / LLM_MODEL (the model must accept images), or run Ollama with the vision model and set SNAP_READER=ollama.") from e
    pairs = [(b, l.strip()) for b, l in zip(res["boxes"], res["lines"]) if l.strip()]
    boxes = [b for b, _ in pairs]
    lines = [l for _, l in pairs]
    page = res["prepped"].color
    h, w = page.shape[:2]
    frac = [{"x": round(x / w, 4), "y": round(y / h, 4), "width": round(bw / w, 4), "height": round(bh / h, 4)} for x, y, bw, bh in boxes]
    text = "\n".join(lines)
    confidence = round(0.5 + 0.45 * (len(lines) / max(1, len(res["lines"]))), 2) if lines else 0.0
    result = {"file_id": file_id, "type": _type(lines), "language": _language(text), "text": text, "confidence": confidence}
    cv2.imwrite(str(snap_dir() / f"{file_id}.png"), page)
    (snap_dir() / f"{file_id}.json").write_text(json.dumps({"room_id": room_id, "lines": lines, "boxes": frac, "text": text}, ensure_ascii=False), encoding="utf-8")
    return result, {"quality_ok": True, "retake_tip": "", "lines": lines, "boxes": frac}


def load_lines(file_id: str) -> dict | None:
    """The saved lines and boxes of a photo, or None (unknown id, never an arbitrary path)."""
    if not re.fullmatch(r"f_[0-9a-f]{8}", file_id or ""):
        return None
    p = snap_dir() / f"{file_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def page_image_path(file_id: str) -> Path | None:
    if not re.fullmatch(r"f_[0-9a-f]{8}", file_id or ""):
        return None
    p = snap_dir() / f"{file_id}.png"
    return p if p.exists() else None

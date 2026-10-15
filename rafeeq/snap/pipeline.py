"""preprocess -> detect text lines -> read. Shared by evaluate.py and the tutor agent."""
from __future__ import annotations

from .lines import crop_lines, detect_lines
from .preprocess import PRESETS, preprocess


def run_pipeline(img, reader, preset: str = "pre", line_method: str = "projection", force_lines: bool = False) -> dict:
    p = preprocess(img, **PRESETS[preset])
    if reader.use_lines or force_lines:
        boxes = detect_lines(p.lines_src, method=line_method)
        if not boxes:                                          # nothing found: read the whole image as one line
            h, w = p.reader.shape[:2]
            boxes = [(0, 0, w, h)]
        crops = crop_lines(p.reader, boxes)
        lines = reader.read_many(crops) if hasattr(reader, "read_many") else [reader.read(c) for c in crops]       # a hosted model reads the page in ONE request (rate limits)
    else:
        boxes, lines = [], [reader.read(p.reader)]
    return {"prepped": p, "boxes": boxes, "lines": lines, "text": "\n".join(lines)}

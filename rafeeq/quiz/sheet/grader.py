"""Read the bubbles of a photographed answer sheet with OpenCV.

Steps: find the 4 corner squares -> straighten the page -> remove uneven
lighting -> measure how dark each bubble is -> decide per question.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rafeeq.quiz.sheet import layout as L

SCALE = 2.0  # pixels per point on the straightened page
CANON_W = int(L.PAGE_W * SCALE)
CANON_H = int(L.PAGE_H * SCALE)

DARK_LEVEL = 150  # after lighting fix: below this (of 255) counts as ink
MARK_MIN = 0.30  # share of dark pixels inside a bubble to call it marked
DOUBLE_RATIO = 0.6  # second mark this strong (vs the first) = two marks
MAX_SIDE = 1600  # photos are shrunk to this before processing


class SheetError(ValueError):
    """The photo cannot be read as an answer sheet (message is for the student)."""


@dataclass
class Mark:
    choice: int | None  # 0..3, or None when blank / two marks
    confidence: float  # 0..1
    fills: list[float]  # dark share of A..D (useful for debugging)


def _decode(data: bytes) -> np.ndarray:
    gray = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SheetError("this file is not an image")
    h, w = gray.shape
    if max(h, w) > MAX_SIDE:
        k = MAX_SIDE / max(h, w)
        gray = cv2.resize(gray, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return gray


def _find_markers(gray: np.ndarray) -> dict[str, tuple[float, float]]:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block = int(min(h, w) * 0.12) | 1
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(h * w)
    cands: list[tuple[float, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (2e-4 * img_area <= area <= 1e-2 * img_area):
            continue
        _, (rw, rh), _ = cv2.minAreaRect(c)
        if min(rw, rh) < 1 or min(rw, rh) / max(rw, rh) < 0.6:
            continue
        if area / (rw * rh) < 0.80:  # a square scores ~0.9+, a circle at most ~0.8
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            cands.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))

    found: dict[str, tuple[float, float]] = {}
    used: set[int] = set()
    for name, (px, py) in {"tl": (0, 0), "tr": (w, 0), "bl": (0, h), "br": (w, h)}.items():
        best = None
        for i, (cx, cy) in enumerate(cands):
            if i in used:
                continue
            d = (cx - px) ** 2 + (cy - py) ** 2
            if best is None or d < best[0]:
                best = (d, i)
        if best is None:
            raise SheetError(
                "could not find the 4 black corner squares; photograph the whole "
                "page with all four corners visible and good light"
            )
        used.add(best[1])
        found[name] = cands[best[1]]

    tl, tr, bl, br = (found[k] for k in ("tl", "tr", "bl", "br"))
    if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
        raise SheetError("the page looks rotated or upside down; hold the phone upright")
    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    if cv2.contourArea(quad) < 0.08 * img_area:
        raise SheetError("the page is too small in the photo; move closer")

    # The four squares must form the same rectangle as on the printed page. Without this check,
    # a photo that cuts off the bottom of the page makes filled bubbles pass as corner squares
    # and the sheet is read wrongly instead of rejected.
    def _dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    top, bottom = _dist(tl, tr), _dist(bl, br)
    left, right = _dist(tl, bl), _dist(tr, br)
    cx, cy = L.MARKER_CENTERS["tl"], L.MARKER_CENTERS["br"]
    want = (L.MARKER_CENTERS["tr"][0] - cx[0]) / (cy[1] - cx[1])  # width / height of the square layout
    ok = (
        min(top, bottom) > 0
        and min(left, right) > 0
        and abs(top - bottom) / max(top, bottom) < 0.25
        and abs(left - right) / max(left, right) < 0.25
        and abs(((top + bottom) / (left + right)) - want) / want < 0.25
    )
    if not ok:
        raise SheetError(
            "the four corner squares do not form the page; photograph the WHOLE page "
            "with all four black corner squares visible (nothing cut off, no stray marks)"
        )
    return found


def _straighten(gray: np.ndarray, markers: dict[str, tuple[float, float]]) -> np.ndarray:
    order = ("tl", "tr", "br", "bl")
    src = np.float32([markers[k] for k in order])
    dst = np.float32([[L.MARKER_CENTERS[k][0] * SCALE, L.MARKER_CENTERS[k][1] * SCALE] for k in order])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, m, (CANON_W, CANON_H), flags=cv2.INTER_LINEAR, borderValue=255)


def _remove_shadows(warped: np.ndarray) -> np.ndarray:
    """Divide by an estimate of the empty paper, so shadows and ink compare fairly."""
    k = int(L.BUBBLE_R * SCALE * 2.6) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    paper = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)  # erases dark blobs smaller than k
    paper = np.maximum(paper, 1)
    return cv2.divide(warped, paper, scale=255)


def _fills(dark: np.ndarray, index: int) -> list[float]:
    r = int(round(L.BUBBLE_R * SCALE * 0.62))  # inside the printed ring
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    disc = (xx * xx + yy * yy) <= r * r
    out: list[float] = []
    for j in range(4):
        cx, cy = L.bubble_center(index, j)
        x, y = int(round(cx * SCALE)), int(round(cy * SCALE))
        patch = dark[y - r : y + r + 1, x - r : x + r + 1]
        out.append(float(patch[disc].mean()) if patch.shape == disc.shape else 0.0)
    return out


def _decide(fills: list[float]) -> Mark:
    order = sorted(range(4), key=lambda j: -fills[j])
    best, second = fills[order[0]], fills[order[1]]
    if best < MARK_MIN:
        return Mark(None, 0.0, fills)  # blank
    if second >= MARK_MIN and second > DOUBLE_RATIO * best:
        return Mark(None, 0.0, fills)  # two marks
    conf = min(1.0, best / 0.6) * (1.0 - second / best)
    return Mark(order[0], round(conf, 3), fills)


def detect_marks(data: bytes, n_items: int) -> list[Mark]:
    """One Mark per question, in sheet order. Raises SheetError if the page can't be found."""
    if not 1 <= n_items <= L.MAX_ITEMS:
        raise ValueError(f"n_items must be 1..{L.MAX_ITEMS}")
    gray = _decode(data)
    markers = _find_markers(gray)
    flat = _straighten(gray, markers)
    norm = _remove_shadows(flat)
    dark = norm < DARK_LEVEL
    return [_decide(_fills(dark, i)) for i in range(n_items)]
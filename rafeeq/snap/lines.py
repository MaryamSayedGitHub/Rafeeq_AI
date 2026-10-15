"""Find text lines as boxes (x, y, w, h), top to bottom. Two methods: projection profile or contours."""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import to_gray


def ink_mask(img: np.ndarray, remove_rules: bool = True) -> np.ndarray:
    """Ink = 255. Works for BGR, gray or already-binary input. Removes notebook ruling lines."""
    gray = to_gray(img)
    if np.unique(gray[::4, ::4]).size <= 2:                    # already binary (black text on white)
        mask = (gray < 128).astype(np.uint8) * 255
    else:
        mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h, w = mask.shape
    m = max(3, int(0.015 * min(h, w)))                         # page edges / table / shadows at the border are not text
    mask[:m, :], mask[-m:, :], mask[:, :m], mask[:, -m:] = 0, 0, 0, 0
    if remove_rules:
        hor = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 4), 1)))
        ver = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 4))))
        mask = cv2.subtract(mask, cv2.dilate(hor, np.ones((3, 1), np.uint8)))
        mask = cv2.subtract(mask, cv2.dilate(ver, np.ones((1, 3), np.uint8)))
    return mask


def _pad_box(x, y, w, h, pad, W, H):
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def detect_lines_projection(img, min_ink_frac=0.02, min_h_frac=0.012, min_w_frac=0.03, pad_frac=0.004):
    mask = ink_mask(img)
    H, W = mask.shape
    joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 40), 1)))  # join words
    prof = (joined > 0).sum(axis=1).astype(float)
    k = max(3, H // 200)
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    on = prof > max(3.0, min_ink_frac * W)

    segs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            segs.append([start, y])
            start = None
    if start is not None:
        segs.append([start, H])

    gap = max(3, H // 150)
    merged: list[list[int]] = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)

    boxes, pad, min_h = [], max(2, int(pad_frac * H)), max(8, int(min_h_frac * H))
    for y0, y1 in merged:
        if y1 - y0 < min_h:
            continue
        cols = np.where((joined[y0:y1] > 0).any(axis=0))[0]
        if cols.size == 0 or (cols[-1] - cols[0]) < min_w_frac * W:
            continue
        boxes.append(_pad_box(cols[0], y0, cols[-1] - cols[0] + 1, y1 - y0, pad, W, H))
    return boxes


def _merge_vertical(boxes, min_overlap=0.5):
    out: list[list[int]] = []
    for x, y, w, h in sorted(boxes, key=lambda b: b[1]):
        if out:
            px, py, pw, ph = out[-1]
            overlap = min(py + ph, y + h) - max(py, y)
            if overlap > min_overlap * min(ph, h):
                nx, ny = min(px, x), min(py, y)
                out[-1] = [nx, ny, max(px + pw, x + w) - nx, max(py + ph, y + h) - ny]
                continue
        out.append([x, y, w, h])
    return [tuple(b) for b in out]


def detect_lines_contours(img, min_h_frac=0.012, min_w_frac=0.03, pad_frac=0.004):
    mask = ink_mask(img)
    H, W = mask.shape
    blob = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, W // 12), max(3, H // 300))))
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = [b for b in boxes if b[3] >= max(8, min_h_frac * H) and b[2] >= min_w_frac * W]
    pad = max(2, int(pad_frac * H))
    return [_pad_box(*b, pad, W, H) for b in _merge_vertical(boxes)]


def detect_lines(img, method: str = "projection"):
    boxes = detect_lines_projection(img) if method == "projection" else detect_lines_contours(img)
    return sorted(boxes, key=lambda b: b[1])


def crop_lines(img: np.ndarray, boxes, border: int = 8):
    """Crop every box and add a white margin (VLMs read better with some air around the text)."""
    white = (255,) * img.shape[2] if img.ndim == 3 else 255
    return [cv2.copyMakeBorder(img[y:y + h, x:x + w], border, border, border, border, cv2.BORDER_CONSTANT, value=white)
            for x, y, w, h in boxes]

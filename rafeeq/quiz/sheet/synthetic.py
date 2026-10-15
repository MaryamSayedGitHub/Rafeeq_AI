"""Fake answer sheets and fake phone photos, for tests and demos without printing."""
from __future__ import annotations

import cv2
import numpy as np

from rafeeq.quiz.sheet import layout as L
from rafeeq.quiz.sheet.grader import CANON_H, CANON_W, SCALE


def draw_sheet(answers: list, scale: float = SCALE, ink: int = 40) -> np.ndarray:
    """Gray image of a filled sheet. Each answer is 0..3, None (blank) or a tuple (two marks)."""
    img = np.full((int(L.PAGE_H * scale), int(L.PAGE_W * scale)), 255, np.uint8)
    half = L.MARKER_SIZE * scale / 2
    for cx, cy in L.MARKER_CENTERS.values():
        cv2.rectangle(img, (int(cx * scale - half), int(cy * scale - half)),
                      (int(cx * scale + half), int(cy * scale + half)), 0, -1)
    cv2.putText(img, "EduPilot quiz", (int(150 * scale), int(85 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 2, cv2.LINE_AA)
    for i, a in enumerate(answers):
        marked = a if isinstance(a, (tuple, list)) else (a,)
        x0, y0 = L.bubble_center(i, 0)
        cv2.putText(img, str(i + 1), (int((x0 - 42) * scale), int((y0 + 4) * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 1, cv2.LINE_AA)
        for j in range(4):
            cx, cy = L.bubble_center(i, j)
            c = (int(cx * scale), int(cy * scale))
            cv2.circle(img, c, int(L.BUBBLE_R * scale), 0, max(1, int(scale)), cv2.LINE_AA)
            if j in marked:
                cv2.circle(img, c, int(L.BUBBLE_R * scale * 0.8), ink, -1, cv2.LINE_AA)
    return img


def fake_photo(sheet: np.ndarray, rng: np.random.Generator, tilt: float = 1.0,
               shadow: float = 0.35, noise: float = 6.0, blur: float = 1.2) -> bytes:
    """JPEG bytes of the sheet as if photographed: perspective, uneven light, noise, blur."""
    h, w = sheet.shape
    out_w, out_h = int(w * 1.25), int(h * 1.2)
    mx, my = w * 0.1, h * 0.08

    def j() -> float:
        return float(rng.uniform(-40, 40)) * tilt

    dst = np.float32([[mx + j(), my + j()], [w + mx + j(), my + j()],
                      [w + mx + j(), h + my + j()], [mx + j(), h + my + j()]])
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    photo = cv2.warpPerspective(sheet, cv2.getPerspectiveTransform(src, dst),
                                (out_w, out_h), borderValue=110).astype(np.float32)
    light = np.outer(np.linspace(1.0, 1.0 - shadow * rng.random(), out_h),
                     np.linspace(1.0, 1.0 - shadow * rng.random(), out_w))
    photo = photo * light + rng.normal(0, noise, photo.shape)
    photo = cv2.GaussianBlur(np.clip(photo, 0, 255).astype(np.uint8), (0, 0), blur)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 75])
    assert ok
    return buf.tobytes()
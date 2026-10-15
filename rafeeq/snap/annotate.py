"""Coloured boxes on the photo: green = ok, orange = check, red = error (hints are shown beside it in the app)."""
import cv2
import numpy as np

COLORS_BGR = {"ok": (60, 170, 60), "check": (0, 150, 255), "error": (40, 40, 220)}
COLORS_HEX = {"ok": "#3caa3c".replace(" ", ""), "check": "#ff9600", "error": "#dc2828"}


def draw_boxes(img: np.ndarray, boxes, steps) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    t = max(2, out.shape[1] // 400)
    for (x, y, w, h), s in zip(boxes, steps):
        c = COLORS_BGR[s["status"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), c, t)
        r = max(12, out.shape[1] // 60)
        cv2.circle(out, (x - r if x - r > 0 else x + r, y + r), r, c, -1)
        cv2.putText(out, str(s["i"]), (max(0, x - r) if x - r > 0 else x, y + r + r // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, r / 24, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    return out

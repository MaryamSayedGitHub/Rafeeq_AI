"""OpenCV preprocessing: perspective correction -> deskew -> denoise -> adaptive threshold.

Every step can be switched on/off so you can measure its effect (see PRESETS and evaluate.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ----------------------------------------------------------------------------- io
def imread(path) -> np.ndarray:
    """cv2.imread that also works with Arabic / non-ASCII file names on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {path} (HEIC is not supported, convert to JPG/PNG)")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def limit_size(img: np.ndarray, max_side: int = 2000) -> np.ndarray:
    s = max_side / max(img.shape[:2])
    if s >= 1:
        return img
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- quality
def quality_check(img: np.ndarray, min_sharp: float = 60.0, min_contrast: float = 40.0, min_mean: float = 70.0):
    """Return (ok, tip, metrics). Thresholds are heuristics: tune them on your own 20 photos."""
    gray = to_gray(img)
    s = 1000 / gray.shape[1]
    gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics = {"sharpness": round(sharp, 1), "contrast": round(float(np.percentile(gray, 99) - np.percentile(gray, 0.5)), 1), "brightness": round(float(gray.mean()), 1)}
    if metrics["brightness"] < min_mean:
        return False, "الصورة مظلمة. صوّر في إضاءة أحسن | Image is too dark: retake with more light.", metrics
    if sharp < min_sharp:
        return False, "الصورة مش واضحة (blurry). ثبّت الموبايل وصوّر تاني | Image is blurry: hold the phone steady and retake.", metrics
    if metrics["contrast"] < min_contrast:
        return False, "الكتابة باهتة. استخدم قلم أغمق أو إضاءة أحسن | Low contrast: use a darker pen or better light.", metrics
    return True, "", metrics


# ----------------------------------------------------------------------------- perspective
def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]  # top-right
    rect[3] = pts[np.argmax(d)]  # bottom-left
    return rect


def four_point_transform(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = rect = order_points(pts)
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def correct_perspective(img: np.ndarray, min_area_ratio: float = 0.25):
    """Find the page (largest 4-corner contour) and flatten it. Returns (img, found)."""
    s = min(1.0, 800 / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.GaussianBlur(to_gray(small), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_min = min_area_ratio * small.shape[0] * small.shape[1]
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > area_min:
            return four_point_transform(img, approx.reshape(4, 2).astype("float32") / s), True
    return img, False


# ----------------------------------------------------------------------------- deskew
def adaptive_inv(gray: np.ndarray) -> np.ndarray:
    """Ink = 255. Adaptive so shadows on phone photos do not turn into big black blobs."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    bs = max(15, (min(gray.shape) // 25) | 1)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 12)


def estimate_skew(img: np.ndarray, max_angle: float = 15.0, step: float = 0.5) -> float:
    """Angle (deg) that maximises the variance of the row projection = text lines horizontal."""
    mask = adaptive_inv(to_gray(img))
    s = min(1.0, 640 / max(mask.shape))
    small = cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else mask
    h, w = small.shape
    centre = (w / 2, h / 2)
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        m = cv2.getRotationMatrix2D(centre, float(a), 1.0)
        r = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(r.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew_image(img: np.ndarray, max_angle: float = 15.0):
    angle = estimate_skew(img, max_angle)
    if abs(angle) < 0.3:
        return img, 0.0
    return rotate(img, angle), angle


# ----------------------------------------------------------------------------- denoise / threshold
def denoise_gray(gray: np.ndarray, h: int = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def adaptive_threshold(gray: np.ndarray, min_speck: int = 12) -> np.ndarray:
    """Black text on white background, tiny specks removed."""
    ink = adaptive_inv(gray)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_speck
    ink = np.where(keep[labels], 255, 0).astype(np.uint8)
    return 255 - ink


# ----------------------------------------------------------------------------- pipeline
@dataclass
class Prepped:
    color: np.ndarray            # after perspective/deskew, BGR (draw boxes on this one)
    gray: np.ndarray
    binary: np.ndarray | None
    reader: np.ndarray           # what goes to the OCR / VLM
    lines_src: np.ndarray        # what line detection looks at
    info: dict = field(default_factory=dict)


PRESETS = {
    "raw":      dict(perspective=False, deskew=False, denoise=False, threshold=False),
    "pre":      dict(perspective=True,  deskew=True,  denoise=True,  threshold=True),
    "pre_gray": dict(perspective=True,  deskew=True,  denoise=True,  threshold=True, feed_gray=True),
    # single-step ablations
    "only_perspective": dict(perspective=True,  deskew=False, denoise=False, threshold=False),
    "only_deskew":      dict(perspective=False, deskew=True,  denoise=False, threshold=False),
    "only_denoise":     dict(perspective=False, deskew=False, denoise=True,  threshold=False),
    "only_threshold":   dict(perspective=False, deskew=False, denoise=False, threshold=True),
}


def preprocess(img: np.ndarray, perspective=True, deskew=True, denoise=True, threshold=True,
               feed_gray=False, max_side=2000) -> Prepped:
    info: dict = {}
    cur = limit_size(img, max_side)               # same resize for every preset (fair comparison)
    if perspective:
        cur, info["page_found"] = correct_perspective(cur)
    if deskew:
        cur, info["skew_deg"] = deskew_image(cur)
    gray = to_gray(cur)
    if denoise:
        gray = denoise_gray(gray)
    binary = adaptive_threshold(gray) if threshold else None

    if binary is not None and not feed_gray:
        reader = binary
    elif denoise or threshold:
        reader = gray
    else:
        reader = cur
    lines_src = binary if binary is not None else reader
    return Prepped(color=cur, gray=gray, binary=binary, reader=reader, lines_src=lines_src, info=info)

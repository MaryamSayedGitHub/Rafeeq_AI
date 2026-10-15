"""OpenCV clean-up before any reader sees the photo.

Steps, each skipped safely when it does not apply:
1. decode, applying the phone's EXIF rotation (portrait photos are stored sideways with a "rotate" tag)
2. resize so the longest side is at most 1600 px (enough for handwriting, far fewer model tokens)
3. find the page (largest 4-corner shape covering >= 30% of the photo) and flatten it (perspective warp)
4. measure sharpness (variance of the Laplacian: low = blurry)
5. boost local contrast (CLAHE) so shadows and faint pencil become readable
"""
from __future__ import annotations

import io

MAX_SIDE = 1600
MIN_PAGE_AREA = 0.30


class NotAnImage(ValueError):
    pass


def _decode_upright(photo_bytes: bytes):
    """BGR array with the EXIF rotation applied, or None if the bytes are not an image."""
    import cv2
    import numpy as np

    try:
        from PIL import Image, ImageOps

        img = ImageOps.exif_transpose(Image.open(io.BytesIO(photo_bytes))).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:  # no Pillow, or a format Pillow cannot read: OpenCV's own decoder
        return cv2.imdecode(np.frombuffer(photo_bytes, np.uint8), cv2.IMREAD_COLOR)


def _order(pts):
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    import numpy as np

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], dtype="float32")


def find_page(gray):
    """The page's 4 corners, or None when no large 4-corner shape is found."""
    import cv2

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < MIN_PAGE_AREA * h * w:
            break
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            return _order(approx.reshape(4, 2).astype("float32"))
    return None


def _warp(img, quad):
    import cv2
    import numpy as np

    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return img
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (width, height))


def clean(photo_bytes: bytes) -> tuple[bytes, float | None]:
    """Returns (cleaned JPEG bytes, sharpness). Sharpness is None if OpenCV is not installed,
    in which case the original bytes are passed through unchanged."""
    try:
        import cv2
    except ImportError:
        if not photo_bytes:
            raise NotAnImage("empty upload")
        return photo_bytes, None

    img = _decode_upright(photo_bytes) if photo_bytes else None
    if img is None:
        raise NotAnImage("not an image")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad = find_page(gray)
    if quad is not None:  # straighten only when a whole page was found
        gray = cv2.cvtColor(_warp(img, quad), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, buf = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes(), sharpness

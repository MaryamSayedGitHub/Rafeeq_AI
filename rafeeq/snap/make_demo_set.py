"""A DEMO test set to check that the measuring pipeline works: 20 images (5 Arabic, 5 English, 5 math, 5 code) with their ground truth.

    python -m rafeeq.snap.make_demo_set                 -> data/demo_images/*.jpg + data/demo_ground_truth.csv

THESE ARE NOT REAL HANDWRITING. They are text drawn with a handwriting-style font, then rotated, tinted, blurred and put in perspective
like a phone photo, so every OpenCV step has something to correct. They prove the pipeline end to end; the numbers you report must come
from your 20 REAL photos in data/images/ (see rafeeq/snap/init_gt.py and evaluate.py).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT, GT = ROOT / "data" / "demo_images", ROOT / "data" / "demo_ground_truth.csv"
FONTS = Path("C:/Windows/Fonts")

SAMPLES = {
    "ar": ["المعادلة من الدرجة الثانية لها جذران", "الميتوكوندريا تنتج الطاقة للخلية", "الخلية هي وحدة بناء الكائن الحي",
           "القوة تساوي الكتلة في التسارع", "الماء يتكون من الهيدروجين والأكسجين"],
    "en": ["Newton's second law states that F = ma", "The mitochondria produce energy for the cell", "A list is an ordered collection of items",
           "Water boils at one hundred degrees Celsius", "The derivative measures the rate of change"],
    "math": ["2x + 6 = 14\n2x = 14 - 6\n2x = 8\nx = 4", "3x - 5 = 10\n3x = 10 + 5\n3x = 15\nx = 5", "x^2 = 49\nx = 7", "5 + 3 * 2 = 11", "4(x + 2) = 20\nx + 2 = 5\nx = 3"],
    "code": ["def add(a, b):\n    return a + b", "for i in range(3):\n    print(i)", "x = [1, 2, 3]\nprint(len(x))", "if n > 0:\n    print('yes')", "def sq(n):\n    return n * n"],
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    name = {"ar": "majalla.ttf", "en": "Inkfree.ttf", "math": "segoepr.ttf", "code": "comic.ttf"}[kind]
    path = FONTS / name
    return ImageFont.truetype(str(path if path.exists() else FONTS / "arial.ttf"), size)


def _shape(text: str, kind: str) -> str:
    if kind != "ar":
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


def _render(text: str, kind: str, rng: random.Random) -> np.ndarray:
    lines = text.split("\n")
    size = 120 if kind == "ar" else 90
    page = Image.new("RGB", (1500, 220 + 190 * len(lines)), (245, 242, 232))             # paper
    draw = ImageDraw.Draw(page)
    for i, line in enumerate(lines):
        shaped = _shape(line, kind)
        font = _font(kind, size)
        w = draw.textlength(shaped, font=font)
        x = 1500 - 100 - w if kind == "ar" else 100
        draw.text((x, 90 + i * 190), shaped, fill=(25, 30, 70), font=font)
    img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    # a phone photo: the page is slightly rotated, seen in perspective, unevenly lit, a little blurred and noisy
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4, 4), 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=(70, 70, 70))
    j = lambda: rng.randint(0, 40)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, np.float32([[j(), j()], [w - j(), j()], [w - j(), h - j()], [j(), h - j()]])), (w, h), borderValue=(70, 70, 70))
    shade = np.tile(np.linspace(rng.uniform(0.75, 0.9), 1.0, w, dtype=np.float32), (h, 1))[:, :, None]
    img = np.clip(img * shade, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    rows = []
    n = 0
    for kind, texts in SAMPLES.items():
        for text in texts:
            n += 1
            name = f"{kind}_{n:02d}.jpg"
            cv2.imencode(".jpg", _render(text, kind, rng), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(OUT / name))
            rows.append([name, kind, text.replace("\n", "\\n")])
    with open(GT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image", "type", "text"])
        w.writerows(rows)
    print(f"wrote {len(rows)} DEMO images (not real handwriting) to {OUT} and {GT}")


if __name__ == "__main__":
    main()

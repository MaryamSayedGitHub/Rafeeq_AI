"""Create data/ground_truth.csv from the images in data/images  (python -m rafeeq.snap.init_gt), then fill the `text` column yourself.

Name your photos with the type as prefix:  ar_01.jpg  en_02.jpg  math_03.jpg  code_04.jpg
Write multi-line text with a literal \\n between lines. Save the CSV as UTF-8.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR, OUT = ROOT / "data" / "images", ROOT / "data" / "ground_truth.csv"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if OUT.exists():
    raise SystemExit(f"{OUT} already exists, not overwriting.")
files = sorted(p.name for p in IMG_DIR.iterdir() if p.suffix.lower() in EXT)
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["image", "type", "text"])
    for name in files:
        w.writerow([name, name.split("_")[0] if "_" in name else "", ""])
print(f"wrote {OUT} with {len(files)} rows. Fill the 'text' column by hand.")

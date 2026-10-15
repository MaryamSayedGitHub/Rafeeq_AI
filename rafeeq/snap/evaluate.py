"""Measure CER / WER on the 20-photo test set, with and without preprocessing, per reader.

    python -m rafeeq.snap.evaluate --readers easyocr gemini --presets raw pre pre_gray
    python -m rafeeq.snap.evaluate --readers gemini --presets raw only_perspective only_deskew only_denoise only_threshold pre
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import pandas as pd

from .metrics import cer, wer
from .pipeline import run_pipeline
from .preprocess import imread
from .readers import get_reader

ROOT = Path(__file__).resolve().parents[2]


def load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    df["text"] = df["text"].astype(str).str.replace("\\n", "\n", regex=False)
    return df[df["text"].str.strip() != ""]


def evaluate_cer(gt: pd.DataFrame, images: Path, readers: list[str], presets: list[str], line_method: str = "projection", debug: bool = False,
                 unify_alef: bool = False, out_dir: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """The CER / WER of every reader x preset x image (one row each). Shared by this command and report.py."""
    out_dir = out_dir or ROOT / "results"
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)
    norm = dict(unify_alef=unify_alef)
    rows = []
    for rname in readers:
        reader = get_reader(rname)
        for preset in presets:
            for _, r in gt.iterrows():
                img = imread(Path(images) / r["image"])
                t0 = time.time()
                res = run_pipeline(img, reader, preset, line_method)
                rows.append({
                    "reader": rname, "preset": preset, "image": r["image"], "type": r["type"],
                    "cer": cer(r["text"], res["text"], **norm), "wer": wer(r["text"], res["text"], **norm),
                    "n_lines": len(res["boxes"]), "sec": round(time.time() - t0, 2),
                    "truth": r["text"], "hyp": res["text"],
                })
                if debug and res["boxes"]:
                    dbg = cv2.cvtColor(res["prepped"].reader, cv2.COLOR_GRAY2BGR) if res["prepped"].reader.ndim == 2 else res["prepped"].reader.copy()
                    for x, y, w, h in res["boxes"]:
                        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.imencode(".jpg", dbg)[1].tofile(str(out_dir / "debug" / f"{rname}_{preset}_{Path(r['image']).stem}.jpg"))
                if verbose:
                    print(f"{rname:9s} {preset:16s} {r['image']:24s} CER={rows[-1]['cer']:.3f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(ROOT / "data" / "ground_truth.csv"))
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    ap.add_argument("--readers", nargs="+", default=["easyocr"])
    ap.add_argument("--presets", nargs="+", default=["raw", "pre", "pre_gray"])
    ap.add_argument("--line-method", default="projection", choices=["projection", "contours"])
    ap.add_argument("--debug", action="store_true", help="save the detected line boxes for every image")
    ap.add_argument("--unify-alef", action="store_true", help="count أ إ آ as ا")
    a = ap.parse_args()
    os.environ.setdefault("SNAP_MARK_UNCERTAIN", "0")          # CER on the plain transcription prompt

    out_dir = ROOT / "results"
    df = evaluate_cer(load_gt(Path(a.gt)), Path(a.images), a.readers, a.presets, a.line_method, a.debug, a.unify_alef, out_dir)
    df.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    summary = df.pivot_table(index=["reader", "preset"], columns="type", values="cer", aggfunc="mean")
    summary["ALL"] = df.groupby(["reader", "preset"])["cer"].mean()
    summary = summary.round(3)
    summary.to_csv(out_dir / "summary_cer.csv", encoding="utf-8-sig")
    print("\nMean CER (lower is better)\n", summary.to_string())

    base = df[df["preset"] == "raw"].set_index(["reader", "image"])["cer"]
    for preset in a.presets:
        if preset == "raw" or base.empty:
            continue
        cur = df[df["preset"] == preset].set_index(["reader", "image"])["cer"]
        delta = (cur - base).dropna().groupby(level=0).mean().round(3)
        print(f"\nPreprocessing effect, '{preset}' minus 'raw' (negative = preprocessing helps):\n{delta.to_string()}")


if __name__ == "__main__":
    main()

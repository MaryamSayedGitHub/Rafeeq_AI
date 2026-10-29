"""Smart montage (M5). Everything here is MEASURED on a lecture we build ourselves, whose speech and pauses are known
exactly, so the numbers are repeatable and need no model or internet. The two live metrics (caption WER, chapter
position) need real recordings with hand-written labels: put them in evals/data/montage/ (see its README)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from evals.core import ROOT, Metric, complement, length, mean_abs_offset, overlap, wer
from rafeeq.teacher import montage as M

NAME = "montage"
DESCRIPTION = "Pause cutting, timeline consistency, GIF overlay, speed (synthetic lecture); caption WER + chapter position (your labelled recordings)"

SPEECH = [(0.0, 5.0), (9.0, 14.0), (19.0, 22.0), (26.0, 30.0)]        # what we say; the gaps between are silence
TOTAL = 30.0
GAPS = complement(SPEECH, TOTAL)
DATA = ROOT / "evals" / "data" / "montage"


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True, capture_output=True)


def _lecture(path: Path) -> Path:
    gate = "+".join(f"between(t,{a},{b})" for a, b in SPEECH)
    _ff("-f", "lavfi", "-i", f"color=c=blue:s=320x180:d={TOTAL}:r=15", "-f", "lavfi", "-i", f"sine=f=300:d={TOTAL}",
        "-filter_complex", f"[1:a]volume='if({gate},1,0)':eval=frame[a]", "-map", "0:v", "-map", "[a]", "-shortest",
        "-pix_fmt", "yuv420p", str(path))
    return path


def _gif(path: Path) -> Path:
    _ff("-f", "lavfi", "-i", "testsrc=s=96x64:d=1:r=10", str(path))
    return path


def _frame(video: Path, t: float) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(video), "-frames:v", "1", "-vf", "scale=64:36,format=gray",
                           "-f", "rawvideo", "-"], capture_output=True, check=True).stdout


def _diff(a: bytes, b: bytes) -> float:
    n = min(len(a), len(b))
    return sum(abs(x - y) for x, y in zip(a[:n], b[:n])) / max(1, n)


def run(ctx: dict) -> list[Metric]:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return [Metric("synthetic_suite", None, note="ffmpeg is not installed")]
    out: list[Metric] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = _lecture(tmp / "lecture.mp4")

        # ---- 1. which parts does the pause cutter keep?
        keep = M.keep_segments(M.detect_silences(src), TOTAL)
        removed = complement(keep, TOTAL)
        achievable = sum(max(0.0, (b - a) - 2 * M.PAD) for a, b in GAPS)
        out.append(Metric("speech_kept", overlap(keep, SPEECH) / length(SPEECH) * 100, "%", gate=99.0,
                          note="share of the spoken seconds that survive the cut (must be ~100)"))
        out.append(Metric("silence_removed", min(100.0, overlap(removed, GAPS) / achievable * 100), "%", gate=90.0,
                          note=f"share of the removable pause time that was cut (the {M.PAD}s padding around speech is kept on purpose)"))
        out.append(Metric("wrongly_cut_seconds", length(removed) - overlap(removed, GAPS), "s", higher_is_better=False, gate=0.3,
                          note="seconds removed that were not silence"))

        # ---- 2. the whole team, with everything on: length, sound, speed
        gif = _gif(tmp / "party.gif")
        stk = [{"start": 20.0, "end": 21.5, "x": .8, "y": .75, "w": .3, "src": {"giphy": "https://media2.giphy.com/media/x/g.gif"}}]
        opts = M.clean_options({"cut_silence": True, "intro": True, "outro": True, "title": "Eval", "stickers": stk,
                                "captions": True, "segments": [{"start": 20.2, "end": 21.4, "text": "marker"}]})
        work = tmp / "w"; work.mkdir()
        t0 = time.time()
        res = M.run(src, work, opts, fetch=lambda url, dst: (dst.write_bytes(gif.read_bytes()), dst)[1])
        elapsed = time.time() - t0
        expected = sum(b - a for a, b in keep) + 2 * M.CARD_SECONDS
        info = M.probe(res["file"])
        out.append(Metric("length_error", abs(info["seconds"] - expected), "s", higher_is_better=False, gate=0.6,
                          note=f"final length {info['seconds']:.1f}s vs {expected:.1f}s expected (kept speech + 2 cards)"))
        out.append(Metric("sound_survives", 100.0 if info["audio"] else 0.0, "%", gate=100.0))
        out.append(Metric("seconds_per_video_minute", elapsed / (TOTAL / 60), "s", higher_is_better=False, gate=120.0,
                          note="processing time for one minute of video on THIS machine (cut + cards + GIF + captions)"))
        out.append(Metric("failed_steps", float(sum(1 for n in res["notes"] if "failed" in n)), "", higher_is_better=False, gate=0.0,
                          note="notes the team raised: " + (", ".join(res["notes"]) or "none")))

        # ---- 3. do captions and the GIF land where the lecturer put them, AFTER the cut?
        removed_before = sum(min(b, 20.2) - a for a, b in removed if a < 20.2)
        expected_start = 20.2 - removed_before + M.CARD_SECONDS            # moved by the cut, then by the intro card
        first = next((ln for ln in res["srt"].splitlines() if "-->" in ln), "")
        h, m_, rest = first.split(" --> ")[0].split(":") if first else ("0", "0", "0")
        got = int(h) * 3600 + int(m_) * 60 + float(rest.replace(",", "."))
        out.append(Metric("caption_time_error", abs(got - expected_start), "s", higher_is_better=False, gate=0.2,
                          note="a caption placed on the ORIGINAL timeline vs where it must be in the cut video"))
        (tmp / "p").mkdir()
        plain = M.run(src, tmp / "p", M.clean_options({"cut_silence": True, "intro": True, "outro": True, "title": "Eval"}))
        win_start = 20.0 - sum(min(b, 20.0) - a for a, b in removed if a < 20.0) + M.CARD_SECONDS
        inside = _diff(_frame(res["file"], win_start + 0.7), _frame(plain["file"], win_start + 0.7))
        outside = _diff(_frame(res["file"], win_start + 3.0), _frame(plain["file"], win_start + 3.0))
        out.append(Metric("gif_visible_in_window", inside, "", gate=1.0, note="pixel difference vs the same frame without the GIF (0-255)"))
        out.append(Metric("gif_absent_outside_window", outside, "", higher_is_better=False, gate=0.6, note="must stay near 0"))

    out += _live(ctx)
    return out


def _live(ctx: dict) -> list[Metric]:
    """Real recordings with labels: evals/data/montage/<name>.mp4 + <name>.json {"transcript": "...", "chapter_starts": [..]}."""
    pairs = [(v, v.with_suffix(".json")) for v in sorted(DATA.glob("*.mp4")) if v.with_suffix(".json").exists()]
    if not pairs:
        return [Metric("caption_wer", None, note="no labelled recording in evals/data/montage/ (see its README)"),
                Metric("chapter_position_error", None, note="no labelled recording in evals/data/montage/")]
    if not os.environ.get("GOOGLE_API_KEY") or not ctx.get("live"):
        why = "run with --live and GOOGLE_API_KEY (it calls Gemini)"
        return [Metric("caption_wer", None, note=why), Metric("chapter_position_error", None, note=why)]
    wers, offs = [], []
    for video, label_file in pairs:
        label = json.loads(label_file.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            got = M.transcribe_for_editor(video, Path(tmp))
        if label.get("transcript"):
            wers.append(wer(label["transcript"], " ".join(s["text"] for s in got["segments"])))
        o = mean_abs_offset(label.get("chapter_starts", []), [c["start"] for c in got["chapters"]])
        if o is not None:
            offs.append(o)
    return [Metric("caption_wer", (sum(wers) / len(wers) * 100) if wers else None, "%", higher_is_better=False, gate=25.0,
                   note=f"word error rate over {len(wers)} recording(s); lower is better"),
            Metric("chapter_position_error", (sum(offs) / len(offs)) if offs else None, "s", higher_is_better=False, gate=15.0,
                   note="mean distance between your chapter starts and the nearest suggested one")]

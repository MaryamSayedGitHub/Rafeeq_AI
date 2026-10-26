"""Smart montage (M5): turn a lecturer's raw recording into a finished lesson video.

The lecturer picks a video from the library and ticks what they want. Every step is optional:

    1. cut_silence   ffmpeg `silencedetect` finds the long pauses and they are cut out (no AI, no cost).
    2. captions      Gemini listens to the audio and returns timed sentences -> an .srt (optionally burned in).
    3. chapters      an LLM reads the timed text and names 3 to 8 chapters with start times.
    4. intro/outro   a title card (teacher name + lesson title) before and a closing card after.
    5. music         a library audio file under the voice, low volume, fading out at the end.

The work is done by a team of agents (see montage_team/): supervisor, analyst, cutter, transcriber, chapters,
designer, assembler and QA. The lecturer assistant's orchestrator can start it too (montage_agent).

The result is a NEW library video (origin "edited", parent = the original, which is never touched). Captions
(srt text) and chapters are kept in the asset's settings["montage"], so the page can show and download them.
The job runs in a background thread (a 10 minute video takes a few minutes); the page polls `view()`.

Nothing here invents words: captions come from the audio, chapter titles from the captions.
"""
import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

from rafeeq.teacher import assets as asset_lib
from rafeeq.teacher.media_tools import MediaError, ffmpeg_path

log = logging.getLogger(__name__)

NOISE_DB = -32            # quieter than this counts as silence
MIN_SILENCE = 0.8         # seconds; shorter pauses are kept (they are natural)
PAD = 0.2                 # seconds of silence kept around speech, so words are not clipped
MIN_KEEP = 0.3            # a piece of speech shorter than this is dropped
MAX_SEGMENTS = 120        # more pieces than this are merged (the ffmpeg command stays small)
MAX_SECONDS = 30 * 60     # longest video accepted
CARD_SECONDS = 3.0
FFMPEG_TIMEOUT = 1800
MAX_TITLE = 120
MAX_CHAPTERS = 8
MAX_STICKERS = 12         # animated pictures (GIF / sticker) on one video
STICKER_MIN, STICKER_MAX_SECONDS = 0.5, 30.0
STICKER_MAX_BYTES = 8 * 1024 * 1024
GIPHY_HOST = re.compile(r"^(media\d*|i)\.giphy\.com$")      # the only hosts a picture may be downloaded from
STEPS = ("queued", "cutting", "transcribing", "chapters", "assembling", "saving")

_running: set[str] = set()
_slot = threading.Semaphore(1)       # one montage at a time per process: ffmpeg uses all the cores


# ------------------------------------------------------------------ ffmpeg helpers
def _exe(name: str = "ffmpeg") -> str:
    path = ffmpeg_path() if name == "ffmpeg" else None
    if name != "ffmpeg":
        import shutil
        path = shutil.which(name)
    if path is None:
        raise MediaError("ffmpeg is not installed on the server, so video editing is unavailable. "
                         "Install ffmpeg (https://ffmpeg.org) and restart Rafeeq.")
    return path


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = FFMPEG_TIMEOUT) -> subprocess.CompletedProcess:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd) if cwd else None)
    except subprocess.TimeoutExpired:
        raise MediaError("The montage took too long and was stopped") from None
    if done.returncode != 0:
        raise MediaError("ffmpeg failed: " + (done.stderr or "unknown error").strip()[-300:])
    return done


def probe(path: Path) -> dict:
    """{'seconds', 'width', 'height', 'audio'} of a video."""
    out = _run([_exe("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                "-of", "json", str(path)], timeout=60).stdout
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"seconds": float(data["format"]["duration"]), "width": int(video.get("width") or 1280),
            "height": int(video.get("height") or 720),
            "audio": any(s.get("codec_type") == "audio" for s in data.get("streams", []))}


# ------------------------------------------------------------------ 1. silence
def detect_silences(path: Path) -> list[tuple[float, float]]:
    """[(start, end)] of every pause of MIN_SILENCE seconds or more."""
    done = subprocess.run([_exe(), "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af",
                           f"silencedetect=noise={NOISE_DB}dB:d={MIN_SILENCE}", "-f", "null", "-"],
                          capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if done.returncode != 0:
        raise MediaError("ffmpeg could not read the audio: " + (done.stderr or "").strip()[-200:])
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", done.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", done.stderr)]
    out = []
    for i, s in enumerate(starts):
        out.append((max(0.0, s), ends[i] if i < len(ends) else float("inf")))   # a silence that runs to the end
    return out


def keep_segments(silences: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    """The pieces to keep: everything between the silences, with PAD of silence left on each side."""
    pieces, cursor = [], 0.0
    for s, e in sorted(silences):
        s = min(s, total)
        if s - cursor > 0:
            pieces.append((cursor, min(total, s + PAD)))
        cursor = min(total, e - PAD) if e != float("inf") else total
    if cursor < total:
        pieces.append((cursor, total))
    merged: list[list[float]] = []
    for a, b in pieces:
        a = max(0.0, a)
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    kept = [(a, b) for a, b in merged if b - a >= MIN_KEEP]
    while len(kept) > MAX_SEGMENTS:           # join the two pieces with the smallest gap between them
        gaps = [kept[i + 1][0] - kept[i][1] for i in range(len(kept) - 1)]
        i = gaps.index(min(gaps))
        kept[i:i + 2] = [(kept[i][0], kept[i + 1][1])]
    return kept


def _filter_arg(cmd_head: list[str], graph: str, work: Path) -> list[list[str]]:
    """The graph goes through a file: a long graph does not fit on a Windows command line. Newer ffmpeg
    spells the option '-/filter_complex file', older ones '-filter_complex_script file'."""
    script = work / "graph.txt"
    script.write_text(graph, encoding="utf-8")
    return [cmd_head + ["-filter_complex_script", str(script)], cmd_head + ["-/filter_complex", str(script)]]


def _run_graph(head: list[str], graph: str, tail: list[str], work: Path) -> None:
    last = None
    for cmd in _filter_arg(head, graph, work):
        try:
            _run(cmd + tail)
            return
        except MediaError as e:
            last = e
            if "Unrecognized option" not in str(e) and "Option not found" not in str(e):
                raise
    raise last                                  # type: ignore[misc]


def render_cut(src: Path, segments: list[tuple[float, float]], dst: Path, has_audio: bool, work: Path) -> Path:
    parts, labels = [], ""
    for i, (a, b) in enumerate(segments):
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
        if has_audio:
            parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
            labels += f"[v{i}][a{i}]"
        else:
            labels += f"[v{i}]"
    n = len(segments)
    parts.append(f"{labels}concat=n={n}:v=1:a={1 if has_audio else 0}" + ("[v][a]" if has_audio else "[v]"))
    maps = ["-map", "[v]"] + (["-map", "[a]"] if has_audio else [])
    tail = maps + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p"] + \
           (["-c:a", "aac", "-b:a", "128k"] if has_audio else []) + ["-movflags", "+faststart", str(dst)]
    _run_graph([_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)], ";".join(parts), tail, work)
    return dst


# ------------------------------------------------------------------ 2. captions
def extract_audio(src: Path, dst: Path) -> Path:
    """Mono 16 kHz mp3 at a low bitrate: small enough to send inline (10 minutes is about 2.5 MB)."""
    _run([_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
          "-b:a", "32k", str(dst)])
    return dst


TRANSCRIBE_PROMPT = (
    "Transcribe this lecture recording exactly as spoken, in the language spoken (Arabic, including Egyptian "
    "dialect, English, or a mix: keep each word in the language it was said in). Do not translate, summarise or "
    "add words. Split it into caption lines of one short sentence or phrase, 2 to 7 seconds each, in time order. "
    'Reply with JSON only: {"segments": [{"start": <seconds>, "end": <seconds>, "text": "..."}]}. '
    "start and end are seconds from the beginning of the audio, as numbers.")


def _gemini_json(parts: list[dict], model: str) -> dict:
    import httpx

    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise MediaError("Captions need GOOGLE_API_KEY (Gemini) in .env")
    res = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                     headers={"x-goog-api-key": key},
                     json={"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json",
                                                                                 "temperature": 0}},
                     timeout=300)
    if res.status_code == 429:
        raise MediaError("Gemini says too many requests; wait a minute and try again")
    res.raise_for_status()
    text = "".join(p.get("text", "") for p in res.json()["candidates"][0]["content"]["parts"])
    return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip())


def gemini_transcribe(audio: Path) -> list[dict]:
    model = os.getenv("MONTAGE_STT_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    try:
        data = _gemini_json([{"text": TRANSCRIBE_PROMPT},
                             {"inlineData": {"mimeType": "audio/mp3", "data": base64.b64encode(audio.read_bytes()).decode()}}], model)
    except MediaError:
        raise
    except Exception as e:                                           # noqa: BLE001
        raise MediaError(f"The captions could not be made ({type(e).__name__})") from None
    return data.get("segments", []) if isinstance(data, dict) else []


def clean_segments(raw: list[dict], total: float) -> list[dict]:
    """Keep only well-formed lines, in order, inside the video. Never raises."""
    out, last_end = [], 0.0
    for r in raw if isinstance(raw, list) else []:
        try:
            start, end, text = float(r["start"]), float(r["end"]), " ".join(str(r["text"]).split())
        except (KeyError, TypeError, ValueError):
            continue
        start = max(start, last_end)
        end = min(end, total)
        if not text or end - start < 0.2 or start >= total:
            continue
        out.append({"start": round(start, 2), "end": round(end, 2), "text": text[:300]})
        last_end = end
    return out


def _stamp(sec: float) -> str:
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_srt(segments: list[dict], offset: float = 0.0) -> str:
    return "\n".join(f"{i}\n{_stamp(s['start'] + offset)} --> {_stamp(s['end'] + offset)}\n{s['text']}\n"
                     for i, s in enumerate(segments, 1))


# ------------------------------------------------------------------ 3. chapters
CHAPTER_SYSTEM = (
    "You split a lecture into chapters for students. You get the lecture's caption lines, each with its start "
    "time in seconds. Return 3 to 8 chapters in time order. The first chapter starts at 0. Each title is 2 to 7 "
    "words, in the language of the lecture, and uses only words and ideas that appear in the captions. "
    'Reply with JSON {"chapters": [{"title": "...", "start": <seconds>}]}.')


def clean_chapters(raw, total: float) -> list[dict]:
    out, last = [], -1.0
    for c in raw if isinstance(raw, list) else []:
        try:
            title, start = " ".join(str(c["title"]).split())[:80], float(c["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if not title or start <= last or start >= total:
            continue
        out.append({"title": title, "start": round(start, 1)})
        last = start
    if len(out) < 2:
        return []
    out[0]["start"] = 0.0
    return out[:MAX_CHAPTERS]


def make_chapters(segments: list[dict], total: float, propose: Callable | None) -> list[dict]:
    if not segments or propose is None:
        return []
    lines = "\n".join(f"{s['start']:.0f}s: {s['text']}" for s in segments)[:24000]
    try:
        out = propose(CHAPTER_SYSTEM, lines) or {}
    except Exception:                                                # noqa: BLE001
        log.exception("montage: chapters failed")
        return []
    return clean_chapters(out.get("chapters"), total)


# ------------------------------------------------------------------ 4. cards, 5. music, final pass
def _shape(text: str) -> str:
    """Arabic needs joining and right-to-left order when drawn with Pillow; harmless for English."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:                                                # noqa: BLE001
        return text


def _font(size: int, bold: bool):
    from PIL import ImageFont

    try:
        from rafeeq.tutor.export import _font_files
        files = _font_files()
        return ImageFont.truetype(str(files.get("B" if bold else "") or files[""]), size)
    except Exception:                                                # noqa: BLE001
        return ImageFont.load_default()


def make_card(title: str, subtitle: str, size: tuple[int, int], dst: Path, accent: str = "#A0466F") -> Path:
    """A plain title card: dark background, accent bar, the lesson title and a smaller line under it."""
    from PIL import Image, ImageDraw

    w, h = size
    img = Image.new("RGB", (w, h), "#1F2A2E")
    d = ImageDraw.Draw(img)
    d.rectangle([w * 0.1, h * 0.36, w * 0.1 + 12, h * 0.64], fill=accent)

    def wrap(text: str, font, width: int) -> list[str]:
        words, lines, cur = text.split(), [], ""
        for word in words:
            trial = (cur + " " + word).strip()
            if d.textlength(_shape(trial), font=font) <= width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        return lines + ([cur] if cur else [])

    big, small = _font(max(28, h // 11), True), _font(max(20, h // 24), False)
    y = h * 0.38
    for line in wrap(title, big, int(w * 0.72))[:3]:
        shaped = _shape(line)
        x = w * 0.1 + 48
        d.text((x, y), shaped, font=big, fill="#F6F1E7")
        y += big.size * 1.25 if hasattr(big, "size") else 70
    if subtitle:
        d.text((w * 0.1 + 48, y + 12), _shape(subtitle[:120]), font=small, fill="#E8A9C4")
    img.save(dst)
    return dst


def assemble(main: Path, dst: Path, work: Path, *, width: int, height: int, has_audio: bool, intro: Path | None,
             outro: Path | None, music: Path | None, burn_srt: Path | None, total: float,
             stickers: list[dict] | None = None) -> Path:
    """One last pass: intro + lesson + outro joined, music mixed under the voice, captions burned in.
    `stickers` ([{path,start,end,x,y,w}], times on the lesson part) are laid over the lesson, under the captions."""
    inputs, parts, v_labels, a_labels = [], [], [], []
    idx = 0
    norm_v = f"fps=30,scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
    cards, main_idx = [], 0
    if intro:
        cards.append(("intro", intro))
    cards.append(("main", main))
    if outro:
        cards.append(("outro", outro))
    for name, path in cards:
        if name == "main":
            inputs += ["-i", str(path.resolve())]
            main_idx = idx
            sub = f",subtitles=captions.srt:force_style='FontName=Arial,FontSize=20,Outline=2,MarginV=28'" if burn_srt else ""
            if stickers:                                            # the pictures go on first, the captions on top of them
                parts.append(f"[{idx}:v]{norm_v}[m{idx}]")
            else:
                parts.append(f"[{idx}:v]{norm_v}{sub}[v{idx}]")
            if has_audio:
                parts.append(f"[{idx}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{idx}]")
            else:
                parts.append(f"anullsrc=r=44100:cl=stereo,atrim=duration={total:.3f}[a{idx}]")
        else:
            inputs += ["-loop", "1", "-t", f"{CARD_SECONDS}", "-i", str(path.resolve())]
            fade_out = CARD_SECONDS - 0.5
            parts.append(f"[{idx}:v]{norm_v},fade=t=in:st=0:d=0.5,fade=t=out:st={fade_out}:d=0.5[v{idx}]")
            parts.append(f"anullsrc=r=44100:cl=stereo,atrim=duration={CARD_SECONDS}[a{idx}]")
        v_labels.append(f"[v{idx}]")
        a_labels.append(f"[a{idx}]")
        idx += 1
    n = len(cards)
    parts.append("".join(f"{v}{a}" for v, a in zip(v_labels, a_labels)) + f"concat=n={n}:v=1:a=1[vout][aout]")
    final_a = "[aout]"
    if music:
        inputs += ["-stream_loop", "-1", "-i", str(music.resolve())]
        full = total + CARD_SECONDS * (n - 1)
        parts.append(f"[{idx}:a]aresample=44100,aformat=channel_layouts=stereo,volume=0.12,"
                     f"afade=t=in:st=0:d=2,afade=t=out:st={max(0.0, full - 3):.2f}:d=3[mus]")
        parts.append("[aout][mus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[amix]")
        final_a = "[amix]"
    if stickers:
        n_in = sum(1 for x in inputs if x == "-i")
        cur = f"m{main_idx}"
        for k, st in enumerate(stickers):
            gif = st["path"].suffix.lower() == ".gif"
            dur = max(STICKER_MIN, st["end"] - st["start"])
            inputs += (["-ignore_loop", "0"] if gif else ["-loop", "1"]) + ["-t", f"{dur:.3f}", "-i", str(st["path"].resolve())]
            sw = max(16, int(width * st["w"]) // 2 * 2)
            sh = max(16, int(sw * st["ratio"]) // 2 * 2)
            x = int(min(max(0, width * st["x"] - sw / 2), width - sw))
            y = int(min(max(0, height * st["y"] - sh / 2), height - sh))
            lab = f"s{k}"
            parts.append(f"[{n_in + k}:v]format=rgba,scale={sw}:{sh},setpts=PTS-STARTPTS+{st['start']:.3f}/TB[{lab}]")
            nxt = f"o{k}"
            parts.append(f"[{cur}][{lab}]overlay=x={x}:y={y}:eof_action=pass:repeatlast=0:"
                         f"enable='between(t,{st['start']:.3f},{st['end']:.3f})'[{nxt}]")
            cur = nxt
        sub = f"subtitles=captions.srt:force_style='FontName=Arial,FontSize=20,Outline=2,MarginV=28'" if burn_srt else "null"
        parts.append(f"[{cur}]{sub}[v{main_idx}]")
    head = [_exe(), "-y", "-hide_banner", "-loglevel", "error"] + inputs
    tail = ["-map", "[vout]", "-map", final_a, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst.resolve())]
    last = None
    script = work / "final_graph.txt"
    script.write_text(";".join(parts), encoding="utf-8")
    for opt in ("-filter_complex_script", "-/filter_complex"):
        try:
            _run(head + [opt, str(script)] + tail, cwd=work)
            return dst
        except MediaError as e:
            last = e
            if "Unrecognized option" not in str(e) and "Option not found" not in str(e):
                raise
    raise last                                                      # type: ignore[misc]


# ------------------------------------------------------------------ the whole pipeline
def run(src: Path, work: Path, opts: dict, *, music_file: Path | None = None, transcribe: Callable | None = None,
        propose: Callable | None = None, progress: Callable[[str], None] = lambda step: None,
        sticker_files: dict | None = None, fetch: Callable | None = None) -> dict:
    """Do the montage of `src` in the folder `work`. The work is done by a small team of agents
    (rafeeq/teacher/montage_team): a supervisor plans, then analyst, cutter, transcriber, chapters, designer,
    assembler and a QA agent run in turn. Returns {'file': Path, 'srt', 'captions', 'chapters', 'cut', 'notes',
    'seconds', 'agents', 'plan'}.
    `transcribe(audio_path) -> [{'start','end','text'}]` and `propose(system, user) -> dict` are injectable (tests)."""
    from rafeeq.teacher.montage_team.graph import run_team

    return run_team(src, work, opts, music_file=music_file, transcribe=transcribe, propose=propose, progress=progress,
                    sticker_files=sticker_files, fetch=fetch)


# ------------------------------------------------------------------ the editor: decisions made on the ORIGINAL timeline
def clean_keep(raw, total: float | None = None) -> list[tuple[float, float]]:
    """The pieces of the original video to keep, from the editor. Sorted, clipped, merged, never raises."""
    pieces = []
    for r in raw if isinstance(raw, list) else []:
        try:
            a, b = float(r[0]), float(r[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        a = max(0.0, a)
        if total is not None:
            b = min(b, total)
        if b - a >= 0.1:
            pieces.append((a, b))
    merged: list[list[float]] = []
    for a, b in sorted(pieces):
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(round(a, 2), round(b, 2)) for a, b in merged][:MAX_SEGMENTS * 2]


def remap_time(t: float, keep: list[tuple[float, float]]) -> float:
    """A time on the original video -> the same moment on the cut video (a time inside a cut-out part
    becomes the start of the next kept piece)."""
    acc = 0.0
    for a, b in keep:
        if t < a:
            return acc
        if t <= b:
            return acc + (t - a)
        acc += b - a
    return acc


def remap_segments(segments: list[dict], keep: list[tuple[float, float]]) -> list[dict]:
    if not keep:
        return segments
    out = []
    for s in segments:
        a, b = remap_time(s["start"], keep), remap_time(s["end"], keep)
        if b - a >= 0.2:
            out.append({**s, "start": round(a, 2), "end": round(b, 2)})
    return out


def remap_chapters(chapters: list[dict], keep: list[tuple[float, float]]) -> list[dict]:
    if not keep:
        return chapters
    return [{**c, "start": remap_time(c["start"], keep)} for c in chapters]


# ------------------------------------------------------------------ animated pictures (GIFs / stickers)
def giphy_search(query: str, limit: int = 24) -> list[dict]:
    """Search Giphy (rating G). Needs GIPHY_API_KEY in .env. Returns [{id, title, thumb, url, w, h}]; an empty query
    gives the trending ones. `thumb` is small (for the grid), `url` is what is placed on the video."""
    import httpx

    key = os.environ.get("GIPHY_API_KEY", "").strip()
    if not key:
        raise MediaError("Searching GIFs needs GIPHY_API_KEY in .env (free key from developers.giphy.com)")
    q = " ".join(str(query or "").split())[:80]
    params = {"api_key": key, "limit": max(1, min(int(limit), 40)), "rating": "g", "bundle": "messaging_non_clips"}
    if q:
        params["q"] = q
    try:
        res = httpx.get("https://api.giphy.com/v1/gifs/" + ("search" if q else "trending"), params=params, timeout=15)
    except httpx.HTTPError:
        raise MediaError("Giphy could not be reached; check the internet connection") from None
    if res.status_code in (401, 403):
        raise MediaError("Giphy refused the key; check GIPHY_API_KEY")
    if res.status_code == 429:
        raise MediaError("Giphy says too many searches; wait a minute and try again")
    if res.status_code >= 400:
        raise MediaError(f"Giphy error ({res.status_code})")
    out = []
    for g in (res.json().get("data") or []):
        im = g.get("images") or {}
        full = im.get("downsized_medium") or im.get("downsized") or im.get("fixed_height") or {}
        thumb = im.get("fixed_width_small") or im.get("fixed_width") or full
        url, tu = full.get("url") or "", thumb.get("url") or ""
        try:
            w, h = int(full.get("width") or 0), int(full.get("height") or 0)
        except (TypeError, ValueError):
            w = h = 0
        if _giphy_ok(url) and _giphy_ok(tu) and w > 0 and h > 0:
            out.append({"id": str(g.get("id", ""))[:40], "title": str(g.get("title", ""))[:80], "thumb": tu, "url": url, "w": w, "h": h})
    return out


def _giphy_ok(url: str) -> bool:
    from urllib.parse import urlparse

    u = urlparse(str(url))
    return u.scheme == "https" and bool(GIPHY_HOST.match(u.hostname or "")) and u.port in (None, 443)


def clean_stickers(raw, total: float | None = None) -> list[dict]:
    """The pictures placed in the editor, on the ORIGINAL timeline. src is a Giphy picture ({'giphy': url}) or a
    library picture ({'asset_id': id}). x, y = the centre as a fraction of the video, w = width as a fraction."""
    out = []
    for r in raw[:MAX_STICKERS * 2] if isinstance(raw, list) else []:
        try:
            a, b = float(r["start"]), float(r["end"])
            x, y, w = float(r["x"]), float(r["y"]), float(r["w"])
            src = r.get("src") or {}
            giphy, asset_id = str(src.get("giphy") or ""), str(src.get("asset_id") or "")[:40]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        a = max(0.0, a)
        if total is not None:
            b = min(b, total)
        b = min(b, a + STICKER_MAX_SECONDS)
        if b - a < STICKER_MIN or not (giphy and _giphy_ok(giphy) or asset_id and not giphy):
            continue
        out.append({"start": round(a, 2), "end": round(b, 2), "x": round(min(1.0, max(0.0, x)), 3), "y": round(min(1.0, max(0.0, y)), 3),
                    "w": round(min(0.8, max(0.05, w)), 3), "src": {"giphy": giphy} if giphy else {"asset_id": asset_id}})
    return sorted(out, key=lambda s: s["start"])[:MAX_STICKERS]


def remap_stickers(stickers: list[dict], keep: list[tuple[float, float]]) -> list[dict]:
    """Move the pictures from the original timeline onto the cut one. A picture whose whole time was cut away is dropped."""
    if not keep:
        return stickers
    out = []
    for s in stickers:
        a, b = remap_time(s["start"], keep), remap_time(s["end"], keep)
        if b - a >= STICKER_MIN:
            out.append({**s, "start": round(a, 2), "end": round(b, 2)})
    return out


def picture_size(path: Path) -> tuple[int, int]:
    """(width, height) of a picture or GIF; a GIF has no duration, so this does not use probe()."""
    out = _run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                "-of", "json", str(path)], timeout=60).stdout
    st = (json.loads(out).get("streams") or [{}])[0]
    return int(st.get("width") or 0), int(st.get("height") or 0)


def fetch_giphy(url: str, dst: Path) -> Path:
    """Download one Giphy picture (only from giphy.com hosts, no redirects, at most 8 MB)."""
    import httpx

    if not _giphy_ok(url):
        raise MediaError("This picture is not from Giphy")
    try:
        with httpx.stream("GET", url, follow_redirects=False, timeout=30) as res:
            if res.status_code != 200:
                raise MediaError(f"Giphy picture not available ({res.status_code})")
            data = bytearray()
            for chunk in res.iter_bytes():
                data += chunk
                if len(data) > STICKER_MAX_BYTES:
                    raise MediaError("This GIF is too big")
    except httpx.HTTPError:
        raise MediaError("The GIF could not be downloaded") from None
    if not bytes(data[:6]).startswith((b"GIF8", b"\x89PNG", b"RIFF")):
        raise MediaError("This is not a picture")
    dst.write_bytes(bytes(data))
    return dst


def analyze(path: Path) -> dict:
    """What the editor needs to draw the timeline: length, size, sound and the long pauses (original timeline)."""
    info = probe(path)
    if info["seconds"] > MAX_SECONDS:
        raise MediaError(f"The video is longer than {MAX_SECONDS // 60} minutes")
    silences = detect_silences(path) if info["audio"] else []
    return {**info, "silences": [[round(s, 2), round(min(e, info["seconds"]), 2)] for s, e in silences],
            "min_silence": MIN_SILENCE, "pad": PAD}


def transcribe_for_editor(path: Path, work: Path, *, transcribe: Callable | None = None, propose: Callable | None = None) -> dict:
    """Captions (and suggested chapters) of the ORIGINAL video, for the editor to show and let the teacher fix."""
    info = probe(path)
    if not info["audio"]:
        raise MediaError("This video has no sound to write captions from")
    audio = extract_audio(path, work / "speech.mp3")
    segments = clean_segments((transcribe or gemini_transcribe)(audio), info["seconds"])
    if propose is None:
        try:
            from rafeeq.knowledge.topics.room_topics import default_propose
            propose = default_propose()
        except Exception:                                           # noqa: BLE001
            propose = None
    return {"segments": segments, "chapters": make_chapters(segments, info["seconds"], propose)}


# ------------------------------------------------------------------ the library job
def clean_options(raw: dict) -> dict:
    """Only the known options, as plain values (they come from a web form)."""
    return {
        "cut_silence": bool(raw.get("cut_silence")), "captions": bool(raw.get("captions")),
        "burn_captions": bool(raw.get("captions") and raw.get("burn_captions")),
        "chapters": bool(raw.get("chapters")), "intro": bool(raw.get("intro")), "outro": bool(raw.get("outro")),
        "title": " ".join(str(raw.get("title") or "").split())[:MAX_TITLE],
        "subtitle": " ".join(str(raw.get("subtitle") or "").split())[:MAX_TITLE],
        "outro_text": " ".join(str(raw.get("outro_text") or "").split())[:MAX_TITLE],
        "music": bool(raw.get("music_asset_id")), "music_asset_id": str(raw.get("music_asset_id") or "")[:40],
        **_editor_decisions(raw),
    }


def _editor_decisions(raw: dict) -> dict:
    """Decisions made in the editor (all optional). None means: the team decides, as before."""
    keep = clean_keep(raw.get("keep")) if raw.get("keep") is not None else None
    segments = None
    if raw.get("segments") is not None:
        segments = []
        for s in raw["segments"][:3000] if isinstance(raw["segments"], list) else []:
            try:
                segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": " ".join(str(s["text"]).split())[:300]})
            except (KeyError, TypeError, ValueError):
                continue
    chapter_list = None
    if raw.get("chapter_list") is not None:
        chapter_list = []
        for c in raw["chapter_list"][:30] if isinstance(raw["chapter_list"], list) else []:
            try:
                chapter_list.append({"title": " ".join(str(c["title"]).split())[:80], "start": float(c["start"])})
            except (KeyError, TypeError, ValueError):
                continue
    stickers = clean_stickers(raw.get("stickers")) if raw.get("stickers") is not None else None
    return {"keep": keep, "segments": segments, "chapter_list": chapter_list, "stickers": stickers}


def public_options(opts: dict) -> dict:
    """The options as kept in the asset settings: the long lists are replaced by their sizes."""
    out = {k: v for k, v in opts.items() if k not in ("keep", "segments", "chapter_list", "stickers")}
    for k in ("keep", "segments", "chapter_list", "stickers"):
        if opts.get(k) is not None:
            out[k + "_count"] = len(opts[k])
    return out


def start(store, owner: str, source: dict, raw_opts: dict, *, run_in_thread: bool = True, propose: Callable | None = None,
          transcribe: Callable | None = None) -> dict:
    """Create the new (pending) library video and start the work. The same source and options give the same asset."""
    opts = clean_options(raw_opts)
    if not any(opts[k] for k in ("cut_silence", "captions", "chapters", "intro", "outro", "music")) and not opts["keep"] and not opts["stickers"]:
        raise MediaError("Pick at least one thing to do")
    if opts["intro"] and not opts["title"]:
        raise MediaError("Write the lesson title for the intro card")
    path = asset_lib.asset_file(source)
    if path is None:
        raise MediaError("This video has no file yet")
    music_asset = None
    if opts["music"]:
        music_asset = asset_lib.get_asset(store, owner, opts["music_asset_id"])
        if music_asset is None or music_asset["kind"] != "audio" or asset_lib.asset_file(music_asset) is None:
            raise MediaError("Pick an audio file from your library for the music")
    sticker_files: dict[str, Path] = {}
    for st in opts["stickers"] or []:
        aid = st["src"].get("asset_id")
        if aid and aid not in sticker_files:
            pic = asset_lib.get_asset(store, owner, aid)
            f = asset_lib.asset_file(pic) if pic and pic["kind"] == "image" and pic["status"] == "ready" else None
            if f is None:
                raise MediaError("A picture you placed is not in your library any more")
            sticker_files[aid] = f
    key = "montage:" + __import__("hashlib").sha1(f"{source['asset_id']}|{json.dumps(opts, sort_keys=True)}".encode()).hexdigest()[:24]
    asset = asset_lib.register_asset(
        store, owner, kind="video", title=(opts["title"] or source["title"] + " (montage)")[:200], origin="edited",
        status="pending", parent_asset_id=source["asset_id"], version=source["version"] + 1, room_id=source.get("room_id", ""),
        session_id=source.get("session_id", ""), settings_used={"montage": {"options": public_options(opts), "step": "queued"}}, idempotency_key=key)
    if asset["status"] == "ready" or asset["asset_id"] in _running:
        return asset
    if asset["status"] == "failed":
        asset = asset_lib.update_asset(store, owner, asset["asset_id"], status="pending", error="",
                                       settings={"montage": {"options": public_options(opts), "step": "queued"}})
    asset_id = asset["asset_id"]
    _running.add(asset_id)

    def step(name: str) -> None:
        cur = asset_lib.get_asset(store, owner, asset_id)
        if cur:
            asset_lib.update_asset(store, owner, asset_id, settings={"montage": {**cur["settings"].get("montage", {}), "step": name}})

    def work() -> None:
        try:
            with _slot, tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                if propose is None:
                    try:
                        from rafeeq.knowledge.topics.room_topics import default_propose
                        prop = default_propose()
                    except Exception:                               # noqa: BLE001
                        prop = None
                else:
                    prop = propose
                out = run(path, tmp, opts, music_file=asset_lib.asset_file(music_asset) if music_asset else None,
                          transcribe=transcribe, propose=prop, progress=step, sticker_files=sticker_files)
                step("saving")
                root, ref = asset_lib.save_file(owner, asset_id, out["file"].read_bytes(), "montage.mp4")
                cur = asset_lib.get_asset(store, owner, asset_id)
                info = {**cur["settings"].get("montage", {}), "step": "done", "srt": out["srt"], "captions": out["captions"],
                        "chapters": out["chapters"], "cut": out["cut"], "notes": out["notes"], "seconds": out["seconds"],
                        "agents": out.get("agents", []), "plan": out.get("plan", [])}
                asset_lib.update_asset(store, owner, asset_id, status="ready", storage_root=root, storage_ref=ref,
                                       settings={"montage": info})
        except Exception as e:                                      # noqa: BLE001 - never leave a job pending forever
            log.exception("montage failed")
            msg = str(e) if isinstance(e, MediaError) else f"The montage failed ({type(e).__name__})"
            asset_lib.update_asset(store, owner, asset_id, status="failed", error=msg[:300])
        finally:
            _running.discard(asset_id)

    if run_in_thread:
        threading.Thread(target=work, name=f"montage-{asset_id}", daemon=True).start()
    else:
        work()
    return asset_lib.get_asset(store, owner, asset_id) or asset


def view(store, owner: str, asset_id: str) -> dict | None:
    """The asset for the page. A job that was running when the server restarted is marked failed (run it again)."""
    a = asset_lib.get_asset(store, owner, asset_id)
    if a is None or "montage" not in (a.get("settings") or {}):
        return None
    if a["status"] == "pending" and asset_id not in _running:
        a = asset_lib.update_asset(store, owner, asset_id, status="failed", error="The server restarted while this was running. Run it again.")
    return a

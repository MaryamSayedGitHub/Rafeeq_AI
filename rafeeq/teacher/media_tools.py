"""Independent media editing for the Visual & Media Studio. Every edit makes a NEW file; the caller
saves it as a new version of the asset (the original is never overwritten).

- Images: OpenCV (already a dependency): crop, rotate, flip, resize, brightness, contrast, grayscale.
- Video / audio: the `ffmpeg` program. It is not a Python package: when it is not installed, these
  functions raise MediaError("ffmpeg is not installed ...") and the page says so.
- Speech: Gemini TTS (same GOOGLE_API_KEY as the text model) first, edge-tts (Microsoft's online voices) as the
  fallback; both need an internet connection. TTS_PROVIDER=edge|gemini forces one.
"""

import asyncio
import base64
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".webm", ".mov"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg"}
MAX_UPLOAD_BYTES = {"image": 15 * 1024 * 1024, "video": 1024 * 1024 * 1024, "audio": 40 * 1024 * 1024}
VOICES = {
    "ar-EG-SalmaNeural": "Arabic (Egypt), female", "ar-EG-ShakirNeural": "Arabic (Egypt), male",
    "ar-SA-ZariyahNeural": "Arabic (Saudi), female", "ar-SA-HamedNeural": "Arabic (Saudi), male",
    "en-US-JennyNeural": "English (US), female", "en-US-GuyNeural": "English (US), male",
    "en-GB-SoniaNeural": "English (UK), female",
}
MAX_SPEECH_CHARS = 3000
FFMPEG_TIMEOUT = 300


class MediaError(ValueError):
    """The edit cannot be done (bad input, tool missing, tool failed)."""


def kind_for(filename: str) -> str:
    """'image' | 'video' | 'audio' | '' from the file extension."""
    ext = Path(filename or "").suffix.lower()
    return "image" if ext in IMAGE_EXTS else "video" if ext in VIDEO_EXTS else "audio" if ext in AUDIO_EXTS else ""


def _num(ops: dict, key: str, lo: float, hi: float, default: float) -> float:
    value = ops.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise MediaError(f"'{key}' must be a number") from None
    if not lo <= value <= hi:
        raise MediaError(f"'{key}' must be between {lo:g} and {hi:g}")
    return value


# ------------------------------------------------------------------ images
def edit_image(data: bytes, ext: str, ops: dict) -> bytes:
    """ops (all optional): crop {x,y,w,h} in percent of the picture, rotate 0/90/180/270,
    flip ''|'h'|'v'|'both', width in pixels (keeps the proportions), brightness -100..100,
    contrast 0.5..2, grayscale true/false. Order: crop, rotate, flip, resize, colour."""
    import cv2
    import numpy as np

    ext = ext.lower()
    if ext not in IMAGE_EXTS:
        raise MediaError("Unsupported image type")
    if ext == ".gif":
        raise MediaError("An animated GIF cannot be edited as a picture (it would lose its animation). "
                         "Edit the source video and convert it to a GIF again.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise MediaError("This is not a readable image")

    crop = ops.get("crop")
    if crop:
        x, y = _num(crop, "x", 0, 99, 0), _num(crop, "y", 0, 99, 0)
        w, h = _num(crop, "w", 1, 100, 100), _num(crop, "h", 1, 100, 100)
        if x + w > 100.0001 or y + h > 100.0001:
            raise MediaError("The crop is outside the picture")
        rows, cols = image.shape[:2]
        x0, y0 = int(cols * x / 100), int(rows * y / 100)
        image = image[y0:max(y0 + 1, int(rows * (y + h) / 100)), x0:max(x0 + 1, int(cols * (x + w) / 100))]

    rotate = int(_num(ops, "rotate", 0, 270, 0))
    if rotate not in (0, 90, 180, 270):
        raise MediaError("'rotate' must be 0, 90, 180 or 270")
    image = {0: image, 90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 180: cv2.rotate(image, cv2.ROTATE_180),
             270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)}[rotate]

    flip = str(ops.get("flip") or "")
    if flip not in ("", "h", "v", "both"):
        raise MediaError("'flip' must be h, v or both")
    if flip:
        image = cv2.flip(image, {"h": 1, "v": 0, "both": -1}[flip])

    if ops.get("width") not in (None, "", 0):
        width = int(_num(ops, "width", 16, 4000, 0))
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    brightness, contrast = _num(ops, "brightness", -100, 100, 0), _num(ops, "contrast", 0.5, 2, 1)
    if brightness or contrast != 1:
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 2.55)
    if ops.get("grayscale"):
        image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in (".jpg", ".jpeg") else []
    ok, encoded = cv2.imencode(".jpg" if ext == ".jpeg" else ext, image, params)
    if not ok:
        raise MediaError("The edited picture could not be saved")
    return encoded.tobytes()


# ------------------------------------------------------------------ video / audio (ffmpeg)
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def edit_media(src: Path, dst_dir: Path, kind: str, ops: dict) -> Path:
    """ops: trim {start,end} in seconds; mute true (video); volume 0..4 (audio/video);
    extract_audio true (video -> mp3). Returns the new file in dst_dir (mp4 for video, mp3 for audio)."""
    exe = ffmpeg_path()
    if exe is None:
        raise MediaError("ffmpeg is not installed on the server, so video and audio editing is unavailable. "
                         "Install ffmpeg (https://ffmpeg.org) and restart Rafeeq.")
    if kind not in ("video", "audio"):
        raise MediaError("Only video and audio are edited with ffmpeg")
    extract = bool(ops.get("extract_audio")) and kind == "video"
    out_kind = "audio" if (extract or kind == "audio") else "video"
    dst = dst_dir / ("edited.mp3" if out_kind == "audio" else "edited.mp4")

    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    trim = ops.get("trim")
    if trim:
        start, end = _num(trim, "start", 0, 86400, 0), _num(trim, "end", 0, 86400, 0)
        if end <= start:
            raise MediaError("The end of the clip must be after its start")
        cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]
    cmd += ["-i", str(src)]
    volume = _num(ops, "volume", 0, 4, 1)
    if out_kind == "audio":
        cmd += ["-vn"] + (["-af", f"volume={volume:g}"] if volume != 1 else []) + ["-acodec", "libmp3lame"]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if ops.get("mute"):
            cmd += ["-an"]
        else:
            cmd += (["-af", f"volume={volume:g}"] if volume != 1 else []) + ["-c:a", "aac"]
    cmd.append(str(dst))
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise MediaError("The edit took too long and was stopped") from None
    if done.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise MediaError("ffmpeg could not do this edit: " + (done.stderr or "unknown error").strip()[-200:])
    return dst


# ------------------------------------------------------------------ speech
GEMINI_TTS_MODEL_DEFAULT = "gemini-2.5-flash-preview-tts"
_GEMINI_VOICES = {"female": "Kore", "male": "Charon"}          # prebuilt Gemini voices; the language follows the text


def _speech_providers() -> list[str]:
    forced = os.getenv("TTS_PROVIDER", "").strip().lower()
    if forced in ("edge", "gemini"):
        return [forced]
    return (["gemini"] if os.getenv("GOOGLE_API_KEY") else []) + ["edge"]


def speech_available() -> bool:
    if os.getenv("GOOGLE_API_KEY") and "gemini" in _speech_providers():
        return True
    try:
        import edge_tts  # noqa: F401
        return "edge" in _speech_providers()
    except ImportError:
        return False


GEMINI_TTS_RETRIES = 3


def _retry_after(response, default: float) -> float:
    """Seconds Google asks us to wait ('retryDelay': '23s' in the error body), capped at one minute."""
    try:
        found = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', response.text)
    except Exception:                                                         # noqa: BLE001
        found = None
    return min(60.0, float(found.group(1)) + 1 if found else default)


def _gemini_speech(text: str, voice: str, dst: Path) -> Path:
    import httpx

    model = os.getenv("GEMINI_TTS_MODEL") or GEMINI_TTS_MODEL_DEFAULT
    name = _GEMINI_VOICES["male" if "male" in VOICES[voice] and "female" not in VOICES[voice] else "female"]
    body = {"contents": [{"parts": [{"text": "Say clearly and calmly, like a teacher: " + text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": name}}}}}
    pcm = None
    for attempt in range(GEMINI_TTS_RETRIES + 1):
        try:
            res = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                             headers={"x-goog-api-key": os.environ["GOOGLE_API_KEY"]}, json=body, timeout=120)
            res.raise_for_status()
            parts = res.json()["candidates"][0]["content"]["parts"]
            pcm = base64.b64decode(next(p["inlineData"]["data"] for p in parts if "inlineData" in p))
            break
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 429 and attempt < GEMINI_TTS_RETRIES:                    # too many requests: wait as Google asks, then try again
                time.sleep(_retry_after(e.response, 20 * (attempt + 1)))
                continue
            hint = " - the voice quota is used up; wait a minute or check the quota of your key" if code == 429 else ""
            raise MediaError(f"Gemini voice refused the request ({code}){hint}") from None
        except Exception as e:
            raise MediaError(f"Gemini voice failed: {type(e).__name__}") from None
    exe = ffmpeg_path()
    if exe is None:
        raise MediaError("ffmpeg is not installed (needed to turn the Gemini voice into an mp3)")
    done = subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
                           "-c:a", "libmp3lame", "-q:a", "4", str(dst)], input=pcm, capture_output=True, timeout=FFMPEG_TIMEOUT)
    if done.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        raise MediaError("The Gemini voice could not be converted to mp3")
    return dst


def _edge_speech(text: str, voice: str, dst: Path) -> Path:
    try:
        import edge_tts
    except ImportError:
        raise MediaError("edge-tts is not installed (uv sync)") from None
    try:
        asyncio.run(edge_tts.Communicate(text=text, voice=voice).save(str(dst)))
    except Exception as e:
        raise MediaError(f"The Microsoft voice service refused or could not be reached: {type(e).__name__}") from None
    if not dst.is_file() or dst.stat().st_size == 0:
        raise MediaError("The voice service returned no audio")
    return dst


def synthesize_speech(text: str, voice: str, dst: Path) -> Path:
    """Narration as an .mp3: Gemini TTS when GOOGLE_API_KEY is set, then edge-tts. Needs internet access."""
    text = (text or "").strip()
    if len(text) < 2:
        raise MediaError("Write the text to read aloud")
    if len(text) > MAX_SPEECH_CHARS:
        raise MediaError(f"The text is longer than {MAX_SPEECH_CHARS} characters")
    if voice not in VOICES:
        raise MediaError("Unknown voice")
    errors = []
    for provider in _speech_providers():
        try:
            return (_gemini_speech if provider == "gemini" else _edge_speech)(text, voice, dst)
        except MediaError as e:
            errors.append(str(e))
    raise MediaError(" / ".join(errors))

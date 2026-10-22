"""Multi-scene story videos (about 30 seconds) for the Visual & Media Studio.

The video model (Wan) makes one clip of at most a few seconds, so a longer video is a chain:

  1. plan_story()    a text LLM writes ONE character description, ONE setting and N scenes
                     (visual prompt + narration). The same character/setting/style text is repeated
                     in every scene prompt: this is what keeps the look stable.
  2. advance()       one small step per call (the page polls every few seconds): submit scene 1,
                     wait, save the clip, take its LAST FRAME and submit scene 2 with that frame as
                     the first frame (image-to-video) ... so each scene continues the previous one.
  3. stitch()        ffmpeg joins the clips (cross-fade or hard cut), optional narration track.

State lives in the asset record (settings.story): no new storage. The finished video is the asset's
file (kind generated_video), so it appears in My Library like every other video.

Honest limits: continuity is good for background and colours, not perfect for faces/hands over six
hops; every scene is a paid video job; ffmpeg must be installed.
"""

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

from rafeeq.teacher import assets as asset_lib
from rafeeq.teacher import media_tools
from rafeeq.teacher.lecturer import video_gen

log = logging.getLogger(__name__)

MIN_SCENES, MAX_SCENES, DEFAULT_SCENES = 2, 8, 6
SCENE_SECONDS = 5
NARRATION_WORDS = (18, 32)          # a scene is held on its last frame while the teacher finishes speaking
FADE_SECONDS = 0.5
FFMPEG_TIMEOUT = 600

STYLES = {
    "storybook": "Warm hand-painted children's storybook illustration, soft watercolor textures, gentle light",
    "flat": "Clean flat vector illustration, bold simple shapes, bright educational colour palette",
    "clay": "Soft 3D clay-animation look, rounded friendly shapes, studio lighting",
    "realistic": "Realistic documentary footage, natural light, shallow depth of field, 4K detail",
}

_locks: dict[str, threading.Lock] = {}
_finishing: set[str] = set()          # stories whose joining/narration runs in a background thread
FINISH_WAIT_SECONDS = 20              # a poll waits this long for a quick finish before it answers "still working"


class StoryError(ValueError):
    """Bad input, a missing tool, or a failed step (the message is shown to the teacher)."""


# ------------------------------------------------------------------ planning
def _extract_json(text: str):
    """The first JSON object in an LLM answer (drops <think> blocks and code fences)."""
    if not isinstance(text, str):                       # some providers (Gemini) answer with a list of blocks
        text = "".join(b if isinstance(b, str) else str(b.get("text", "")) for b in (text or []) if isinstance(b, (str, dict)))
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start = text.find("{")
    if start == -1:
        raise StoryError("The planner did not return a story")
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        raise StoryError("The planner returned a story that could not be read") from None
    return data


def _clean(value, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _facts_block(bp: dict) -> str:
    lines = ["Verified teaching notes (use them, in order of importance):"]
    if bp.get("objective"):
        lines.append(f"- Goal: {bp['objective']}")
    lines += [f"- Fact: {f}" for f in bp.get("facts", [])]
    for key, label in (("analogy", "Analogy a child understands"), ("example", "Everyday example"), ("misconception", "Common mistake to correct")):
        if bp.get(key):
            lines.append(f"- {label}: {bp[key]}")
    return "\n".join(lines) + "\n"


def _blueprint(llm, topic: str, lang: str, material: str = "") -> dict:
    """First pass: the subject knowledge (accurate facts, an analogy, a common mistake) before any scene is written.
    With the room's material, the facts come from it (the topic means what it means there). Never fatal."""
    course = (f"Course material from the lecturer's files (the main source: take the facts from it, and read the topic in the "
              f"meaning it has here):\n{material}\n") if material else ""
    prompt = (
        f"You are an expert teacher. Topic: {topic}\n" + course +
        f"Write the teaching notes for a short video, in {lang}. Be accurate and specific (names, numbers, steps), no filler.\n"
        'Answer with JSON only: {"objective": "what the student can do after the video", "facts": ["6 to 8 short, correct, '
        'concrete facts in teaching order"], "analogy": "...", "example": "...", "misconception": "..."}')
    try:
        raw = llm.invoke(prompt)
        data = _extract_json(getattr(raw, "content", raw))
    except Exception as e:                                   # noqa: BLE001 - an optional first pass
        log.info("story_video: blueprint skipped (%s)", type(e).__name__)
        return {}
    if not isinstance(data, dict):
        return {}
    facts = [_clean(f, 220) for f in data.get("facts", []) if isinstance(f, str) and len(_clean(f, 220)) > 5][:8]
    if len(facts) < 3:
        return {}
    return {"facts": facts, **{k: _clean(data.get(k), 220) for k in ("objective", "analogy", "example", "misconception")}}


def plan_story(topic: str, scenes: int = DEFAULT_SCENES, language: str = "ar", style: str = "storybook", llm=None,
               material: str = "") -> dict:
    """{'style','character','setting','scenes':[{'visual','narration'}]}. Raises StoryError."""
    topic = _clean(topic, 400)
    if len(topic) < 3:
        raise StoryError("Write what the video should explain")
    if style not in STYLES:
        raise StoryError("Unknown style")
    scenes = max(MIN_SCENES, min(MAX_SCENES, int(scenes)))
    if llm is None:
        from rafeeq.teacher.lecturer.llm import get_llm
        llm = get_llm(provider=os.getenv("STORY_LLM_PROVIDER") or None, temperature=0.6)
    lang = "Arabic" if language == "ar" else "English"
    facts = _blueprint(llm, topic, lang, material)
    prompt = (
        f"You are a teacher writing a short EXPLAINER video of exactly {scenes} scenes that teaches: {topic}\n"
        + (_facts_block(facts) if facts else "") +
        "The video must TEACH, not just show a character walking around. Follow this arc: "
        "scene 1 = a hook / why it matters; middle scenes = one new idea or one step of the process each, in order, "
        "with a concrete fact, number or example; last scene = a one-sentence recap.\n"
        "Rules:\n"
        "- 'character': ONE fixed description of the main subject (a person, animal, object or phenomenon) that "
        "appears in the scenes, or an empty string when the topic is a process with no character.\n"
        "- 'setting': ONE fixed look and colour palette.\n"
        "- each scene: 'visual' = in English, what the camera shows and how it moves. Show the REAL thing being explained "
        "(the process, the object, the experiment), different in every scene. Never ask for text, labels or numbers on screen.\n"
        f"- each scene: 'narration' = what the teacher says over the scene, in {lang}, {NARRATION_WORDS[0]} to {NARRATION_WORDS[1]} words, "
        "simple, correct, and containing the teaching point of that scene (never filler such as 'let us see'). "
        "Use the facts above; do not add facts that are not true.\n"
        "Answer with JSON only: "
        '{"character": "...", "setting": "...", "scenes": [{"visual": "...", "narration": "..."}]}')
    try:
        raw = llm.invoke(prompt)
        data = _extract_json(getattr(raw, "content", raw))
    except StoryError:
        raise
    except Exception as e:
        log.warning("story_video: planner failed: %r", e)
        provider = os.getenv("STORY_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "default"
        where = f" at {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}" if provider == "ollama" else ""
        raise StoryError(f"The planner ('{provider}'{where}) is not available ({type(e).__name__}). "
                         "Check that the model server is running and the name in .env is right.") from None
    rows = [r for r in data.get("scenes", []) if isinstance(r, dict) and _clean(r.get("visual"), 10)]
    if len(rows) < MIN_SCENES:
        raise StoryError("The planner returned too few scenes")
    return {
        "style": style,
        "character": _clean(data.get("character"), 300),
        "setting": _clean(data.get("setting"), 200),
        "scenes": [{"visual": _clean(r.get("visual"), 300), "narration": _clean(r.get("narration"), 320)} for r in rows[:scenes]],
    }


def scene_prompt(story: dict, index: int, continuity: str = "") -> str:
    """The text sent to the video model for one scene. Style, character and setting come first
    (the model weighs the start most); every scene repeats them."""
    parts = [STYLES.get(story.get("style"), STYLES["storybook"]) + ".",
             f"Main character: {story.get('character', '')}." if story.get("character") else "",
             f"Setting: {story.get('setting', '')}." if story.get("setting") else "",
             f"Scene {index + 1} of {len(story['scenes'])}: {story['scenes'][index]['visual']}.",
             f"The previous scene ended like this: {continuity}." if continuity else "",
             "Keep exactly the same character design, colours and art style. Smooth slow camera motion. "
             "No text, captions or watermarks."]
    return " ".join(p for p in parts if p)[:780]


# ------------------------------------------------------------------ ffmpeg
def _exe(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise StoryError("ffmpeg is not installed on the server, so story videos cannot be joined. "
                         "Install ffmpeg (https://ffmpeg.org) and restart Rafeeq.")
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise StoryError("ffmpeg took too long and was stopped") from None
    if done.returncode != 0:
        raise StoryError("ffmpeg failed: " + (done.stderr or "unknown error").strip()[-200:])
    return done


def probe(path: Path) -> tuple[float, int, int]:
    """(seconds, width, height) of a video; sensible defaults if ffprobe cannot tell."""
    try:
        out = _run([_exe("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=width,height:format=duration", "-of", "json", str(path)]).stdout
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]
        return float(data["format"]["duration"]), int(stream.get("width") or 1280), int(stream.get("height") or 720)
    except (StoryError, KeyError, ValueError, TypeError):
        return float(SCENE_SECONDS), 1280, 720


def stitch_filter(durations: list[float], width: int, height: int, fade: float) -> tuple[str, str]:
    """ffmpeg filter_complex that joins the clips: every clip is scaled/padded to the same size and
    frame rate, then cross-faded (fade > 0) or cut (fade == 0). Returns (filter, output label)."""
    n = len(durations)
    norm = [f"[{i}:v]fps=24,scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v{i}]" for i in range(n)]
    if n == 1:
        return ";".join(norm) + ";[v0]null[out]", "[out]"
    if fade <= 0:
        return ";".join(norm) + ";" + "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]", "[out]"
    parts, last, elapsed = list(norm), "v0", 0.0
    for i in range(1, n):
        elapsed += durations[i - 1]
        offset = max(0.0, elapsed - i * fade)
        label = "out" if i == n - 1 else f"x{i}"
        parts.append(f"[{last}][v{i}]xfade=transition=fade:duration={fade:g}:offset={offset:.3f}[{label}]")
        last = label
    return ";".join(parts), "[out]"


def stitch(clips: list[Path], dst: Path, fade: float = FADE_SECONDS, narration: Path | None = None) -> Path:
    """Join the clips into dst (mp4, h264). With narration (mp3) it becomes the audio track; the video
    length wins: a longer narration is cut at the end of the picture."""
    if not clips:
        raise StoryError("There are no clips to join")
    exe = _exe("ffmpeg")
    durations = []
    for c in clips:
        sec, w, h = probe(c)
        durations.append(sec)
        if c is clips[0]:
            width, height = w - w % 2, h - h % 2
    graph, label = stitch_filter(durations, width, height, fade)
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error"]
    for c in clips:
        cmd += ["-i", str(c)]
    if narration is not None:
        cmd += ["-i", str(narration)]
    cmd += ["-filter_complex", graph, "-map", label]
    if narration is not None:
        cmd += ["-map", f"{len(clips)}:a", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    _run(cmd)
    if not dst.is_file() or dst.stat().st_size == 0:
        raise StoryError("The joined video is empty")
    return dst



def extend_clip(clip: Path, dst: Path, seconds: float) -> Path:
    """Hold the last frame of a clip until it lasts `seconds` (so the narration can finish)."""
    sec, _, _ = probe(clip)
    if seconds <= sec + 0.05:
        return clip
    _run([_exe("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", str(clip),
          "-vf", f"tpad=stop_mode=clone:stop_duration={seconds - sec:.3f}", "-an",
          "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)])
    return dst


def mix_narration(parts: list[tuple[Path, float]], dst: Path) -> Path:
    """Place every scene's narration (mp3) at its start time (seconds) on one audio track."""
    if not parts:
        raise StoryError("There is no narration to place")
    cmd = [_exe("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    for path, _ in parts:
        cmd += ["-i", str(path)]
    delays = [f"[{i}:a]adelay={int(max(0.0, at) * 1000)}|{int(max(0.0, at) * 1000)}[a{i}]" for i, (_, at) in enumerate(parts)]
    mix = "".join(f"[a{i}]" for i in range(len(parts))) + f"amix=inputs={len(parts)}:normalize=0[out]"
    _run(cmd + ["-filter_complex", ";".join(delays + [mix]), "-map", "[out]", "-c:a", "libmp3lame", "-q:a", "4", str(dst)])
    if not dst.is_file() or dst.stat().st_size == 0:
        raise StoryError("The narration track is empty")
    return dst


def last_frame(clip: Path, dst: Path) -> Path:
    """The final picture of a clip (becomes the first frame of the next scene)."""
    _run([_exe("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-sseof", "-0.2", "-i", str(clip),
          "-frames:v", "1", "-update", "1", "-q:v", "2", str(dst)])
    if not dst.is_file() or dst.stat().st_size == 0:
        raise StoryError("Could not read the last frame of the clip")
    return dst


def to_gif(src: Path, dst: Path, start: float = 0.0, end: float | None = None, fps: int = 12, width: int = 480) -> Path:
    """A short looping GIF (palette-optimised so the colours stay clean) from a video."""
    cmd = [_exe("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if end:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-vf",
            f"fps={int(fps)},scale={int(width)}:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", str(dst)]
    _run(cmd)
    if not dst.is_file() or dst.stat().st_size == 0:
        raise StoryError("The GIF could not be made")
    return dst


# ------------------------------------------------------------------ optional local vision (Ollama)
def describe_frame(png: bytes) -> str:
    """One sentence about a picture from a local vision model (Ollama, e.g. qwen2.5vl:7b), used as the
    'previous scene ended like this' line of the next prompt. '' when Ollama is not reachable."""
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    body = {"model": os.getenv("OLLAMA_VL_MODEL", "qwen2.5vl:7b"), "stream": False,
            "messages": [{"role": "user", "images": [base64.b64encode(png).decode("ascii")],
                          "content": "In one English sentence of at most 30 words, describe the main character "
                                     "(clothes, colours), what they are doing, and the background."}]}
    try:
        req = urllib.request.Request(base + "/api/chat", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            return _clean(json.load(resp).get("message", {}).get("content"), 300)
    except Exception as e:
        log.info("story_video: vision note skipped (%s)", type(e).__name__)
        return ""


# ------------------------------------------------------------------ the chain
def story_key(teacher_id: str, topic: str, scenes: int, style: str, language: str, voice: str, fade: bool) -> str:
    digest = hashlib.sha1(f"{topic.strip()}|{scenes}|{style}|{language}|{voice}|{fade}".encode()).hexdigest()[:16]
    return f"story:{digest}"


def start_story(store, teacher_id: str, topic: str, *, scenes: int = DEFAULT_SCENES, style: str = "storybook",
                language: str = "ar", voice: str = "", fade: bool = True, vision: bool = False,
                room_id: str = "", llm=None, retry: bool = False) -> dict:
    """Plan the story and create the pending library video. The same request returns the same video
    (no second payment); a failed one is restarted only with retry=True."""
    scenes = max(MIN_SCENES, min(MAX_SCENES, int(scenes)))
    if voice and voice not in media_tools.VOICES:
        raise StoryError("Unknown voice")
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise StoryError("DASHSCOPE_API_KEY is not set in .env")
    if voice and not media_tools.speech_available():
        raise StoryError("Narration needs GOOGLE_API_KEY (Gemini voice) or edge-tts: add the key, or choose 'No narration'")
    key = story_key(teacher_id, topic, scenes, style, language, voice, fade)
    existing = asset_lib.find_by_key(store, teacher_id, key)
    prior = ((existing or {}).get("settings") or {}).get("story") or {}
    if retry and existing and existing["status"] == "ready" and voice and prior and not prior.get("narrated"):
        # the scenes are paid and saved: only the narration is done again
        prior.pop("narration_error", None)
        return asset_lib.update_asset(store, teacher_id, existing["asset_id"], status="pending", error="",
                                      settings={**existing["settings"], "story": prior})
    if existing and not (existing["status"] == "failed" and retry) and not asset_lib.is_stale_pending(existing, minutes=240):
        return existing
    from rafeeq.teacher.lecturer.material import room_material      # a room's story explains the room's own material

    material = room_material({"room_id": room_id, "topic": topic})
    extra = {"material": material} if material else {}               # no room material: the same call as before
    plan = plan_story(topic, scenes, language, style, llm=llm, **extra)
    story = {**plan, "seed": secrets.randbelow(2_000_000_000), "topic": _clean(topic, 400), "voice": voice, "fade": bool(fade), "vision": bool(vision),
             "scenes": [{**s, "state": "todo", "task_id": "", "clip": "", "frame": "", "note": ""} for s in plan["scenes"]]}
    settings = {"story": story, "duration": SCENE_SECONDS * len(story["scenes"])}
    if existing:
        return asset_lib.update_asset(store, teacher_id, existing["asset_id"], status="pending", error="", settings=settings)
    return asset_lib.register_asset(
        store, teacher_id, kind="generated_video", title=_clean(topic, 80), origin="ai_generated",
        provider="qwencloud", model=video_gen.DEFAULT_I2V_MODEL, settings_used=settings, status="pending",
        room_id=room_id, idempotency_key=key)


def story_view(asset: dict) -> dict:
    story = (asset.get("settings") or {}).get("story") or {}
    scenes = story.get("scenes", [])
    done = sum(1 for s in scenes if s["state"] == "done")
    return {"asset_id": asset["asset_id"], "title": asset["title"], "status": asset["status"], "error": asset.get("error", ""),
            "scene_count": len(scenes), "scenes_done": done, "style": story.get("style", ""),
            "topic": story.get("topic", ""), "voice": story.get("voice", ""), "fade": bool(story.get("fade", True)),
            "finishing": asset["asset_id"] in _finishing, "vision": bool(story.get("vision", False)), "narrated": bool(story.get("narrated", False)),
            "narration_error": story.get("narration_error", ""),
            "scenes": [{"n": i + 1, "state": s["state"], "narration": s.get("narration", ""), "visual": s.get("visual", "")}
                       for i, s in enumerate(scenes)],
            "file_url": f"/teacher/assets/{asset['asset_id']}/file" if asset["status"] == "ready" and asset_lib.asset_file(asset) else None}


def _path(ref: str) -> Path:
    return asset_lib._resolve("assets", ref)


def _fail(store, teacher_id, asset, story, message: str) -> dict:
    return asset_lib.update_asset(store, teacher_id, asset["asset_id"], status="failed", error=message[:300],
                                  settings={**asset["settings"], "story": story})


def advance(store, teacher_id: str, asset_id: str) -> dict | None:
    """Do the next small step of the chain and return the (updated) asset. Safe to call repeatedly:
    nothing happens while a scene is still rendering, and a finished story is returned as it is."""
    lock = _locks.setdefault(asset_id, threading.Lock())
    if not lock.acquire(blocking=False):
        return asset_lib.get_asset(store, teacher_id, asset_id)          # another request is already advancing it
    try:
        asset = asset_lib.get_asset(store, teacher_id, asset_id)
        if asset is None or asset["status"] != "pending":
            return asset
        story = (asset.get("settings") or {}).get("story")
        if not story:
            return asset
        scenes = story["scenes"]
        i = next((k for k, s in enumerate(scenes) if s["state"] != "done"), None)
        if i is None:
            return _finish_in_background(store, teacher_id, asset, story)
        s = scenes[i]

        if s["state"] == "todo":
            first = None
            if i > 0:
                first = _path(scenes[i - 1]["frame"]).read_bytes()
            prompt = scene_prompt(story, i, scenes[i - 1].get("note", "") if i else "")
            result = video_gen.submit_task(prompt, first, "frame.jpg", duration=SCENE_SECONDS,
                                           prompt_extend=False, seed=story.get("seed"))
            if not result["ok"]:
                return _fail(store, teacher_id, asset, story, f"Scene {i + 1} could not start: {result['error']}")
            s.update(state="running", task_id=result["task_id"], prompt=prompt)
        else:
            result = video_gen.poll_task(s["task_id"])
            if result["status"] in ("FAILED", "CANCELED", "ERROR"):
                return _fail(store, teacher_id, asset, story, f"Scene {i + 1} failed: {result['error'] or result['status']}")
            if result["status"] != "SUCCEEDED" or not result["video_url"]:
                return asset                                              # still rendering
            try:
                _, clip_ref = asset_lib.save_file(teacher_id, asset_id, video_gen.download_video(result["video_url"]), f"scene_{i + 1:02d}.mp4")
                with tempfile.TemporaryDirectory() as tmp:
                    frame = last_frame(_path(clip_ref), Path(tmp) / "last.jpg")
                    frame_bytes = frame.read_bytes()
                _, frame_ref = asset_lib.save_file(teacher_id, asset_id, frame_bytes, f"scene_{i + 1:02d}_last.jpg")
            except (StoryError, asset_lib.AssetError, ValueError, OSError) as e:
                return _fail(store, teacher_id, asset, story, f"Scene {i + 1} could not be saved: {e}")
            s.update(state="done", clip=clip_ref, frame=frame_ref)
            if story.get("vision") and i < len(scenes) - 1:
                s["note"] = describe_frame(frame_bytes)
        return asset_lib.update_asset(store, teacher_id, asset_id, settings={**asset["settings"], "story": story})
    finally:
        lock.release()


def silence_points(audio: Path) -> list[float]:
    """Middle of every pause in a voice track (seconds): the places where it can be cut without breaking a word."""
    done = subprocess.run([_exe("ffmpeg"), "-hide_banner", "-nostats", "-i", str(audio), "-af", "silencedetect=noise=-35dB:d=0.15",
                           "-f", "null", "-"], capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    starts = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", done.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", done.stderr)]
    return [(a + b) / 2 for a, b in zip(starts, ends)]


def cut_audio(src: Path, dst: Path, start: float, end: float) -> Path:
    _run([_exe("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
          "-c:a", "libmp3lame", "-q:a", "4", str(dst)])
    return dst


def split_points(weights: list[int], total: float, pauses: list[float], reach: float = 2.0) -> list[float]:
    """Where each scene's part of ONE voice track ends: proportional to its text, moved to the nearest pause
    (within `reach` seconds) so a word is never cut. The last point is the end of the track."""
    whole = sum(weights) or 1
    points, acc = [], 0
    for w in weights[:-1]:
        acc += w
        target = total * acc / whole
        near = [p for p in pauses if abs(p - target) <= reach]
        points.append(min(near, key=lambda p: abs(p - target)) if near else target)
    return [*points, total]


def _narrate(story: dict, clips: list[Path], tmp: Path) -> tuple[list[Path], Path]:
    """ONE voice request for the whole video (voice services limit the number of requests), cut at pauses into one
    part per scene. A scene lasts at least as long as its part (the last frame is held), and each part starts when
    its scene starts on the joined video."""
    fade = FADE_SECONDS if story.get("fade") else 0.0
    texts = [(sc.get("narration") or "").strip() for sc in story["scenes"]]
    spoken = [i for i, t in enumerate(texts) if len(t) >= 2]
    if not spoken:
        raise StoryError("No scene has narration")
    full = "\n".join(texts[i] for i in spoken)[:media_tools.MAX_SPEECH_CHARS]
    voice_file = media_tools.synthesize_speech(full, story["voice"], tmp / "voice.mp3")
    total, _, _ = probe(voice_file)
    ends = split_points([len(texts[i]) for i in spoken], total, silence_points(voice_file))
    begins = [0.0, *ends[:-1]]
    part = {i: cut_audio(voice_file, tmp / f"voice_{i}.mp3", begins[k], ends[k]) for k, i in enumerate(spoken)}
    out, parts, at = [], [], 0.0
    for i, clip in enumerate(clips):
        if i in part:
            clip = extend_clip(clip, tmp / f"held_{i}.mp4", probe(part[i])[0] + 0.4)
            parts.append((part[i], at))
        out.append(clip)
        at += probe(clip)[0] - fade
    return out, mix_narration(parts, tmp / "narration.mp3")


def _finish_in_background(store, teacher_id: str, asset: dict, story: dict) -> dict:
    """Narration (one voice call per scene) and the final encode can take minutes, longer than one web request.
    They run in a thread; this call waits a few seconds so a quick finish is returned at once."""
    asset_id = asset["asset_id"]
    if asset_id in _finishing:
        return asset
    _finishing.add(asset_id)

    def work():
        try:
            _finish(store, teacher_id, asset, story)
        except Exception as e:                                         # noqa: BLE001 - never leave the story pending forever
            log.exception("story_video: finishing failed")
            _fail(store, teacher_id, asset, story, f"The scenes could not be joined: {type(e).__name__}")
        finally:
            _finishing.discard(asset_id)

    worker = threading.Thread(target=work, name=f"story-{asset_id}", daemon=True)
    worker.start()
    worker.join(FINISH_WAIT_SECONDS)
    return asset_lib.get_asset(store, teacher_id, asset_id) or asset


def _finish(store, teacher_id: str, asset: dict, story: dict) -> dict:
    """All scenes are saved: join them (and the narration) into the library video."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clips = [_path(s["clip"]) for s in story["scenes"]]
            narration = None
            if story.get("voice") and any(s.get("narration") for s in story["scenes"]):
                try:
                    clips, narration = _narrate(story, clips, tmp)
                    story.pop("narration_error", None)
                except (media_tools.MediaError, StoryError) as e:
                    log.warning("story_video: narration skipped: %s", e)       # the silent video is still delivered
                    story["narration_error"] = _clean(e, 200)
            fade = FADE_SECONDS if story.get("fade") else 0
            final = stitch(clips, tmp / "story.mp4", fade=fade, narration=narration)
            root, ref = asset_lib.save_file(teacher_id, asset["asset_id"], final.read_bytes(), "story.mp4")
            story["narrated"] = narration is not None
    except (StoryError, asset_lib.AssetError, OSError) as e:
        return _fail(store, teacher_id, asset, story, f"The scenes could not be joined: {e}")
    return asset_lib.update_asset(store, teacher_id, asset["asset_id"], status="ready", storage_root=root, storage_ref=ref,
                                  error="", settings={**asset["settings"], "story": story})

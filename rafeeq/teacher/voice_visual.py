"""Voice-to-Visual (M5): what the lecturer SAYS becomes pictures on the video timeline.

    1. the lecture's caption lines (from the editor, or transcribed now with Gemini)
    2. an LLM reads them and picks the moments where a picture helps students: a diagram for a process or a
       structure, an illustration for a thing or a scene, a short animation (GIF) for an action or a reaction
    3. each moment is drawn: diagrams and illustrations by the image model (image_gen, async, saved in the
       library), animations found on Giphy
    4. the editor places them as stickers at the moment they are spoken; the lecturer can move, resize or
       delete any of them before the final render

Nothing invents content: every moment and every picture idea comes from the captions.
"""
import hashlib
import logging
import os
from typing import Callable
from rafeeq.observability import traced

log = logging.getLogger(__name__)

MAX_VISUALS = 12           # also the editor's sticker limit
MIN_SECONDS, MAX_SECONDS, DEFAULT_SECONDS = 3.0, 12.0, 6.0
MIN_GAP = 2.0                     # seconds between two visuals, so the screen is never crowded
KINDS = ("diagram", "illustration", "animation")

PLAN_SYSTEM = (
    "You add visuals to a recorded lecture. You get its caption lines, each with start and end in seconds. "
    f"Pick the 3 to {MAX_VISUALS} moments where a picture would most help a student understand what is being SAID "
    "(a process, a structure, a comparison, an object, a scene). Skip greetings, jokes and small talk. "
    "For each moment give:\n"
    '- "start": the second the idea is first said, "end": when the picture should leave (4 to 10 seconds later)\n'
    '- "kind": "diagram" for a process, steps, a structure or a relation; "illustration" for a thing, place or scene; '
    '"animation" only for a simple action or a reaction (a short GIF)\n'
    '- "label": 2 to 5 words in the language of the lecture naming the idea\n'
    '- "prompt": for diagram and illustration, ONE English prompt (at most 50 words) for an image model; '
    "describe shapes, arrows and layout, never ask for written words or letters\n"
    '- "query": for animation, 1 to 3 English words to search GIFs\n'
    "Use only ideas that appear in the captions. Moments must not overlap. "
    'Reply with JSON only: {"visuals": [{"start": 0, "end": 0, "kind": "...", "label": "...", "prompt": "...", "query": "..."}]}')

STYLE = {
    "diagram": "Clean flat educational diagram on a plain white background, simple shapes, clear arrows, soft colours, "
               "plenty of empty space, no text, no letters, no numbers.",
    "illustration": "Clean flat educational illustration, simple shapes, soft colours, plain light background, "
                    "no text, no letters, no people's faces.",
}


class PlanError(Exception):
    """The LLM gave no usable answer (not the same as 'nothing needs a picture')."""


def _sec(x) -> float:
    """A time from the LLM: 12, 12.5, "12.5", "12.5s", "0:12" or "1:02:05"."""
    if isinstance(x, (int, float)):
        return float(x)
    text = str(x).strip().lower().rstrip("s").strip()
    if ":" in text:
        out = 0.0
        for part in text.split(":"):
            out = out * 60 + float(part)
        return out
    return float(text)


def _items(out) -> list | None:
    """The list of moments, whatever the model called it."""
    if isinstance(out, list):
        return out
    if isinstance(out, dict):
        for key in ("visuals", "moments", "items", "results"):
            if isinstance(out.get(key), list):
                return out[key]
        return next((v for v in out.values() if isinstance(v, list)), None)
    return None


def clean_plan(raw, total: float) -> list[dict]:
    """Keep only well-formed moments, in time order, inside the video and not overlapping. Never raises."""
    items = []
    for r in raw if isinstance(raw, list) else []:
        try:
            start = _sec(r["start"])
            end = _sec(r.get("end") or start + DEFAULT_SECONDS)
            kind = str(r.get("kind") or "illustration").strip().lower()
            label = " ".join(str(r.get("label") or "").split())[:40]
            prompt = " ".join(str(r.get("prompt") or "").split())[:500]
            query = " ".join(str(r.get("query") or "").split())[:40]
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if kind not in KINDS:
            kind = "illustration"
        if kind == "animation" and not query:
            kind, prompt = "illustration", prompt or label
        if kind != "animation" and not prompt:
            if not label:
                continue
            prompt = label
        if start < 0 or start >= total - 1:
            continue
        end = min(total, max(start + MIN_SECONDS, min(end, start + MAX_SECONDS)))
        items.append({"start": round(start, 2), "end": round(end, 2), "kind": kind, "label": label or query or kind,
                      "prompt": prompt, "query": query})
    out: list[dict] = []
    for it in sorted(items, key=lambda x: x["start"]):
        if out and it["start"] < out[-1]["end"] + MIN_GAP:
            continue
        out.append(it)
    return out[:MAX_VISUALS]


@traced("voice_visual.plan", "generation")
def plan_visuals(segments: list[dict], total: float, propose: Callable | None) -> list[dict]:
    """The moments that deserve a picture. Needs the LLM (`propose(system, user) -> dict`).
    Raises PlanError when the LLM fails, so the lecturer is not told 'nothing needs a picture' by mistake."""
    if not segments or propose is None:
        return []
    lines = "\n".join(f"{s['start']:.1f}-{s['end']:.1f}s: {s['text']}" for s in segments)[:80000]
    wanted = max(3, min(MAX_VISUALS, round(total / 300)))          # about one picture every 5 minutes, 3 to 12
    try:
        out = propose(PLAN_SYSTEM, f"Pick at most {wanted} moments, spread over the whole lecture.\n\n{lines}")
    except Exception as e:                                           # noqa: BLE001
        log.exception("voice-to-visual: planning failed")
        raise PlanError(f"{type(e).__name__}: {str(e)[:150]}") from None
    items = _items(out)
    if items is None:
        log.warning("voice-to-visual: the LLM answer has no list of moments: %r", str(out)[:300])
        raise PlanError("the LLM did not answer, or its answer was not JSON")
    plan = clean_plan(items, total)
    if items and not plan:
        log.warning("voice-to-visual: every moment was dropped, first one was %r", str(items[0])[:300])
    return plan


def image_prompt(item: dict) -> str:
    return f"{item['prompt']}. {STYLE.get(item['kind'], STYLE['illustration'])}"[:800]


def job_key(source_id: str, item: dict) -> str:
    return "v2v:" + hashlib.sha1(f"{source_id}|{item['start']}|{item['kind']}|{item['prompt']}".encode()).hexdigest()[:24]


def gemini_available() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY", "").strip())


def gemini_propose(system: str, user: str):
    """The planner on Gemini (JSON mode), the same key and call the captions use."""
    from rafeeq.teacher.montage import _gemini_json

    model = os.getenv("VOICE_VISUAL_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    return _gemini_json([{"text": system + "\n\nCaption lines:\n" + user}], model)

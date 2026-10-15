"""Diffusion image generation for slide pictures (QwenCloud / DashScope text-to-image, async, paid).

Same account, key and task polling as video_gen.py; only the endpoint and the model differ:
    IMAGE_MODEL  (default wan2.2-t2i-flash)   IMAGE_SIZE  (default 1280*720)

Provider-dependent: the model name and endpoint follow DashScope's documentation and have not been
run against the live service from this repository's tests (they use a fake). Every function returns a
dict with an error message instead of raising.
"""

import json
import logging
import os
import urllib.error
import urllib.request

from rafeeq.teacher.lecturer import video_gen

log = logging.getLogger(__name__)

IMAGE_URL = f"{video_gen.BASE_URL}/api/v1/services/aigc/text2image/image-synthesis"
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "wan2.2-t2i-flash")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1280*720")
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


def submit_task(prompt: str) -> dict:
    """{"ok": True, "task_id", "model"} or {"ok": False, "error"}."""
    if not video_gen._api_key():
        return {"ok": False, "error": "DASHSCOPE_API_KEY is not set in .env"}
    body = {"model": IMAGE_MODEL, "input": {"prompt": prompt.strip()[:800]}, "parameters": {"size": IMAGE_SIZE, "n": 1}}
    try:
        data = video_gen._post(IMAGE_URL, body, extra_headers={"X-DashScope-Async": "enable"})
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        log.warning("image_gen: submit failed (%s): %s", e.code, detail)
        return {"ok": False, "error": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        log.warning("image_gen: submit failed: %s: %s", type(e).__name__, str(e)[:300])
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    task_id = (data.get("output") or {}).get("task_id")
    if not task_id:
        return {"ok": False, "error": f"No task_id in response: {json.dumps(data)[:300]}"}
    return {"ok": True, "task_id": task_id, "model": IMAGE_MODEL}


def poll_task(task_id: str) -> dict:
    """{"status": RUNNING|SUCCEEDED|FAILED|ERROR|..., "image_url": str|None, "error": str|None}."""
    if not video_gen._api_key():
        return {"status": "ERROR", "image_url": None, "error": "DASHSCOPE_API_KEY is not set in .env"}
    try:
        data = video_gen._get(video_gen.TASK_URL.format(task_id=task_id))
    except Exception as e:
        return {"status": "ERROR", "image_url": None, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    output = data.get("output") or {}
    status = output.get("task_status", "UNKNOWN")
    results = output.get("results") or []
    url = next((r.get("url") for r in results if isinstance(r, dict) and r.get("url")), None)
    return {"status": status, "image_url": url,
            "error": (output.get("message") or "Image generation failed") if status in ("FAILED", "ERROR", "CANCELED") else None}


def download_image(url: str) -> bytes:
    """Fetch the finished picture (the provider link expires). Raises on any problem."""
    if not url.lower().startswith("https://"):
        raise ValueError("Only https image links are downloaded")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
    if not data or len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("The image download was empty or too large")
    return data

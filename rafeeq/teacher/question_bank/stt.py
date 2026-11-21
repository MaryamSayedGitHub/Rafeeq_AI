"""Speech to text for voice search: the teacher speaks, we get the words.

Uses an OpenAI-compatible /audio/transcriptions endpoint (Groq Whisper by default).
Settings (all optional, read from .env):
  M6_STT_API_KEY   falls back to GROQ_API_KEY, then LLM_API_KEY
  M6_STT_BASE_URL  falls back to LLM_BASE_URL when that is a Groq URL, else https://api.groq.com/openai/v1
  M6_STT_MODEL     default whisper-large-v3-turbo
"""
from __future__ import annotations

import os
from typing import Optional

GROQ_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "whisper-large-v3-turbo"
TIMEOUT_SECONDS = 60
MAX_AUDIO_BYTES = 10 * 1024 * 1024   # 10 MB is minutes of speech; the API limit is 25 MB


class SttUnavailable(RuntimeError):
    """No key is set, so voice search cannot be used."""


class SttError(RuntimeError):
    """The speech service refused or failed."""


def _settings() -> tuple[str, str, str]:
    key = (os.getenv("M6_STT_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
    if not key or key.startswith("put-your"):
        raise SttUnavailable("Voice search needs M6_STT_API_KEY (or GROQ_API_KEY) in .env")
    llm_base = os.getenv("LLM_BASE_URL", "").strip()
    base = (os.getenv("M6_STT_BASE_URL") or (llm_base if "groq" in llm_base.lower() else "") or GROQ_BASE).rstrip("/")
    return key, base, os.getenv("M6_STT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def transcribe(audio: bytes, filename: str = "audio.webm", content_type: Optional[str] = None,
               language: Optional[str] = None) -> str:
    """The words in `audio`. `language` ("ar", "en", ...) is optional: leave it out to let Whisper detect it."""
    import httpx

    key, base, model = _settings()
    data = {"model": model, "response_format": "json", "temperature": "0"}
    if language:
        data["language"] = language
    try:
        response = httpx.post(
            f"{base}/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (filename or "audio.webm", audio, content_type or "application/octet-stream")},
            data=data,
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise SttError(f"could not reach the speech service: {exc}") from exc
    if response.status_code >= 400:
        raise SttError(f"the speech service answered {response.status_code}: {response.text[:200]}")
    return str(response.json().get("text", "")).strip()
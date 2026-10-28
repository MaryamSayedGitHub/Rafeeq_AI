"""One function per provider: llm(prompt) -> text. Keys come from .env."""
from __future__ import annotations

import os
import re
import time
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-3.5-flash-lite"}

TRANSIENT = ("503", "UNAVAILABLE", "overloaded", "high demand", "timed out", "Timeout", "Connection", "Network")
RATE_LIMIT = ("429", "RESOURCE_EXHAUSTED", "rate limit")
NO_RETRY = ("insufficient_quota", "no credits")  # billing problems: waiting won't help


def _with_retries(fn: Callable[[str], str], tries: int = 4) -> Callable[[str], str]:
    """Retry temporary errors with a growing wait. Rate-limit errors wait at
    least 20 seconds (per-minute limits reset after a minute)."""

    def wrapped(prompt: str) -> str:
        delay = 2.0
        for attempt in range(tries):
            try:
                return fn(prompt)
            except Exception as exc:
                text = str(exc)
                if any(s in text for s in NO_RETRY):
                    raise
                rate_limited = any(s in text for s in RATE_LIMIT)
                transient = rate_limited or any(s in text for s in TRANSIENT)
                if not transient or attempt == tries - 1:
                    raise
                wait = max(delay, 20.0) if rate_limited else delay
                time.sleep(min(wait, 60.0))
                delay *= 2
        return ""

    return wrapped


def make_llm(provider: str, model: str | None = None) -> Callable[[str], str]:
    model = model or DEFAULT_MODELS.get(provider, "")

    if provider in ("compatible", "openai_compatible"):
        # The app's main model: any OpenAI-compatible endpoint (Groq, Gemini's compatible mode, ...).
        key = os.getenv("LLM_API_KEY", "").strip()
        if not key or key.startswith("put-your"):
            raise RuntimeError("LLM_API_KEY is missing from .env")
        base = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = model if model and model != DEFAULT_MODELS.get(provider) else os.getenv("LLM_MODEL", "gpt-4o-mini")

        def call_compatible(prompt: str) -> str:
            import httpx

            r = httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                           json={"model": model, "messages": [{"role": "user", "content": prompt}],
                                 "response_format": {"type": "json_object"}, "temperature": 0.3},
                           timeout=90)
            if r.status_code >= 400:  # the status code in the message drives _with_retries
                raise RuntimeError(f"{r.status_code} {r.text[:200]}")
            return r.json()["choices"][0]["message"]["content"] or ""

        return _with_retries(call_compatible)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing from .env")
        from openai import OpenAI

        client = OpenAI()

        def call_openai(prompt: str) -> str:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return r.choices[0].message.content or ""

        return _with_retries(call_openai)

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing from .env")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        config = types.GenerateContentConfig(response_mime_type="application/json")

        def call_gemini(prompt: str) -> str:
            r = client.models.generate_content(model=model, contents=prompt, config=config)
            return r.text or ""

        return _with_retries(call_gemini)

    if provider == "compat":
        # Any OpenAI-compatible host (Groq, Cerebras, OpenRouter, ...) with
        # open models such as GPT OSS or Qwen.
        key = os.getenv("COMPAT_API_KEY")
        base = os.getenv("COMPAT_BASE_URL")
        if not (key and base and model):
            raise RuntimeError(
                "Set COMPAT_API_KEY, COMPAT_BASE_URL and M4_LLM_MODEL in .env"
            )
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base)

        def call_compat(prompt: str) -> str:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            text = r.choices[0].message.content or ""
            # reasoning models may print their thinking first; drop it
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

        return _with_retries(call_compat)

    raise ValueError(f"unknown provider: {provider}")
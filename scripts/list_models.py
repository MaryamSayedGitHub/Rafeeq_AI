"""Which models can your keys use? Prints the model ids of the OpenAI-compatible endpoint in .env (LLM_BASE_URL + LLM_API_KEY)
and, when GROQ_API_KEY is set, of Groq. Use one of them for LLM_MODEL / GROQ_MODEL.

    uv run python scripts/list_models.py
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def show(name: str, base: str, key: str, current: str) -> None:
    try:
        r = requests.get(base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        r.raise_for_status()
        ids = sorted(m["id"] for m in r.json().get("data", []))
    except Exception as e:  # noqa: BLE001
        print(f"{name}: could not list models ({type(e).__name__}: {str(e)[:150]})")
        return
    print(f"{name} ({base}): {len(ids)} models")
    for i in ids:
        print(("  * " if i == current else "    ") + i)
    if current and current not in ids:
        print(f"  !! {current} (set in .env) is NOT in this list: pick one above")


key = os.getenv("LLM_API_KEY", "").strip()
if key and not key.startswith("put-your"):
    show("LLM_* (Copilot, tutor, quiz, Snap)", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), key, os.getenv("LLM_MODEL", ""))
if os.getenv("GROQ_API_KEY"):
    show("GROQ_* (M5 lecturer assistant)", "https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"], os.getenv("GROQ_MODEL", ""))
google = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if google:
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": google, "pageSize": 200}, timeout=20)
        r.raise_for_status()
        ids = sorted(m["name"].removeprefix("models/") for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", []))
        current = os.getenv("GEMINI_MODEL", "")
        print(f"Google Gemini (GOOGLE_API_KEY): {len(ids)} text models")
        for i in ids:
            print(("  * " if i == current else "    ") + i)
    except Exception as e:  # noqa: BLE001
        print(f"Google Gemini: could not list models ({type(e).__name__})")   # the key is in the URL: never print the error text

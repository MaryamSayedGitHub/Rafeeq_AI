"""Checks the orchestrator's model settings without starting the app.

    uv run python -m rafeeq.copilot.check_llm

Prints which provider and model are used, whether a JSON chat call works, whether the
model can read an image (needed by Snap), and how a few sample messages are routed.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv(".env")

from rafeeq.copilot.llm import chat_json, configured, vision_json  # noqa: E402
import time  # noqa: E402

from rafeeq.copilot.router import ROUTES, classify_with_source, rules  # noqa: E402
from rafeeq.copilot.router import SYSTEM as ROUTER_SYSTEM  # noqa: E402

SAMPLES = [
    ("اشرحلي يعني إيه overfitting", False, "ask"),
    ("Explain gradient descent in simple words", False, "ask"),
    ("اختبرني في الفصل التاني", False, "quiz"),
    ("give me 5 practice questions", False, "quiz"),
    ("صحّحلي الحل ده", True, "check_work"),
    ("is my answer right?", True, "check_work"),
    ("أذاكر إيه النهارده؟", False, "plan"),
    ("hmm", False, "clarify"),
]


def _tiny_png() -> bytes:
    """A 64x32 white image with a black bar, made without extra libraries."""
    import struct
    import zlib

    w, h = 64, 32
    rows = b"".join(b"\x00" + bytes(255 if not (12 <= y < 20 and 8 <= x < 56) else 0 for x in range(w)) for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main() -> None:
    key = os.environ.get("LLM_API_KEY", "")
    print("LLM_API_KEY :", "set" if key else "MISSING", "(still the placeholder!)" if key.startswith("put-your") else "")
    print("LLM_BASE_URL:", os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1 (default)"))
    print("LLM_MODEL   :", os.environ.get("LLM_MODEL", "gpt-4o-mini (default)"))
    if not configured():
        print("\nNo key: the Copilot will route with keyword rules, and Snap will answer 503.")
        return

    out = chat_json('Reply with JSON {"ok": true}.', "ping")
    print("\nJSON chat   :", "OK" if out else "FAILED (wrong key, base URL or model; or no JSON mode)")

    seen = vision_json('Reply with JSON {"shape": "<what you see>"}.', "Describe the image in a few words.", _tiny_png())
    print("Image input :", f"OK -> {seen}" if seen else "FAILED (this model may not accept images; Snap needs it)")

    if not out:
        return
    print("\nThe model alone (expected | llm):")
    right = 0
    for text, has_file, expected in SAMPLES:
        time.sleep(1)  # stay under free-tier rate limits
        r = chat_json(ROUTER_SYSTEM, f"Recent: \nPhoto attached: {has_file}\nMessage: {text}") or {}
        llm_route = str(r.get("route", "")).strip().lower()
        llm_route = llm_route if llm_route in ROUTES else "?"
        right += llm_route == expected
        print(f"  {expected:<10} | {llm_route:<10}  {text}")
    print(f"The model alone: {right}/{len(SAMPLES)}  ('?' = no answer after retries)")

    print("\nWhat the app actually does (expected | route | decided by):")
    right = 0
    for text, has_file, expected in SAMPLES:
        time.sleep(1)
        route, source = classify_with_source(text, has_file, [])
        right += route == expected
        print(f"  {expected:<10} | {route:<10} | {source:<7}  {text}")
    print(f"The app: {right}/{len(SAMPLES)}  (keyword rules alone: {sum(rules(t, f) == e for t, f, e in SAMPLES)}/{len(SAMPLES)})")

if __name__ == "__main__":
    main()

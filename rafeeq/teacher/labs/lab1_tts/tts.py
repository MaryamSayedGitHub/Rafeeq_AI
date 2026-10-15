"""
Lab 1 — Arabic Text-to-Speech with edge-tts
=============================================

Goal: Convert an Arabic script line into a .mp3 file and save it locally.
Later this becomes the tts_node inside the teacher LangGraph pipeline.

Steps:
  1. Install edge-tts
  2. Choose an Arabic voice
  3. Convert text -> speech -> save as .mp3
  4. (Optional) List all available Arabic voices

Run:
    cd D:\\NTI\\Final_Project\\rafeeq
    uv add edge-tts
    python -m rafeeq.teacher.labs.lab1_tts.tts
"""

import asyncio
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1 — Choose your voice and text
# ---------------------------------------------------------------------------
# Best Arabic voices in edge-tts:
#   ar-EG-SalmaNeural   → Egyptian female  ✅ recommended
#   ar-EG-ShakirNeural  → Egyptian male
#   ar-SA-ZariyahNeural → Saudi female
#   ar-SA-HamedNeural   → Saudi male

VOICE = "ar-EG-SalmaNeural"

# The Arabic text to convert — replace this with the slide script later
TEXT = """
مرحباً بكم في درس Python functions.
سنتعلم اليوم كيفية تعريف الدوال واستخدامها في برامجنا.
الدالة هي كتلة من الكود يمكن إعادة استخدامها في أي وقت.
"""

# ---------------------------------------------------------------------------
# Step 2 — Output path
# ---------------------------------------------------------------------------
# Saves the .mp3 next to this file under outputs/
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "lab1_output.mp3"


# ---------------------------------------------------------------------------
# Step 3 — Convert text to speech and save
# ---------------------------------------------------------------------------
async def text_to_speech(text: str, voice: str, output_path: Path) -> None:
    """
    Sends the text to Microsoft Edge TTS service and saves the audio as .mp3.
    Requires internet connection on first use.
    """
    try:
        import edge_tts
    except ImportError:
        raise SystemExit("edge-tts not installed. Run: pip install edge-tts")

    print(f"[TTS] Voice  : {voice}")
    print(f"[TTS] Output : {output_path}")
    print(f"[TTS] Text   :\n{text.strip()}\n")

    # Communicate() streams the audio from Microsoft servers
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))

    size_kb = output_path.stat().st_size / 1024
    print(f"[TTS] Done! {size_kb:.1f} KB saved to {output_path}")


# ---------------------------------------------------------------------------
# Step 4 — (Optional) List all Arabic voices
# ---------------------------------------------------------------------------
async def list_arabic_voices() -> None:
    """Prints all Arabic voices supported by edge-tts."""
    import edge_tts
    voices = await edge_tts.list_voices()
    arabic = [v for v in voices if v["Locale"].startswith("ar-")]
    print(f"\nAvailable Arabic voices ({len(arabic)}):")
    for v in arabic:
        print(f"  {v['ShortName']:35s}  {v['Gender']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    # EXPERIMENT 1: Convert the default text above to speech
    await text_to_speech(TEXT, VOICE, OUTPUT_FILE)

    # EXPERIMENT 2: Uncomment to try the male voice
    # await text_to_speech(TEXT, "ar-EG-ShakirNeural", OUTPUT_DIR / "lab1_male.mp3")

    # EXPERIMENT 3: Uncomment to see all available Arabic voices
    # await list_arabic_voices()


if __name__ == "__main__":
    asyncio.run(main())

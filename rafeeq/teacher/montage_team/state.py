"""What the montage agents share. Every agent reads some keys and writes others (listed in its docstring)."""
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict

# Fixed order. The supervisor picks a subset; the router walks this list (same idea as the lecturer graph).
AGENT_ORDER = ["analyst_agent", "cutter_agent", "transcriber_agent", "chapters_agent", "designer_agent",
               "sticker_agent", "assembler_agent", "qa_agent"]
MAX_ATTEMPTS = 2                  # the assembler may run twice: the second time without burned-in captions
ADD_KEYS = ("notes", "trace")     # keys whose updates are appended, never replaced


class MontageState(TypedDict, total=False):
    # input
    src: Path
    work: Path
    opts: dict
    # supervisor
    plan: list[str]
    reasoning: str
    # analyst
    info: dict                    # seconds, width, height, audio
    keep: list                    # [(start, end)] pieces worth keeping
    removed: float                # seconds the cut would remove
    # cutter / transcriber / chapters
    current: Path                 # the video as it is now (original, or the cut one)
    total: float
    cut: dict
    segments: list[dict]
    chapters: list[dict]
    # sticker
    stickers: list[dict]          # [{path,start,end,x,y,w,ratio}] on the cut timeline
    # designer / assembler
    intro: Path | None
    outro: Path | None
    final: Path | None
    burn: bool
    burn_off: bool                # set when burning the captions failed: render without them
    attempts: int
    # qa
    qa_ok: bool
    # output
    notes: Annotated[list[str], add]
    trace: Annotated[list[str], add]
    result: dict

"""Settings for the lecturer assistant (Member 5). Every value comes from Rafeeq's .env."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# rafeeq/teacher/lecturer/settings.py -> the repository root is three folders up
REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env")

# Generated session files (.pptx, code, quiz). Ignored by git.
OUTPUT_DIR = Path(os.getenv("TEACHER_OUTPUT_DIR", REPO_ROOT / "outputs" / "teacher")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# A teacher's saved character image (Lab 5): used as the default first frame for
# image-to-video generation so their videos keep the same "presenter" each time.
AVATAR_DIR = Path(os.getenv("TEACHER_AVATAR_DIR", REPO_ROOT / "outputs" / "avatars")).resolve()
AVATAR_DIR.mkdir(parents=True, exist_ok=True)

# Teacher library (assets): saved/generated videos, images, edited files. Ignored by git.
ASSET_DIR = Path(os.getenv("TEACHER_ASSET_DIR", REPO_ROOT / "outputs" / "assets")).resolve()
ASSET_DIR.mkdir(parents=True, exist_ok=True)

# LLM
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Tools
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Database. Empty = run without Postgres (memory is lost when the app stops).
DATABASE_URL = os.getenv("DATABASE_URL") or None

# Human-in-the-loop: how many times the lecturer may send a draft back.
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "3"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> None:
    """Call once at startup. Agents log through logging.getLogger(__name__)."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

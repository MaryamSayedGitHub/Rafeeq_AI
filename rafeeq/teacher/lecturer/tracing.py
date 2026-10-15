"""LangSmith tracing.

LangGraph traces every node, LLM call and tool call by itself once these variables are set
in .env (config.settings loads them): LANGSMITH_TRACING=true, LANGSMITH_API_KEY,
LANGSMITH_PROJECT. This module only reports the status and builds the per-run config,
so every trace can be filtered by session and lecturer.
"""

import logging
import os

log = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


def log_tracing_status() -> None:
    if tracing_enabled():
        log.info("LangSmith tracing is ON (project: %s)", os.getenv("LANGSMITH_PROJECT", "default"))
    else:
        log.info("LangSmith tracing is OFF (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)")


def run_config(session_id: str, lecturer_id: str = "") -> dict:
    """The config passed to every graph call: thread_id for memory, tags/metadata for traces."""
    return {
        "configurable": {"thread_id": session_id},
        "tags": ["rafeeq", "teacher", "lecturer-assistant"],
        "metadata": {"session_id": session_id, "lecturer_id": lecturer_id},
    }

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
    from rafeeq.observability import status

    log.info("%s (LangSmith: LANGSMITH_TRACING=true + LANGSMITH_API_KEY; Langfuse: LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY)", status())


def run_config(session_id: str, lecturer_id: str = "") -> dict:
    """The config passed to every graph call: thread_id for memory, tags/metadata for traces."""
    from rafeeq.observability import langchain_callbacks

    config = {
        "configurable": {"thread_id": session_id},
        "tags": ["rafeeq", "teacher", "lecturer-assistant"],
        "metadata": {"session_id": session_id, "lecturer_id": lecturer_id,
                     # Langfuse groups the traces of one lesson session and one teacher with these keys
                     "langfuse_session_id": session_id, "langfuse_user_id": lecturer_id or "anonymous",
                     "langfuse_tags": ["rafeeq", "teacher", "lecturer-assistant"]},
    }
    callbacks = langchain_callbacks()
    if callbacks:
        config["callbacks"] = callbacks
    return config

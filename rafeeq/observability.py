"""Tracing to LangSmith and Langfuse, both optional, for the app and for the evals.

    LangSmith  on when LANGSMITH_TRACING=true and LANGSMITH_API_KEY is set. LangChain / LangGraph runs are traced by
               themselves; `traced()` adds the direct model calls that do not go through LangChain.
    Langfuse   on when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set and the `langfuse` package is installed
               (`uv add langfuse`). LANGFUSE_BASE_URL picks the region (default https://cloud.langfuse.com).
               LangGraph runs get Langfuse's LangChain callback through `langchain_callbacks()`; direct calls through
               `traced()`.

Nothing here may break a request: every helper falls back to "no tracing" on any error.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable

log = logging.getLogger(__name__)
_MAX_TEXT = 2000


def langsmith_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def langfuse_enabled() -> bool:
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    try:
        import langfuse  # noqa: F401
    except ImportError:
        return False
    return True


def langfuse_client():
    """The Langfuse client, or None when Langfuse is off."""
    if not langfuse_enabled():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception as e:                                           # noqa: BLE001
        log.warning("langfuse: client not available (%s)", type(e).__name__)
        return None


def langchain_callbacks() -> list:
    """Callbacks to add to a LangChain / LangGraph config: Langfuse's handler when Langfuse is on, else nothing."""
    if not langfuse_enabled():
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception as e:                                           # noqa: BLE001
        log.warning("langfuse: LangChain callback not available (%s)", type(e).__name__)
        return []


def status() -> str:
    return (f"LangSmith tracing {'ON (project ' + os.getenv('LANGSMITH_PROJECT', 'default') + ')' if langsmith_enabled() else 'OFF'}"
            f" · Langfuse {'ON' if langfuse_enabled() else 'OFF'}")


def _short(value: Any) -> Any:
    """What is sent as a trace's input/output: plain values, long text cut, bytes and files named only."""
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT] + " …"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _short(v) for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        return [_short(v) for v in list(value)[:40]]
    return _short(str(value))


_LANGSMITH_TYPES = {"generation": "llm", "retriever": "retriever", "tool": "tool", "evaluator": "chain"}


def traced(name: str, as_type: str = "span") -> Callable:
    """Decorator: the call becomes a LangSmith run (when its tracing is on) and a Langfuse observation (when on).
    as_type: span | generation | retriever | tool | evaluator (Langfuse's observation types)."""
    def deco(fn: Callable) -> Callable:
        try:
            from langsmith import traceable

            ls_fn = traceable(name=name, run_type=_LANGSMITH_TYPES.get(as_type, "chain"))(fn)
        except Exception:                                            # noqa: BLE001 - langsmith missing: plain call
            ls_fn = fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            client = langfuse_client()
            if client is None:
                return ls_fn(*args, **kwargs)
            try:
                ctx = client.start_as_current_observation(name=name, as_type=as_type,
                                                          input=_short({"args": list(args), **kwargs}))
            except Exception:                                        # noqa: BLE001
                return ls_fn(*args, **kwargs)
            with ctx as obs:
                try:
                    out = ls_fn(*args, **kwargs)
                except Exception as e:
                    try:
                        obs.update(level="ERROR", status_message=f"{type(e).__name__}: {str(e)[:300]}")
                    except Exception:                                # noqa: BLE001
                        pass
                    raise
                try:
                    obs.update(output=_short(out))
                except Exception:                                    # noqa: BLE001
                    pass
                return out
        return wrapper
    return deco


def flush() -> None:
    client = langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception:                                            # noqa: BLE001
            pass

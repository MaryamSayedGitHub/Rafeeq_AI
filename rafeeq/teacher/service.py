"""Starts the lecturer assistant inside the Rafeeq app and hands it to the routes.

    with open_teacher_service() as service:   # in rafeeq/main.py's lifespan
        app.state.teacher = service

Startup never needs an LLM key, so the other members can run Rafeeq without one.
The graph is built on the first generation request; a missing key gives a clear 503 then.

Memory: Rafeeq's DATABASE_URL (Postgres) when it is reachable. If it is not, the app still
starts with in-memory storage and the teacher page shows a warning that nothing is kept
after a restart.
"""

import logging
import threading
from contextlib import ExitStack, contextmanager
from typing import Iterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from rafeeq.teacher.lecturer import settings
from rafeeq.teacher.lecturer.controller import GraphController

log = logging.getLogger(__name__)


class AssistantNotConfigured(Exception):
    """No LLM provider is set in .env."""


class TeacherService:
    def __init__(self, checkpointer, store, persistent: bool):
        self.checkpointer = checkpointer
        self.store = store
        self.persistent = persistent        # False = in-memory, lost on restart
        self._controller: GraphController | None = None
        self._lock = threading.Lock()

    @property
    def controller(self) -> GraphController:
        """The graph, built once on first use."""
        if self._controller is None:
            with self._lock:
                if self._controller is None:
                    from rafeeq.teacher.lecturer.graph import build_graph
                    from rafeeq.teacher.lecturer.llm import choose_provider

                    try:
                        provider = choose_provider()
                    except (RuntimeError, ValueError) as e:
                        raise AssistantNotConfigured(str(e)) from None
                    log.info("lecturer assistant: using LLM provider '%s'", provider)
                    self._controller = GraphController(build_graph(self.checkpointer, self.store))
        return self._controller


@contextmanager
def open_teacher_service() -> Iterator[TeacherService]:
    from rafeeq.teacher.lecturer.tracing import log_tracing_status

    log_tracing_status()
    with ExitStack() as stack:
        service = None
        if settings.DATABASE_URL:
            try:
                from rafeeq.teacher.lecturer.memory import open_memory

                checkpointer, store = stack.enter_context(open_memory())
                service = TeacherService(checkpointer, store, persistent=True)
            except Exception as e:  # Postgres down: keep the rest of Rafeeq usable
                log.error(
                    "teacher: could not use Postgres (%s: %s). Falling back to in-memory storage.",
                    type(e).__name__, str(e)[:200],
                )
        if service is None:
            service = TeacherService(InMemorySaver(), InMemoryStore(), persistent=False)
        from rafeeq.teacher import class_report

        class_report.set_store(service.store)  # student questions/quiz results live in the same store
        from rafeeq.teacher import inbox

        inbox.connect()  # the Copilot files unanswered questions here and reuses the teacher's answers
        try:
            yield service
        finally:
            class_report.set_store(None)

"""SHORT-TERM memory (checkpointer) and the shared Postgres connection pool.

The checkpointer saves the graph state after every step, keyed by thread_id (= session_id).
That is what makes the human-approval pause possible: the graph stops, the state is stored,
and a later request resumes it.

With DATABASE_URL set  -> Postgres (survives a restart).
Without DATABASE_URL   -> in memory (fine for development, lost when the app stops).
"""

import logging
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from rafeeq.teacher.lecturer.settings import DATABASE_URL

log = logging.getLogger(__name__)


@contextmanager
def open_memory() -> Iterator[tuple]:
    """Yield (checkpointer, store). Opens one pool for both and closes it at the end."""
    if not DATABASE_URL:
        log.warning("DATABASE_URL is not set: using in-memory checkpointer and store.")
        yield InMemorySaver(), InMemoryStore()
        return

    # Imported here so the app still starts without Postgres drivers configured.
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # These three connection options are required by PostgresSaver / PostgresStore.
    pool = ConnectionPool(
        conninfo=DATABASE_URL,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    try:
        pool.open(wait=True, timeout=5)
    except Exception:
        # Postgres is not reachable: stop the pool's worker threads right away,
        # otherwise they keep retrying in the background and block shutdown.
        pool.close(timeout=1)
        raise
    try:
        checkpointer = PostgresSaver(pool)
        store = PostgresStore(pool)
        checkpointer.setup()  # creates the tables the first time, does nothing afterwards
        store.setup()
        log.info("Using Postgres for short-term and long-term memory.")
        yield checkpointer, store
    finally:
        pool.close()
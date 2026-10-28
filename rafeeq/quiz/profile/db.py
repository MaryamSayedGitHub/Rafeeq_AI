"""Database session for rafeeq.quiz.

Dev default is a local SQLite file. Set M4_DATABASE_URL to use the team's
Postgres. M1 owns the real database setup and migrations; this keeps us
independent until then.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import rafeeq.quiz.quiz.models

from rafeeq.quiz.profile.models import Base
import rafeeq.quiz.quiz.models  # noqa: F401  (registers the quiz tables on Base)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.getenv("M4_DATABASE_URL", "sqlite:///m4_dev.db"))
        Base.metadata.create_all(_engine)
        _add_missing_columns(_engine)
    return _engine


# columns added after the first release; create_all does not add a column to a table that already exists
_NEW_COLUMNS = {
    "m4_answers": {
        "answer_text": "TEXT DEFAULT ''",
        "score": "FLOAT",
        "feedback": "TEXT DEFAULT ''",
        "needs_review": "BOOLEAN DEFAULT FALSE",
    },
}


def _add_missing_columns(engine) -> None:
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    for table, columns in _NEW_COLUMNS.items():
        if not insp.has_table(table):
            continue
        have = {c["name"] for c in insp.get_columns(table)}
        with engine.begin() as conn:
            for name, ddl in columns.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


def get_session():
    with Session(get_engine()) as session:
        yield session
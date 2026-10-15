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
    return _engine


def get_session():
    with Session(get_engine()) as session:
        yield session
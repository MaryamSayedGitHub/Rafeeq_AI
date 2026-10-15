"""The demo seed fills one room through the real storage, once (fake M2 topics, SQLite, dict store)."""
import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import rafeeq.quiz.quiz.models  # noqa: E402,F401
from rafeeq import demo_seed  # noqa: E402
from rafeeq.platform import service as platform  # noqa: E402
from rafeeq.quiz.profile.models import Base  # noqa: E402
from rafeeq.teacher import class_report, inbox  # noqa: E402


class Store:
    class Item:
        def __init__(self, key, value):
            self.key, self.value = key, value

    def __init__(self):
        self.data = {}

    def put(self, ns, key, value):
        self.data.setdefault(tuple(ns), {})[key] = dict(value)

    def search(self, ns, limit=10, offset=0, **kw):
        return [self.Item(k, dict(v)) for k, v in self.data.get(tuple(ns), {}).items()][offset:offset + limit]


def test_seed_fills_the_room_once(tmp_path, monkeypatch):
    store = platform.PlatformStore(tmp_path / "platform", seed_demo=True)
    platform.set_store(store)
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    monkeypatch.setattr("rafeeq.quiz.profile.db.get_engine", lambda: engine)
    monkeypatch.setattr(demo_seed, "_room_topics", lambda room_id: [("topic_01", "Slow-Wave"), ("topic_02", "Beam")])
    monkeypatch.setattr(class_report, "room_topic_names", lambda room_id: {"topic_01": "Slow-Wave", "topic_02": "Beam"})
    try:
        teacher = store.user_for_token("demo-token-mona-456")
        room_id = store.create_room(teacher, "Microwave Tubes")["room"]["room_id"]
        cs = Store()
        assert demo_seed.seed(room_id, cs) is True
        assert demo_seed.seed(room_id, cs) is False                      # once
        assert len(store.db["members"][room_id]) == len(demo_seed.STUDENTS)
        report = class_report.build_report(cs, room_id)
        assert {t["topic_id"] for t in report["topics"]} == {"topic_01", "topic_02"}
        assert report["students_at_risk"]                                # Omar is weak on purpose
        assert len(inbox.list_questions(room_id, store=cs)) == 2
    finally:
        platform.set_store(None)

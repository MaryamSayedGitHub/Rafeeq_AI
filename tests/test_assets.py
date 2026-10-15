"""Phase 2: the teacher asset library (records, ownership, versions, idempotency, file safety).

No LLM, database or network:  uv run pytest tests/test_assets.py
"""
from types import SimpleNamespace

import pytest

from rafeeq.teacher import assets
from rafeeq.teacher.lecturer import settings


class FakeStore:
    def __init__(self):
        self.data = {}

    def put(self, namespace, key, value):
        self.data[(tuple(namespace), key)] = dict(value)

    def get(self, namespace, key):
        value = self.data.get((tuple(namespace), key))
        return SimpleNamespace(key=key, value=value) if value is not None else None

    def search(self, namespace, limit=10):
        return [SimpleNamespace(key=k, value=v) for (ns, k), v in self.data.items() if ns == tuple(namespace)][:limit]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ASSET_DIR", (tmp_path / "assets").resolve())
    monkeypatch.setattr(settings, "OUTPUT_DIR", (tmp_path / "teacher").resolve())
    (tmp_path / "assets").mkdir()
    (tmp_path / "teacher").mkdir()
    return FakeStore()


def test_register_and_get_round_trip(store):
    a = assets.register_asset(store, "u_mona", kind="source_video", title="Cells", origin="external",
                              source_url="https://youtu.be/x", start_sec=12, end_sec=40, room_id="r1")
    assert a["asset_id"].startswith("a_") and a["status"] == "ready" and a["version"] == 1
    assert assets.get_asset(store, "u_mona", a["asset_id"])["source_url"] == "https://youtu.be/x"


def test_other_teachers_cannot_see_an_asset(store):
    a = assets.register_asset(store, "u_mona", kind="image", title="x", origin="upload")
    assert assets.get_asset(store, "u_other", a["asset_id"]) is None
    assert assets.list_assets(store, "u_other") == []
    assert assets.update_asset(store, "u_other", a["asset_id"], title="hijack") is None


def test_bad_input_is_refused(store):
    with pytest.raises(assets.AssetError):
        assets.register_asset(store, "u_mona", kind="exe", title="x", origin="upload")
    with pytest.raises(assets.AssetError):
        assets.register_asset(store, "u_mona", kind="image", title="x", origin="magic")
    with pytest.raises(assets.AssetError):
        assets.register_asset(store, "u_mona", kind="image", title="x", origin="upload",
                              storage_root="assets", storage_ref="../../etc/passwd")
    with pytest.raises(assets.AssetError):
        assets.register_asset(store, "u_mona", kind="image", title="x", origin="upload", storage_ref="a/b.png")


def test_idempotency_key_prevents_duplicates(store):
    kw = dict(kind="pptx", title="Deck", origin="ai_generated", session_id="s1", idempotency_key="session:s1:deck.pptx")
    first = assets.register_asset(store, "u_mona", **kw)
    again = assets.register_asset(store, "u_mona", **kw)
    assert first["asset_id"] == again["asset_id"]
    assert len(assets.list_assets(store, "u_mona")) == 1


def test_list_filters(store):
    assets.register_asset(store, "u_mona", kind="image", title="a", origin="upload", room_id="r1")
    assets.register_asset(store, "u_mona", kind="pptx", title="b", origin="ai_generated", room_id="r2", session_id="s9")
    assert [a["title"] for a in assets.list_assets(store, "u_mona", kind="pptx")] == ["b"]
    assert [a["title"] for a in assets.list_assets(store, "u_mona", room_id="r1")] == ["a"]
    assert [a["title"] for a in assets.list_assets(store, "u_mona", session_id="s9")] == ["b"]
    assert len(assets.list_assets(store, "u_mona")) == 2


def test_update_changes_only_allowed_fields(store):
    a = assets.register_asset(store, "u_mona", kind="generated_video", title="v", origin="ai_generated", status="pending")
    done = assets.update_asset(store, "u_mona", a["asset_id"], status="ready")
    assert done["status"] == "ready" and done["updated_at"] >= a["updated_at"]
    with pytest.raises(assets.AssetError):
        assets.update_asset(store, "u_mona", a["asset_id"], kind="pptx")
    with pytest.raises(assets.AssetError):
        assets.update_asset(store, "u_mona", a["asset_id"], status="weird")


def test_new_version_keeps_the_original(store):
    a = assets.register_asset(store, "u_mona", kind="pptx", title="Deck", origin="ai_generated", session_id="s1")
    b = assets.new_version(store, "u_mona", a["asset_id"], title="Deck v2")
    assert b["version"] == 2 and b["parent_asset_id"] == a["asset_id"] and b["origin"] == "edited"
    assert assets.get_asset(store, "u_mona", a["asset_id"])["title"] == "Deck"
    assert assets.new_version(store, "u_mona", "a_missing") is None


def test_save_file_stays_inside_the_asset_folder(store):
    root, ref = assets.save_file("u_mona", "a_1234abcd", b"data", "../../evil name.mp4")
    a = assets.register_asset(store, "u_mona", kind="generated_video", title="v", origin="ai_generated",
                              storage_root=root, storage_ref=ref)
    path = assets.asset_file(a)
    assert path is not None and path.read_bytes() == b"data"
    assert path.resolve().is_relative_to(settings.ASSET_DIR)
    assert ".." not in ref


def test_asset_file_is_none_when_missing_or_external(store):
    ext = assets.register_asset(store, "u_mona", kind="source_video", title="v", origin="external")
    assert assets.asset_file(ext) is None
    gone = assets.register_asset(store, "u_mona", kind="pptx", title="d", origin="ai_generated",
                                 storage_root="sessions", storage_ref="s1/deck.pptx")
    assert assets.asset_file(gone) is None


def test_session_files_are_registered_once(store):
    files = ["deck.pptx", "examples.py", "quiz.json", "quiz.md", "notes.txt"]
    first = assets.register_session_files(store, "u_mona", "s1", files, topic="Cells", room_id="r1")
    assert sorted(a["kind"] for a in first) == ["code", "pptx", "quiz", "quiz"]        # notes.txt is not a library file
    assert assets.register_session_files(store, "u_mona", "s1", files, topic="Cells", room_id="r1") == []
    deck = assets.list_assets(store, "u_mona", kind="pptx")[0]
    assert deck["storage_root"] == "sessions" and deck["storage_ref"] == "s1/deck.pptx" and deck["room_id"] == "r1"


def test_find_by_task_and_stale_pending(store):
    a = assets.register_asset(store, "u_mona", kind="generated_video", title="v", origin="ai_generated",
                              status="pending", settings_used={"task_id": "t-1"})
    assert assets.find_by_task(store, "u_mona", "t-1")["asset_id"] == a["asset_id"]
    assert assets.find_by_task(store, "u_other", "t-1") is None
    assert not assets.is_stale_pending(a)
    assert assets.is_stale_pending({**a, "created_at": "2020-01-01T00:00:00+00:00"})
    assert not assets.is_stale_pending({**a, "status": "ready", "created_at": "2020-01-01T00:00:00+00:00"})

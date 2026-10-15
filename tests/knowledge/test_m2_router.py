"""The M2 HTTP routes and the upload hook, end to end with M1's real platform service.

Needs FastAPI (a Rafeeq dependency). Uses a fake embedder and a fake PDF parser, so no model is loaded.
"""
import os
import time
from contextlib import contextmanager

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from kb_fakes import HashingEmbedder  # noqa: E402
from rafeeq.knowledge import integration, service as knowledge  # noqa: E402
from rafeeq.platform import service as platform  # noqa: E402

PAGES = [
    "Self-attention allows each token to attend to other tokens in the sequence of the input text.",
    "Convolutional neural networks are useful for image processing and for many vision tasks.",
]


def auth(login):
    return {"Authorization": f"Bearer {login['token']}"}


@contextmanager
def running(tmp_path, enabled=True):
    saved = os.environ.get("RAFEEQ_KNOWLEDGE")
    os.environ["RAFEEQ_KNOWLEDGE"] = "1" if enabled else "0"
    store = platform.PlatformStore(tmp_path / "platform", seed_demo=False)
    platform.set_store(store)
    platform.clear_upload_hooks()
    platform.clear_delete_hooks()
    integration._hook_installed = False
    knowledge.set_service(knowledge.KnowledgeService(
        tmp_path / "kb", embedder=HashingEmbedder(), status_callback=store.set_file_status,
        parse=lambda path: [{"page": i + 1, "text": t} for i, t in enumerate(PAGES)]))
    app = FastAPI()
    try:
        yield app, store, integration.install(app)
    finally:
        platform.clear_upload_hooks()
        platform.clear_delete_hooks()
        platform.set_store(None)
        knowledge.set_service(None)
        integration._hook_installed = False
        if saved is None:
            os.environ.pop("RAFEEQ_KNOWLEDGE", None)
        else:
            os.environ["RAFEEQ_KNOWLEDGE"] = saved


def wait_ready(store, user, room_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = store.list_files(user, room_id)["files"]
        if files and files[0]["status"] in ("ready", "failed"):
            return files[0]["status"]
        time.sleep(0.05)
    raise AssertionError("indexing did not finish")


def test_disabled_by_default_adds_no_routes_and_no_hook(tmp_path):
    with running(tmp_path, enabled=False) as (app, _, installed):
        assert installed is False
        assert not any(getattr(r, "path", "") == "/search" for r in app.routes)


def test_upload_then_search_topics_and_source_page(tmp_path):
    with running(tmp_path) as (app, store, installed):
        assert installed is True
        teacher = store.register("Teacher", "t@example.com", "secret1", "teacher")
        t_user = store.user_for_token(teacher["token"])
        room = store.create_room(t_user, "Biology")
        room_id, code = room["room"]["room_id"], room["join_code"]
        student = store.register("Student", "s@example.com", "secret1", "student", code)
        outsider = store.register("Other", "o@example.com", "secret1", "student")

        up = store.add_file(t_user, room_id, "course.pdf", b"%PDF-1.4 fake")
        assert up["status"] in ("uploaded", "indexing", "ready")
        assert wait_ready(store, t_user, room_id) == "ready"

        client = TestClient(app)
        r = client.post("/search", json={"room_id": room_id, "query": "self-attention", "top_k": 2}, headers=auth(student))
        assert r.status_code == 200
        chunks = r.json()["chunks"]
        assert chunks[0]["file_name"] == "course.pdf" and chunks[0]["page"] == 1
        assert set(chunks[0]) == {"file_name", "page", "text", "topic_id"}

        t = client.get(f"/rooms/{room_id}/topics", headers=auth(teacher))
        assert t.status_code == 200 and t.json()["topics"]

        p = client.get("/sources/page", params={"room_id": room_id, "file_name": "course.pdf", "page": 2}, headers=auth(student))
        assert p.status_code == 200 and p.json()["text"] == PAGES[1]
        assert client.get("/sources/page", params={"room_id": room_id, "file_name": "course.pdf", "page": 99}, headers=auth(student)).status_code == 404

        # access control: no token -> 401, a user outside the room -> 403
        assert client.post("/search", json={"room_id": room_id, "query": "x"}).status_code == 401
        assert client.post("/search", json={"room_id": room_id, "query": "x"}, headers=auth(outsider)).status_code == 403
        assert client.get(f"/rooms/{room_id}/topics", headers=auth(outsider)).status_code == 403

        # contract limits: top_k 1..20, unknown fields refused
        assert client.post("/search", json={"room_id": room_id, "query": "x", "top_k": 21}, headers=auth(student)).status_code == 422
        assert client.post("/search", json={"room_id": room_id, "query": "x", "extra": 1}, headers=auth(student)).status_code == 422


def test_a_non_pdf_upload_ends_as_failed_not_ready(tmp_path):
    with running(tmp_path) as (app, store, _):
        teacher = store.register("Teacher", "t@example.com", "secret1", "teacher")
        t_user = store.user_for_token(teacher["token"])
        room_id = store.create_room(t_user, "Biology")["room"]["room_id"]
        store.add_file(t_user, room_id, "slides.pptx", b"not a pdf")
        assert wait_ready(store, t_user, room_id) == "failed"

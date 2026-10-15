"""Phase 1: the /teacher JSON API knows who is calling when a Bearer token is sent.

No header = legacy mode (old /teacher pages, sample tests) and keeps working unchanged.
    uv run pytest tests/test_teacher_auth.py
"""
import pytest

MONA = {"Authorization": "Bearer demo-token-mona-456"}   # teacher u_mona (demo account)
SARA = {"Authorization": "Bearer demo-token-sara-123"}   # student u_sara (demo account)


@pytest.fixture
def client(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from fastapi.testclient import TestClient

    from rafeeq.platform import service as platform

    monkeypatch.setattr("rafeeq.teacher.lecturer.settings.DATABASE_URL", None)  # in-memory store
    for name in ("ASSET_DIR", "OUTPUT_DIR"):  # never touch the real outputs/ folders
        folder = (tmp_path / name).resolve()
        folder.mkdir()
        monkeypatch.setattr(f"rafeeq.teacher.lecturer.settings.{name}", folder)
    avatars = (tmp_path / "AVATARS").resolve()
    avatars.mkdir()
    monkeypatch.setattr("rafeeq.teacher.routes.AVATAR_DIR", avatars)
    platform.set_store(platform.PlatformStore(tmp_path / "platform"))
    from rafeeq.main import app

    with TestClient(app) as c:
        yield c
    platform.set_store(None)


def test_no_header_keeps_the_legacy_behaviour(client):
    assert client.get("/teacher/profile", params={"teacher_id": "nobody"}).status_code == 404
    assert client.get("/teacher/sessions", params={"teacher_id": "nobody"}).status_code == 200


def test_unknown_token_is_401(client):
    res = client.get("/teacher/sessions", params={"teacher_id": "u_mona"},
                     headers={"Authorization": "Bearer nope"})
    assert res.status_code == 401


def test_a_student_token_is_refused(client):
    res = client.get("/teacher/sessions", params={"teacher_id": "u_sara"}, headers=SARA)
    assert res.status_code == 403


def test_a_teacher_cannot_name_another_teacher(client):
    for path in ("/teacher/sessions", "/teacher/profile"):
        res = client.get(path, params={"teacher_id": "someone-else"}, headers=MONA)
        assert res.status_code == 403, path


def test_a_teacher_can_use_their_own_id(client):
    assert client.get("/teacher/sessions", params={"teacher_id": "u_mona"}, headers=MONA).status_code == 200


def test_profile_put_is_bound_to_the_token(client):
    body = {"teacher_id": "someone-else", "name": "Mona Test"}
    assert client.put("/teacher/profile", json=body, headers=MONA).status_code == 403
    ok = client.put("/teacher/profile", json={**body, "teacher_id": "u_mona"}, headers=MONA)
    assert ok.status_code == 200 and ok.json()["teacher_id"] == "u_mona"


def test_avatar_upload_is_bound_to_the_token(client):
    files = {"image": ("me.png", b"\x89PNG\r\n", "image/png")}
    res = client.post("/teacher/avatar", data={"teacher_id": "someone-else"}, files=files, headers=MONA)
    assert res.status_code == 403


# ------------------------------------------------------------------ asset library API
VIDEO = {"title": "Cell division", "source_url": "https://www.youtube.com/watch?v=abc123",
         "start_sec": 75, "end_sec": 130, "room_id": "r_1"}


def test_assets_always_need_a_teacher_token(client):
    assert client.get("/teacher/assets").status_code == 401
    assert client.post("/teacher/assets", json=VIDEO).status_code == 401
    assert client.get("/teacher/assets", headers=SARA).status_code == 403


def test_save_a_video_moment_and_list_it(client):
    res = client.post("/teacher/assets", json=VIDEO, headers=MONA)
    assert res.status_code == 201
    asset = res.json()
    assert asset["origin"] == "external" and asset["kind"] == "source_video" and asset["file_url"] is None

    listed = client.get("/teacher/assets", params={"room_id": "r_1"}, headers=MONA).json()["assets"]
    assert [a["asset_id"] for a in listed] == [asset["asset_id"]]
    assert client.get(f"/teacher/assets/{asset['asset_id']}", headers=MONA).json()["title"] == "Cell division"


def test_saving_twice_does_not_duplicate(client):
    first = client.post("/teacher/assets", json=VIDEO, headers=MONA).json()
    second = client.post("/teacher/assets", json=VIDEO, headers=MONA).json()
    assert first["asset_id"] == second["asset_id"]
    assert len(client.get("/teacher/assets", headers=MONA).json()["assets"]) == 1


def test_bad_asset_input_is_422(client):
    assert client.post("/teacher/assets", json={**VIDEO, "source_url": "javascript:alert(1)"}, headers=MONA).status_code == 422
    assert client.post("/teacher/assets", json={**VIDEO, "start_sec": 50, "end_sec": 10}, headers=MONA).status_code == 422


def test_missing_asset_or_file_is_404(client):
    assert client.get("/teacher/assets/a_nope", headers=MONA).status_code == 404
    asset = client.post("/teacher/assets", json=VIDEO, headers=MONA).json()
    assert client.get(f"/teacher/assets/{asset['asset_id']}/file", headers=MONA).status_code == 404


# ------------------------------------------------------------------ Phase 3: videos kept in the library
@pytest.fixture
def provider(monkeypatch):
    """A fake video provider: counts paid submissions, answers polls from `state`."""
    from rafeeq.teacher.lecturer import video_gen

    state = {"submits": 0, "polls": 0, "poll": {"status": "RUNNING", "video_url": None, "error": None},
             "download": b"MP4-DATA"}

    def submit(prompt, image_bytes=None, image_filename="image.jpg", duration=5, resolution="720P", audio=False):
        state["submits"] += 1
        return {"ok": True, "task_id": f"task-{state['submits']}", "model": "wan-test"}

    def poll(task_id):
        state["polls"] += 1
        return dict(state["poll"])

    def download(url):
        if isinstance(state["download"], Exception):
            raise state["download"]
        return state["download"]

    monkeypatch.setattr(video_gen, "submit_task", submit)
    monkeypatch.setattr(video_gen, "poll_task", poll)
    monkeypatch.setattr(video_gen, "download_video", download)
    return state


GEN = {"prompt": "A teacher explains mitosis", "duration": "5"}


def test_generating_twice_pays_once_and_returns_the_same_job(client, provider):
    first = client.post("/teacher/videos/generate", data=GEN, headers=MONA).json()
    again = client.post("/teacher/videos/generate", data=GEN, headers=MONA).json()
    assert provider["submits"] == 1
    assert first["task_id"] == again["task_id"] and first["asset_id"] == again["asset_id"]
    listed = client.get("/teacher/assets", headers=MONA).json()["assets"]
    assert [(a["kind"], a["status"], a["origin"]) for a in listed] == [("generated_video", "pending", "ai_generated")]
    assert listed[0]["settings"]["prompt"] == GEN["prompt"] and listed[0]["provider"] == "qwencloud"


def test_without_a_token_nothing_is_saved_but_it_still_works(client, provider):
    res = client.post("/teacher/videos/generate", data=GEN)
    assert res.status_code == 200 and "asset_id" not in res.json()
    assert provider["submits"] == 1


def test_a_finished_video_is_downloaded_and_kept(client, provider):
    job = client.post("/teacher/videos/generate", data=GEN, headers=MONA).json()
    provider["poll"] = {"status": "SUCCEEDED", "video_url": "https://cdn.example/v.mp4", "error": None}
    done = client.get(f"/teacher/videos/generate/{job['task_id']}", headers=MONA).json()
    assert done["status"] == "SUCCEEDED" and done["file_url"] == f"/teacher/assets/{job['asset_id']}/file"

    asset = client.get(f"/teacher/assets/{job['asset_id']}", headers=MONA).json()
    assert asset["status"] == "ready" and asset["file_url"]
    file = client.get(asset["file_url"], headers=MONA)
    assert file.status_code == 200 and file.content == b"MP4-DATA"
    # the file is only served to its owner (a request without a token is refused)
    assert client.get(asset["file_url"]).status_code == 401

    polls = provider["polls"]
    again = client.get(f"/teacher/videos/generate/{job['task_id']}", headers=MONA).json()
    assert again["file_url"] and provider["polls"] == polls      # answered from our copy: no provider call
    assert client.post("/teacher/videos/generate", data=GEN, headers=MONA).json()["asset_id"] == job["asset_id"]
    assert provider["submits"] == 1


def test_a_failed_job_is_recorded_and_a_retry_is_allowed(client, provider):
    job = client.post("/teacher/videos/generate", data=GEN, headers=MONA).json()
    provider["poll"] = {"status": "FAILED", "video_url": None, "error": "content check failed"}
    client.get(f"/teacher/videos/generate/{job['task_id']}", headers=MONA)
    asset = client.get(f"/teacher/assets/{job['asset_id']}", headers=MONA).json()
    assert asset["status"] == "failed" and "content check" in asset["error"]

    retry = client.post("/teacher/videos/generate", data=GEN, headers=MONA).json()
    assert provider["submits"] == 2 and retry["asset_id"] == job["asset_id"] and retry["task_id"] != job["task_id"]
    assert client.get(f"/teacher/assets/{job['asset_id']}", headers=MONA).json()["status"] == "pending"


def test_if_the_download_fails_the_provider_link_is_still_returned(client, provider):
    job = client.post("/teacher/videos/generate", data=GEN, headers=MONA).json()
    provider["poll"] = {"status": "SUCCEEDED", "video_url": "https://cdn.example/v.mp4", "error": None}
    provider["download"] = OSError("network down")
    done = client.get(f"/teacher/videos/generate/{job['task_id']}", headers=MONA).json()
    assert done["video_url"] == "https://cdn.example/v.mp4" and "file_url" not in done
    asset = client.get(f"/teacher/assets/{job['asset_id']}", headers=MONA).json()
    assert asset["status"] == "failed" and "could not be saved" in asset["error"]


def test_listing_the_library_picks_up_older_sessions_once(client, tmp_path):
    from rafeeq.teacher.lecturer import settings
    from rafeeq.teacher.lecturer.long_term_store import save_session

    folder = settings.OUTPUT_DIR / "old-session"
    folder.mkdir()
    (folder / "intro.pptx").write_bytes(b"PPTX")
    (folder / "quiz.json").write_text("[]")
    store = client.app.state.teacher.store
    save_session(store, {"lecturer_id": "u_mona", "session_id": "old-session", "topic": "Cells", "room_id": "r_1"},
                 ["Cells"], [], files=["intro.pptx", "quiz.json", "gone.pptx"])

    first = client.get("/teacher/assets", headers=MONA).json()["assets"]
    assert sorted(a["kind"] for a in first) == ["pptx", "quiz"]          # gone.pptx is not on disk: skipped
    assert all(a["origin"] == "ai_generated" and a["file_url"] and a["room_id"] == "r_1" for a in first)
    second = client.get("/teacher/assets", headers=MONA).json()["assets"]
    assert len(second) == 2
    deck = next(a for a in second if a["kind"] == "pptx")
    assert client.get(deck["file_url"], headers=MONA).content == b"PPTX"

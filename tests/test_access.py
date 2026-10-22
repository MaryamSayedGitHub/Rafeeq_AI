"""The production access rule (RAFEEQ_ALLOW_ANONYMOUS unset): no anonymous model calls, no reading another room,
no acting as another student, teacher-only data for the room's teacher. The other tests run in development mode."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rafeeq.platform import service
from rafeeq.platform.router import router as platform_router
from rafeeq.quiz.router import router as quiz_router, bearer_user_id
from rafeeq.quiz.deps import current_user_id
from rafeeq.teacher.routes import router as teacher_router
from rafeeq.tutor.events import current_user_id as tutor_user
from rafeeq.tutor.rooms_api import router as rooms_api_router
from rafeeq.tutor.router import router as tutor_router


@pytest.fixture()
def world(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")
    from rafeeq.quiz import deps
    from rafeeq.quiz.profile import db
    from rafeeq.teacher.service import open_teacher_service

    monkeypatch.setenv("RAFEEQ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAFEEQ_ALLOW_ANONYMOUS", "0")
    monkeypatch.setenv("M4_DATABASE_URL", f"sqlite:///{tmp_path / 'm4.db'}")
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr("rafeeq.teacher.lecturer.settings.DATABASE_URL", None)   # in-memory teacher store
    service.set_store(None)
    app = FastAPI()
    for r in (platform_router, quiz_router, teacher_router, rooms_api_router, tutor_router):
        app.include_router(r)
    app.dependency_overrides[current_user_id] = bearer_user_id
    app.dependency_overrides[deps.get_llm] = lambda: (lambda prompt: "{}")      # no model: the access check comes first
    app.dependency_overrides[deps.get_verifier] = lambda: None
    stack = open_teacher_service()
    app.state.teacher = stack.__enter__()
    client = TestClient(app, raise_server_exceptions=False)

    def register(name, role, code=None):
        body = {"name": name, "email": f"{name}@t.test", "password": "secret1", "role": role, **({"join_code": code} if code else {})}
        return client.post("/auth/register", json=body).json()

    teacher = register("teach", "teacher")
    room = client.post("/rooms", json={"name": "Physics"}, headers=auth(teacher)).json()
    student = register("stud", "student", room["join_code"])
    outsider = register("out", "student")
    yield client, room["room"]["room_id"], teacher, student, outsider
    stack.__exit__(None, None, None)
    service.set_store(None)


def auth(user):
    return {"Authorization": f"Bearer {user['token']}"}


def test_anonymous_callers_cannot_use_the_copilot_or_snap(world):
    client, rid, *_ = world
    assert client.post("/copilot/message", json={"room_id": rid, "text": "what is a helix?"}).status_code == 401
    assert client.post("/snap", files={"photo": ("a.png", b"x", "image/png")}).status_code == 401


def test_a_student_of_another_room_is_refused_everywhere(world):
    client, rid, _, _, outsider = world
    h = auth(outsider)
    assert client.post("/copilot/message", json={"room_id": rid, "text": "hi"}, headers=h).status_code in (403, 404)
    assert client.post(f"/rooms/{rid}/tutor/act", json={"action": "auto", "text": "hi"}, headers=h).status_code == 403
    assert client.post(f"/rooms/{rid}/ask", json={"question": "hi"}, headers=h).status_code in (403, 422)
    assert client.post("/tutor/answer", json={"room_id": rid, "question": "hi"}, headers=h).status_code == 403
    assert client.post("/quiz", json={"room_id": rid, "count": 3}, headers=h).status_code == 403
    assert client.get(f"/next-step?room_id={rid}", headers=h).status_code == 403


def test_m3_routes_need_a_login(world):
    client, rid, *_ = world
    assert client.post(f"/rooms/{rid}/tutor/act", json={"action": "auto", "text": "hi"}).status_code == 401
    assert client.get(f"/rooms/{rid}/questions").status_code == 401


def test_teacher_data_is_for_the_rooms_teacher_only(world):
    client, rid, teacher, student, _ = world
    for path in (f"/teacher/rooms/{rid}/report", f"/teacher/rooms/{rid}/insights", f"/rooms/{rid}/instructor/frequent",
                 f"/rooms/{rid}/class-profile", f"/rooms/{rid}/wrong-answers"):
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers=auth(student)).status_code == 403, path
    assert client.get(f"/teacher/rooms/{rid}/report", headers=auth(teacher)).status_code != 403


def test_x_user_id_is_not_trusted(world):
    assert tutor_user({"x-user-id": "u_someone"}) is None


def test_demo_accounts_switched_off_stop_working_even_when_already_saved(world, monkeypatch):
    client, *_ = world
    assert tutor_user({"authorization": "Bearer demo-token-sara-123"}) == "u_sara"      # demo accounts on (the default)
    monkeypatch.setenv("RAFEEQ_DEMO_ACCOUNTS", "0")
    assert tutor_user({"authorization": "Bearer demo-token-sara-123"}) is None
    assert client.post("/auth/login", json={"email": "sara@rafeeq.demo", "password": service.DEMO_PASSWORD}).status_code == 401


def test_a_student_cannot_send_events_for_someone_else(world):
    client, rid, _, student, _ = world
    ev = {"user_id": "u_other", "room_id": rid, "topic_id": "t1", "kind": "question", "text": "q", "time": "2026-10-22T10:00:00Z"}
    assert client.post("/events", json=ev).status_code == 401
    assert client.post("/events", json=ev, headers=auth(student)).status_code == 403
    assert client.post("/events", json={**ev, "user_id": student["user_id"]}, headers=auth(student)).status_code == 204


def test_teacher_api_needs_a_login_to_name_a_teacher(world):
    client, *_ = world
    assert client.get("/teacher/sessions", params={"teacher_id": "teach"}).status_code == 401

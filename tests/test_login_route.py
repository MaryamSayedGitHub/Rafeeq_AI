import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rafeeq.platform import service
from rafeeq.platform.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("RAFEEQ_DATA_DIR", str(tmp_path))
    service.set_store(None)
    yield
    service.set_store(None)


def test_login_returns_the_agreed_fields():
    r = client.post("/auth/login", json={"email": "sara@rafeeq.demo", "password": service.DEMO_PASSWORD})
    assert r.status_code == 200
    fields = {"token", "user_id", "name", "role"}
    assert set(r.json()) == fields


def test_wrong_password_is_401():
    r = client.post("/auth/login", json={"email": "sara@rafeeq.demo", "password": "nope"})
    assert r.status_code == 401


def test_login_needs_email_and_password():
    assert client.post("/auth/login", json={}).status_code == 422

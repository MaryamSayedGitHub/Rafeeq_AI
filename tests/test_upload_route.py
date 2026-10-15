from fastapi import FastAPI
from fastapi.testclient import TestClient

from rafeeq.platform.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_upload_file_returns_the_agreed_fields():
    r = client.post(
        "/rooms/r_bio101/files",
        files={"file": ("notes.pdf", b"fake pdf content", "application/pdf")},
    )
    assert r.status_code == 202
    data = r.json()
    assert set(data) == {"file_id", "status"}
    assert data["status"] in ("uploaded", "indexing", "ready", "failed")

def test_deleting_a_file_needs_a_login():
    r = client.delete("/rooms/r_bio101/files/f_1")
    assert r.status_code == 401

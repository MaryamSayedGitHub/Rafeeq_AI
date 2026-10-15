from fastapi import FastAPI
from fastapi.testclient import TestClient

from rafeeq.platform.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_list_my_rooms_returns_the_agreed_fields():
    r = client.get("/rooms")
    assert r.status_code == 200

    data = r.json()
    assert set(data) == {"rooms"}

    room_fields = {"room_id", "name", "kind", "files_ready"}
    for room in data["rooms"]:
        assert set(room) == room_fields
        assert room["kind"] in ("course", "personal")
        assert room["files_ready"] >= 0
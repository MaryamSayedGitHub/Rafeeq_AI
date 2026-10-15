from fastapi import FastAPI
from fastapi.testclient import TestClient

from rafeeq.platform.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_copilot_message_returns_reply():
    response = client.post(
        "/copilot/message",
        json={"room_id": "r_1", "text": "What does the mitochondria do?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "ask"
    assert body["payload"]["grounded"] is True
    assert body["payload"]["citations"][0]["page"] == 12


def test_copilot_message_needs_text():
    response = client.post("/copilot/message", json={"room_id": "r_1"})
    assert response.status_code == 422

def test_copilot_message_rejects_unknown_field():
    response = client.post(
        "/copilot/message",
        json={"room_id": "r_1", "text": "hi", "extra": "x"},
    )
    assert response.status_code == 422
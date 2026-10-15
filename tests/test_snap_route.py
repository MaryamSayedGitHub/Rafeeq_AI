import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rafeeq.copilot import clients
from rafeeq.platform import service
from rafeeq.platform.router import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def png_bytes() -> bytes:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    img = np.full((120, 240, 3), 255, np.uint8)
    cv2.putText(img, "2x+3=11", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("RAFEEQ_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    service.set_store(None)
    clients.disconnect("read_image")
    yield
    clients.disconnect("read_image")
    service.set_store(None)


def test_snap_returns_result():
    # M3's reader is not connected in tests, so plug in a fake one. The route must return what it read.
    clients.connect("read_image", lambda data: {"type": "student_work", "language": "ar",
                                                "text": "2x + 6 = 14", "confidence": 0.91})
    response = client.post(
        "/snap",
        files={"photo": ("page.png", png_bytes(), "image/png")},
        data={"room_id": "r_1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "student_work"
    assert body["language"] == "ar"
    assert body["confidence"] == 0.91


def test_snap_without_a_reader_is_503_not_a_sample():
    response = client.post("/snap", files={"photo": ("page.png", png_bytes(), "image/png")})
    assert response.status_code == 503


def test_snap_rejects_a_file_that_is_not_an_image():
    pytest.importorskip("cv2")
    response = client.post("/snap", files={"photo": ("page.jpg", b"fake-image", "image/jpeg")})
    assert response.status_code == 422


def test_snap_needs_a_photo():
    response = client.post("/snap")
    assert response.status_code == 422

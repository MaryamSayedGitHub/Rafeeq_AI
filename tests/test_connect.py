"""M3 plugged into M1's Copilot and /snap (rafeeq/tutor/connect.py): the real flows, with only the model, M2 and M4 replaced."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_member3 import FakeReader, _photo, reader, world  # noqa: F401  (fixtures)

from rafeeq.copilot import clients, supervisor
from rafeeq.main import app
from rafeeq.snap import service as snap_service
from rafeeq.tutor import connect


@pytest.fixture
def connected(world, monkeypatch, tmp_path):
    monkeypatch.setenv("RAFEEQ_DATA_DIR", str(tmp_path / "m1"))
    connect.disconnect_m3()
    clients.disconnect("record_event")                          # the app import connects M4's own hook: here M3's hook must deliver (and be seen)
    connect.connect_m3(force=True)
    yield world
    connect.disconnect_m3()


def test_nothing_is_connected_when_switched_off(monkeypatch):
    monkeypatch.setenv("RAFEEQ_CONNECT_M3", "0")
    connect.disconnect_m3()                                     # importing the app already connected M3
    assert connect.connect_m3() is False and not clients.is_connected("tutor_answer")


def test_the_copilot_answers_from_m3_with_citations_a_topic_and_the_m4_event(connected):
    _, sent = connected
    out = supervisor.handle_message("r_bio101", "ما وظيفة الميتوكوندريا؟", None, "auto", "u_sara")
    assert out["route"] == "ask" and "sample" not in out["payload"]                         # live M3, not the contract sample
    assert out["payload"]["grounded"] and out["payload"]["citations"] == [{"file_name": "biology_ch3.pdf", "page": 12}] and out["payload"]["topic_id"] == "t_energy"
    assert [(e["user_id"], e["topic_id"], e["kind"]) for e in sent] == [("u_sara", "t_energy", "question")]     # the Copilot's event, delivered by M3's hook


def test_a_math_problem_gets_the_hints_panel_not_a_citation(connected, monkeypatch):
    out = supervisor.handle_message("r", "2x + 6 = 14", None, "auto", "u_sara")
    p = out["payload"]
    assert out["route"] == "ask" and p["grounded"] is True and p["citations"] == [] and p["help_key"] and p["problem_kind"] == "math"
    from rafeeq.tutor import help as tutor_help
    assert len(tutor_help.get(p["help_key"])["hints"]) == 3


def test_the_snap_route_reads_through_m3_and_check_work_boxes_the_real_line(connected, reader, tmp_path, monkeypatch):
    _, sent = connected
    reader(["2 + 3 = 5", "4 × 2 = 9"])
    client = TestClient(app)
    data = _photo()
    r = client.post("/snap", files={"photo": ("w.png", data, "image/png")}, data={"room_id": "r_bio101"})
    assert r.status_code == 200, r.text
    snap = r.json()
    assert snap["text"] == "2 + 3 = 5\n4 × 2 = 9" and snap["language"] == "en" and snap["type"] == "student_work" and snap["file_id"].startswith("snap_")

    out = supervisor.handle_message("r_bio101", snap["text"], snap["file_id"], "check_work", "u_sara")
    check = out["payload"]
    assert out["route"] == "check_work" and "sample" not in check
    assert [s["correct"] for s in check["steps"]] == [True, False] and check["hint"] and check["boxes"]
    saved = snap_service.load_lines(next(iter((tmp_path / "snap").glob("f_*.json"))).stem)
    assert check["boxes"] == [saved["boxes"][1]]                                           # the REAL box of the wrong line, not a guess by position
    shown = cv2.imread(str(tmp_path / "m1" / "snaps" / f"{snap['file_id']}.jpg"))
    assert shown is not None                                                               # M1's UI draws the box on this page


def test_a_photo_that_is_not_an_image_is_still_a_422_for_m1_ui(connected):
    r = TestClient(app).post("/snap", files={"photo": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_the_reader_being_down_is_a_503_not_a_crash(connected, monkeypatch):
    def boom(name):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(snap_service, "get_reader", boom)
    r = TestClient(app).post("/snap", files={"photo": ("w.png", _photo(), "image/png")})
    assert r.status_code == 503 and "reader" in r.json()["detail"].lower()


def test_unreliable_handwriting_goes_through_the_copilot_as_a_question(connected, reader):
    reader(["3x = 12", "x = [?]"])
    client = TestClient(app)
    snap = client.post("/snap", files={"photo": ("w.png", _photo(("3x = 12", "x = 4")), "image/png")}).json()
    out = supervisor.handle_message("r", snap["text"], snap["file_id"], "check_work", None)
    assert out["payload"]["steps"][1]["correct"] is False and out["payload"]["boxes"] == []        # asked, never marked wrong
    assert "type it" in out["payload"]["hint"] or "read" in out["payload"]["hint"].lower()

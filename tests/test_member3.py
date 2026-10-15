"""Member 3 end to end: the agent (tag_topic -> M2 answer -> log -> M4 event, and Level 2 -> search -> study file -> reopen), the room API
and its contract, handwriting (quality -> lines -> reader -> M2 -> statuses -> one hint each -> M4 event), the anonymous instructor list."""
import base64
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

from rafeeq.samples import load_sample
from rafeeq.snap import readers as snap_readers
from rafeeq.snap import service as snap_service
from rafeeq.tutor import agent, anonymous, events, m2, m4, study_store, topics, web_search
from rafeeq.tutor import llm as llm_mod
from rafeeq.tutor.grading import one_hint
from rafeeq.tutor.rooms_api import router

SPEC = yaml.safe_load((Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))
app = FastAPI()
app.include_router(router)
SARA = {"Authorization": "Bearer demo-token-sara-123"}


def assert_contract(schema, data):
    v = Draft202012Validator({"$ref": f"#/components/schemas/{schema}", "components": SPEC["components"]}, format_checker=FormatChecker())
    errors = [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(data)]
    assert not errors, errors


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Local data, M2 / M4 down (samples), no LLM, no web. `sent` collects the events that would reach M4."""
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path / "study"))
    monkeypatch.setenv("SNAP_DATA_DIR", str(tmp_path / "snap"))
    def down(*a, **k):
        raise ConnectionError("down")

    sent = []

    class Ok:
        def raise_for_status(self): pass

    def post(url, json=None, timeout=None, headers=None, **k):    # requests.post is one shared function: M4's /events answers, everything else is down
        if str(url).endswith("/events"):
            sent.append(json)
            return Ok()
        raise ConnectionError("down")

    monkeypatch.setattr(m2.requests, "post", post)
    monkeypatch.setattr(m2.requests, "get", down)
    monkeypatch.setattr(llm_mod, "ask_json", lambda prompt, max_tokens=None: None)
    return TestClient(app), sent


# ----------------------------------------------------------------------------- core: ask, log, M4 event
def test_level1_answer_has_citations_topic_log_and_the_m4_event(world):
    client, sent = world
    r = client.post("/rooms/r_bio101/ask", json={"question": "ما وظيفة الميتوكوندريا؟"}, headers=SARA)
    assert r.status_code == 200
    body = r.json()
    assert_contract("RoomAskResponse", body)
    assert body["grounded"] and body["citations"] == [{"file_name": "biology_ch3.pdf", "page": 12}] and body["topic_id"] == "t_energy"
    assert body["offer_level2"] is False and body["event_delivered"] is True
    assert [(e["user_id"], e["room_id"], e["topic_id"], e["kind"]) for e in sent] == [("u_sara", "r_bio101", "t_energy", "question")]
    logged = client.get("/rooms/r_bio101/questions", headers=SARA).json()
    assert_contract("LoggedQuestions", logged)
    assert logged[0]["question"] == "ما وظيفة الميتوكوندريا؟" and logged[0]["topic_id"] == "t_energy" and logged[0]["level"] == 1


def test_not_found_offers_level2_by_itself(world, monkeypatch):
    client, _ = world
    monkeypatch.setattr(llm_mod, "ask_json", lambda prompt, max_tokens=None: {"grounded": False, "answer": "", "steps": [], "used": []})
    body = client.post("/rooms/r/ask", json={"question": "who won the 1998 world cup?", "language": "en"}, headers=SARA).json()
    assert_contract("RoomAskResponse", body)
    assert body["grounded"] is False and body["citations"] == [] and body["offer_level2"] is True and body["quiz"] == []
    assert "not find" in body["answer"].lower()


def test_without_a_signed_in_student_nothing_is_sent_to_m4_and_my_questions_is_401(world):
    client, sent = world
    body = client.post("/rooms/r/ask", json={"question": "ما وظيفة الميتوكوندريا؟"}).json()
    assert sent == [] and body["event_delivered"] is False
    assert client.get("/rooms/r/questions").status_code == 401


def test_request_validation(world):
    client, _ = world
    for bad in ({}, {"question": ""}, {"question": "q", "level": 3}, {"question": "q", "extra": 1}, {"question": "q", "pages": 99}):
        assert client.post("/rooms/r/ask", json=bad).status_code == 422, bad


def test_tag_topic_order(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda prompt, max_tokens=None: None)
    tops = [{"topic_id": "t_a", "name": "Cell energy"}, {"topic_id": "t_b", "name": "Genetics"}]
    assert topics.tag_topic("r", "x", [{"topic_id": "t_b"}], tops) == "t_b"                      # the passages know
    assert topics.tag_topic("r", "how does genetics work", None, tops) == "t_b"                  # words of the topic name
    assert topics.tag_topic("r", "zzz", None, []) == "other" and topics.tag_topic("r", "zzz", None, tops) == "other"


# ----------------------------------------------------------------------------- Level 2: search -> study file -> save -> reopen
def test_level2_searches_writes_saves_and_the_file_can_be_reopened(world, monkeypatch):
    client, sent = world
    pages = [{"title": f"Source {i}", "url": f"https://example.org/{i}", "content": "A list in Python is an ordered, changeable collection of items. " * 3, "score": 1 - i / 10}
             for i in range(8)]
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: pages)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    body = client.post("/rooms/r_bio101/ask", json={"question": "what is a python list", "level": 2}, headers=SARA).json()
    assert_contract("RoomAskResponse", body)
    sf = body["study_file"]
    assert body["level"] == 2 and sf and len(sf["sources"]) == 5 and all(s["url"].startswith("https://") for s in sf["sources"])
    assert {"Overview", "Key ideas", "Worked example", "Sources"} <= {l.lstrip("# ").strip() for l in sf["markdown"].splitlines() if l.startswith("#")}
    saved = study_store.get_study_file(sf["id"])
    assert saved and "https://example.org/0" in saved["markdown"]                                  # reopened from the saved file
    assert any(e["text"] == "what is a python list" for e in sent)


def test_level2_after_an_ungrounded_level1_updates_the_same_question(world, monkeypatch):
    client, _ = world
    monkeypatch.setenv("TUTOR_SAMPLE_FALLBACK", "0")                                                # M2 has nothing: Level 1 is not grounded
    first = client.post("/rooms/r/ask", json={"question": "python list", "language": "en"}, headers=SARA).json()
    assert first["offer_level2"] is True
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: [{"title": "T", "url": "https://e.org/x", "content": "A list holds items in order. " * 6, "score": 1}])
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    second = client.post("/rooms/r/ask", json={"question": "python list", "level": 2, "parent_question_id": first["question_id"]}, headers=SARA).json()
    assert second["question_id"] == first["question_id"]
    rows = client.get("/rooms/r/questions", headers=SARA).json()
    assert len(rows) == 1 and rows[0]["level"] == 2 and rows[0]["study_file_id"] == second["study_file"]["id"]


def test_the_search_cache_protects_the_quota(world, monkeypatch):
    client, _ = world
    calls = []
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: calls.append(q) or [{"title": "T", "url": "https://e.org/a", "content": "word " * 30, "score": 1}])
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    web_search.cached_search("same query")
    web_search.cached_search("same query")
    assert calls == ["same query"] and study_store.usage_get()["cache_hits"] == 1


# ----------------------------------------------------------------------------- anonymous instructor questions
def test_instructor_questions_have_no_names_or_ids(world):
    client, _ = world
    client.post("/rooms/r_bio101/ask", json={"question": "I am sara@mail.com call 01012345678 what is ATP? see https://x.org"}, headers=SARA)
    rows = client.get("/rooms/r_bio101/instructor/questions").json()
    assert_contract("InstructorQuestions", rows)
    text = str(rows)
    for private in ("u_sara", "sara@mail.com", "01012345678", "https://x.org", "Sara", "سارة"):
        assert private not in text
    assert set(rows[0]) == {"public_id", "date", "topic_id", "level", "grounded", "question"} and len(rows[0]["public_id"]) == 32
    csv = client.get("/rooms/r_bio101/instructor/questions?format=csv").text
    assert csv.splitlines()[0] == "public_id,date,topic_id,level,grounded,question" and "u_sara" not in csv


def test_instructor_pin_and_publish(world, monkeypatch):
    client, _ = world
    monkeypatch.setenv("INSTRUCTOR_PIN", "1234")
    assert client.get("/rooms/r/instructor/questions").status_code == 401
    assert client.get("/rooms/r/instructor/questions", headers={"X-Instructor-Pin": "1234"}).status_code == 200
    posted = []
    monkeypatch.setenv("M5_PUBLISH_URL", "http://m5.test/in")

    class Ok:
        def raise_for_status(self): pass

    def post(url, json=None, timeout=None, headers=None, **k):
        if url != "http://m5.test/in":
            raise ConnectionError("down")
        posted.append((url, json))
        return Ok()

    monkeypatch.setattr(anonymous.requests, "post", post)
    client.post("/rooms/r/ask", json={"question": "what is ATP"}, headers=SARA)
    out = client.post("/rooms/r/instructor/publish", headers={"X-Instructor-Pin": "1234"}).json()
    assert out["posted"] is True and out["count"] == 1 and posted[0][0] == "http://m5.test/in" and "user_id" not in str(posted[0][1])


def test_the_log_belongs_to_one_student(world):
    client, _ = world
    client.post("/rooms/r/ask", json={"question": "q one"}, headers=SARA)
    client.post("/rooms/r/ask", json={"question": "q two"}, headers={"X-User-Id": "u_omar"})
    assert [r["question"] for r in client.get("/rooms/r/questions", headers=SARA).json()] == ["q one"]
    assert [r["question"] for r in client.get("/rooms/r/questions", headers={"X-User-Id": "u_omar"}).json()] == ["q two"]


# ----------------------------------------------------------------------------- vision
def _photo(lines=("2 + 3 = 5", "4 x 2 = 9"), blur=False) -> bytes:
    img = np.full((520, 900, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (60, 130 + i * 160), cv2.FONT_HERSHEY_SIMPLEX, 3, (30, 30, 30), 8, cv2.LINE_AA)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), 25)
    return cv2.imencode(".png", img)[1].tobytes()


class FakeReader(snap_readers.BaseReader):
    """Reads the lines the test drew, in order: stands for the vision-language model."""
    name, use_lines = "fake", True

    def __init__(self, texts):
        self.texts = list(texts)

    def read(self, img):
        return self.texts.pop(0) if self.texts else ""


@pytest.fixture
def reader(monkeypatch):
    def install(texts):
        monkeypatch.setattr(snap_service, "get_reader", lambda name: FakeReader(texts))
    return install


def test_handwriting_flow_gives_status_one_hint_per_step_boxes_pages_and_the_m4_event(world, reader):
    client, sent = world
    reader(["2 + 3 = 5", "4 × 2 = 9"])
    r = client.post("/rooms/r_bio101/vision/check-work", files={"photo": ("w.png", _photo(), "image/png")}, data={"language": "en", "annotate": "true"}, headers=SARA)
    assert r.status_code == 200, r.text
    body = r.json()
    assert_contract("VisionCheck", body)
    ok, bad = body["lines"]
    assert (ok["status"], ok["hint"]) == ("ok", "") and bad["status"] == "error" and bad["hint"] and "." not in bad["hint"].rstrip(".")
    assert body["needs_help"] and body["quality_ok"] and body["kind"] == "math"
    assert all(0 <= v <= 1 for ln in body["lines"] for v in ln["box"].values()) and bad["box"]["y"] > ok["box"]["y"]
    assert body["citations"] == [] or all({"file_name", "page"} == set(c) for c in body["citations"])
    png = base64.b64decode(body["annotated_png_base64"])
    assert png[:4] == b"\x89PNG"
    assert [(e["kind"], e["room_id"]) for e in sent] == [("question", "r_bio101")]


def test_all_correct_work_tells_m4_the_student_understood(world, reader):
    client, sent = world
    reader(["2 + 3 = 5", "4 × 2 = 8"])
    body = client.post("/rooms/r/vision/check-work", files={"photo": ("w.png", _photo(("2 + 3 = 5", "4 x 2 = 8")), "image/png")}, headers=SARA).json()
    assert [ln["status"] for ln in body["lines"]] == ["ok", "ok"] and body["needs_help"] is False
    assert [e["kind"] for e in sent] == ["understood"]


def test_a_blurry_photo_gets_a_retake_tip_and_no_check(world, reader):
    client, sent = world
    reader(["should not be read"])
    body = client.post("/rooms/r/vision/check-work", files={"photo": ("b.png", _photo(blur=True), "image/png")}, headers=SARA).json()
    assert_contract("VisionCheck", body)
    assert body["quality_ok"] is False and body["retake_tip"] and body["lines"] == [] and sent == []


def test_vision_rejects_a_file_that_is_not_an_image_and_a_missing_photo(world):
    client, _ = world
    assert client.post("/rooms/r/vision/check-work", files={"photo": ("x.txt", b"not an image", "text/plain")}).status_code == 400
    assert client.post("/rooms/r/vision/check-work").status_code == 422


def test_the_opencv_steps_are_all_there():
    from rafeeq.snap.preprocess import PRESETS
    assert PRESETS["pre"]["perspective"] and PRESETS["pre"]["deskew"] and PRESETS["pre"]["denoise"] and PRESETS["pre"]["threshold"]
    assert not any(PRESETS["raw"].values())


# ----------------------------------------------------------------------------- hints: one per step, never the solution
@pytest.mark.parametrize("hint,status,expect_generic", [
    ("Think about the sign when 6 moves to the other side. Then redo it.", "error", False),
    ("2x = 14 - 6", "error", True), ("So x = 4 is the answer", "error", True), ("", "check", True), ("anything", "ok", False)])
def test_one_hint_never_a_line_of_the_solution(hint, status, expect_generic):
    out = one_hint(hint, status, "en")
    if status == "ok":
        assert out == ""
    else:
        assert out and ". " not in out and "=" not in out
        assert (out != hint) == expect_generic or not expect_generic


def test_hints_are_in_the_site_language():
    assert any("؀" <= c <= "ۿ" for c in one_hint("", "error", "ar")) and not any("؀" <= c <= "ۿ" for c in one_hint("", "error", "en"))


def test_the_agent_is_a_langgraph_graph_with_the_agreed_nodes():
    names = set(agent.build_graph().get_graph().nodes)
    assert {"tag_question", "answer", "quiz", "decide_next", "plan_queries", "web_search", "write_study_file", "log_question", "notify_m4"} <= names

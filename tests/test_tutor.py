"""M3 tutor: /tutor/answer and /tutor/check-work. The LLM and M2 are replaced, so no key and no other member's code is needed."""
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator, FormatChecker

from rafeeq.samples import load_sample
from rafeeq.tutor import answer as answer_mod
from rafeeq.tutor import check_work as check_mod
from rafeeq.tutor import events, grading
from rafeeq.tutor import llm as llm_mod
from rafeeq.tutor import m2 as m2_mod
from rafeeq.tutor import search as search_mod
from rafeeq.tutor.router import router

SPEC = yaml.safe_load((Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))
app = FastAPI()
app.include_router(router)
client = TestClient(app)


def assert_matches(schema_name, data):
    v = Draft202012Validator({"$ref": f"#/components/schemas/{schema_name}", "components": SPEC["components"]}, format_checker=FormatChecker())
    errors = [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(data)]
    assert not errors, errors


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """M2 is down (the sample chunks are used), M4 is never called, and the LLM answers nothing unless a test sets it."""
    def down(*a, **k):
        raise ConnectionError("M2 is down in tests")
    monkeypatch.setattr(m2_mod.requests, "post", down)
    monkeypatch.setattr(events, "send_question_event", lambda *a, **k: None)
    monkeypatch.setattr(llm_mod, "ask_json", lambda prompt: None)


def test_answer_from_the_sample_material_matches_the_contract():
    r = client.post("/tutor/answer", json={"room_id": "r_bio101", "question": "ما وظيفة الميتوكوندريا؟"})
    assert r.status_code == 200
    assert_matches("TutorAnswer", r.json())
    body = r.json()
    assert body["grounded"] and body["citations"] and body["steps"] == []        # no LLM: the best passage itself, with its source


def test_answer_with_the_model_cites_only_the_passages_it_used(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda p: {"grounded": True, "answer": "يحدث التنفس الخلوي داخل الميتوكوندريا.", "steps": ["أ", "ب"], "used": [2]})
    body = client.post("/tutor/answer", json={"room_id": "r", "question": "كيف تنتج الميتوكوندريا الطاقة؟", "language": "ar"}).json()
    assert_matches("TutorAnswer", body)
    assert body["citations"] == [{"file_name": "biology_ch3.pdf", "page": 13}] and body["steps"] == ["أ", "ب"]


def test_not_grounded_when_the_model_says_so(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda p: {"grounded": False, "answer": "", "steps": [], "used": []})
    body = client.post("/tutor/answer", json={"room_id": "r", "question": "what is the capital of France?"}).json()
    assert_matches("TutorAnswer", body)
    assert body["grounded"] is False and body["citations"] == []


def test_not_grounded_when_there_is_no_material(monkeypatch):
    monkeypatch.setenv("TUTOR_SAMPLE_FALLBACK", "0")
    body = client.post("/tutor/answer", json={"room_id": "r", "question": "anything"}).json()
    assert body["grounded"] is False


def test_at_most_three_passages_reach_the_model(monkeypatch):
    seen = {}
    many = [{"file_name": "f.pdf", "page": i, "text": f"passage {i}", "topic_id": "t"} for i in range(1, 8)]
    monkeypatch.setattr(m2_mod.requests, "post", lambda *a, **k: type("R", (), {
        "raise_for_status": lambda s: None, "json": lambda s: {"chunks": many}})())
    monkeypatch.setattr(llm_mod, "ask_json", lambda p: seen.update(prompt=p) or {"grounded": True, "answer": "x", "steps": [], "used": [1]})
    client.post("/tutor/answer", json={"room_id": "r", "question": "q"})
    assert "[3]" in seen["prompt"] and "[4]" not in seen["prompt"]


def test_unknown_field_is_rejected():
    assert client.post("/tutor/answer", json={"room_id": "r", "question": "q", "extra": 1}).status_code == 422


def test_the_question_event_gets_the_topic_and_the_user(monkeypatch):
    sent = []
    monkeypatch.setattr(events, "send_question_event", lambda *a: sent.append(a))
    import rafeeq.tutor.router as router_mod
    monkeypatch.setattr(router_mod, "send_question_event", lambda *a: sent.append(a))
    client.post("/tutor/answer", json={"room_id": "r_bio101", "question": "ما وظيفة الميتوكوندريا؟"},
                headers={"Authorization": "Bearer demo-token-sara-123"})
    assert sent == [("u_sara", "r_bio101", "t_energy", "ما وظيفة الميتوكوندريا؟")]


def test_check_work_finds_the_arithmetic_mistake_without_an_llm():
    r = client.post("/tutor/check-work", json={"room_id": "r", "file_id": "f_1", "text": "2 + 3 = 5\n4 * 2 = 9\n10 - 1 = 9"})
    assert r.status_code == 200
    body = r.json()
    assert_matches("HandwritingCheck", body)
    assert [s["correct"] for s in body["steps"]] == [True, False, True]
    assert len(body["boxes"]) == 1 and body["boxes"][0]["y"] == pytest.approx(1 / 3, abs=0.001)   # the 2nd line of 3


def test_check_work_hint_is_a_nudge_not_the_answer(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda p: {"steps": [{"i": 1, "status": "ok", "hint": ""}, {"i": 2, "status": "error", "hint": "ماذا يحدث لإشارة الرقم عند نقله؟"}]})
    body = client.post("/tutor/check-work", json={"room_id": "r", "file_id": "f", "text": "2x + 6 = 14\n2x = 14 + 6"}).json()
    assert_matches("HandwritingCheck", body)
    assert body["hint"] == "ماذا يحدث لإشارة الرقم عند نقله؟" and len(body["boxes"]) == 1


def test_check_work_all_correct_has_no_box():
    body = client.post("/tutor/check-work", json={"room_id": "r", "file_id": "f", "text": "2 + 2 = 4\n3 * 3 = 9"}).json()
    assert_matches("HandwritingCheck", body)
    assert body["boxes"] == [] and all(s["correct"] for s in body["steps"])


def test_check_work_empty_text():
    body = client.post("/tutor/check-work", json={"room_id": "r", "file_id": "f", "text": "  \n "}).json()
    assert_matches("HandwritingCheck", body)
    assert body["steps"] == [] and body["boxes"] == []


def test_the_sample_material_matches_what_the_tutor_expects():
    assert_matches("SearchMaterial", load_sample("search_material"))

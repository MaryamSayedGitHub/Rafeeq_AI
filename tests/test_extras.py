"""M4's quiz client, the spaced review of mistakes and the topic filter (the data behind the tutor agent). M2, M4 and the LLM are replaced."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from rafeeq.samples import load_sample
from rafeeq.tutor import m2, m4, study_store
from rafeeq.tutor import llm as llm_mod


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Everything local: data in a temp folder, M2 / M4 / the LLM down (so the samples and the local scoring are used)."""
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path))
    down = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))
    for mod in (m2, m4):
        monkeypatch.setattr(mod.requests, "post", down)
        monkeypatch.setattr(mod.requests, "get", down)
    monkeypatch.setattr(llm_mod, "ask_json", lambda prompt, max_tokens=None: None)
    return None


# ----------------------------------------------------------------------------- quiz (M4)
def test_m4_is_called_exactly_as_the_contract_says(monkeypatch):
    seen = []

    class QuizRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        room_id: str
        topic_id: str
        count: int

    class Submit(BaseModel):
        model_config = ConfigDict(extra="forbid")
        answers: list[str]

    fake = FastAPI()

    @fake.post("/quiz")
    def _q(body: QuizRequest):
        seen.append(("quiz", body.model_dump()))
        return load_sample("make_quiz")

    @fake.post("/quiz/{quiz_id}/submit")
    def _s(quiz_id: str, body: Submit):
        seen.append(("submit", quiz_id, body.answers))
        return load_sample("submit_quiz")

    c = TestClient(fake)
    path = lambda u: u.replace("http://127.0.0.1:8000", "")
    monkeypatch.setattr(m4.requests, "post", lambda url, json=None, headers=None, timeout=None: c.post(path(url), json=json))
    quiz, src = m4.make_quiz("r1", "t_energy", 2)
    result, src2 = m4.submit_quiz(quiz, ["a", "b"])
    assert (src, src2) == ("m4", "m4") and result == load_sample("submit_quiz")
    assert seen == [("quiz", {"room_id": "r1", "topic_id": "t_energy", "count": 2}), ("submit", "q_001", ["a", "b"])]


def test_without_m4_the_quiz_is_scored_here(monkeypatch):
    monkeypatch.setattr(m4.requests, "post", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    quiz = load_sample("make_quiz")
    result, src = m4.submit_quiz(quiz, ["الميتوكوندريا", "DNA"])
    assert src == "local" and (result["score"], result["total"]) == (1, 2) and len(result["explanations"]) == 2


# ----------------------------------------------------------------------------- review (1 day, 7 days, 30 days)
def test_a_mistake_is_due_after_a_day_and_climbs_the_ladder(client, monkeypatch):
    study_store.add_review("r", "mcq", "2 + 2?", "4", "basic", ["3", "4"])
    assert study_store.due_reviews() == []
    later = lambda days: (datetime.now(timezone.utc) + timedelta(days=days, hours=1)).isoformat(timespec="seconds")
    monkeypatch.setattr(study_store, "now_iso", lambda: later(1))
    [item] = study_store.due_reviews()
    assert item["prompt"] == "2 + 2?" and item["options"] == ["3", "4"]
    assert study_store.grade_review(item["id"], True)                          # knew it: back in 7 days
    assert study_store.due_reviews() == []
    monkeypatch.setattr(study_store, "now_iso", lambda: later(8))
    assert len(study_store.due_reviews()) == 1
    assert study_store.grade_review(item["id"], False)                         # not yet: tomorrow again
    monkeypatch.setattr(study_store, "now_iso", lambda: later(10))
    assert len(study_store.due_reviews()) == 1
    for _ in range(3):                                                         # 3 successes: done for good
        study_store.grade_review(item["id"], True)
    monkeypatch.setattr(study_store, "now_iso", lambda: later(100))
    assert study_store.due_reviews() == [] and study_store.grade_review(item["id"], True) is False


def test_the_same_mistake_is_kept_once(client):
    for _ in range(3):
        study_store.add_review("r", "mcq", "same question", "x")
    assert study_store.count_reviews()["waiting"] == 1



# ----------------------------------------------------------------------------- topic filter, read aloud, camera
def test_topic_filter_keeps_only_that_topics_passages(monkeypatch):
    sent = []
    many = [{"file_name": "f.pdf", "page": i, "text": f"t{i}", "topic_id": "t_a" if i % 2 else "t_b"} for i in range(1, 11)]

    class R:
        def raise_for_status(self): pass
        def json(self): return {"chunks": many}

    monkeypatch.setattr(m2.requests, "post", lambda url, json=None, timeout=None: sent.append(json) or R())
    chunks, src = m2.search("r", "q", topic_id="t_b")
    assert src == "m2" and [c["page"] for c in chunks] == [2, 4, 6] and sent[0]["top_k"] == 12 and set(sent[0]) == {"room_id", "query", "top_k"}

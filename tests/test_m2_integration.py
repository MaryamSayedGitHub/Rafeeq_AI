"""The tutor talks to M2 exactly as the contract says. A fake M2 (strict models, like the contract's additionalProperties: false)
answers instead of the real one, so this passes today and keeps passing when M2's real routes arrive."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field

from rafeeq.tutor import llm as llm_mod
from rafeeq.tutor import m2
from rafeeq.tutor.answer import tutor_answer

fake = FastAPI()
PASSAGES = [{"file_name": "chem.pdf", "page": i, "text": f"Atoms bond by sharing electrons, passage {i}.", "topic_id": "t_bond"} for i in range(1, 8)]
calls = []


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room_id: str
    query: str
    top_k: int = Field(5, ge=1, le=20)


@fake.post("/search")
def _search(body: SearchRequest):
    calls.append(("search", body.model_dump()))
    return {"chunks": PASSAGES[: body.top_k] if "atoms" in body.query.lower() else []}


@fake.get("/rooms/{room_id}/topics")
def _topics(room_id: str):
    return {"topics": [{"topic_id": "t_bond", "name": "الروابط الكيميائية"}]}


@fake.get("/sources/page")
def _page(room_id: str, file_name: str, page: int):
    if file_name == "chem.pdf" and page == 2:
        return {"file_name": file_name, "page": page, "text": "full page two", "image_url": "/files/f_9/pages/2.png"}
    from fastapi import HTTPException
    raise HTTPException(404)


@pytest.fixture
def m2_up(monkeypatch):
    client = TestClient(fake)
    calls.clear()
    path = lambda url: url.replace("http://127.0.0.1:8000", "")
    monkeypatch.setattr(m2.requests, "post", lambda url, json=None, timeout=None: client.post(path(url), json=json))
    monkeypatch.setattr(m2.requests, "get", lambda url, params=None, timeout=None: client.get(path(url), params=params))
    monkeypatch.setattr(llm_mod, "ask_json", lambda prompt, max_tokens=None: None)
    return client


@pytest.fixture
def m2_down(monkeypatch):
    def down(*a, **k):
        raise ConnectionError("M2 is down")
    monkeypatch.setattr(m2.requests, "post", down)
    monkeypatch.setattr(m2.requests, "get", down)
    monkeypatch.setattr(llm_mod, "ask_json", lambda prompt, max_tokens=None: None)


def test_search_sends_exactly_the_agreed_fields_and_never_more_than_three(m2_up):
    chunks, source = m2.search("r_chem", "how do atoms bond", top_k=10)
    assert source == "m2" and len(chunks) == 3                         # the fake rejects extra fields with 422: "m2" proves the shape is right
    assert calls == [("search", {"room_id": "r_chem", "query": "how do atoms bond", "top_k": 3})]


def test_the_answer_cites_m2s_own_file_and_page(m2_up):
    res = tutor_answer("r_chem", "how do atoms bond")
    assert res["source"] == "m2" and res["grounded"] and res["citations"] == [{"file_name": "chem.pdf", "page": 1}]


def test_an_empty_answer_from_m2_means_not_found_not_an_outage(m2_up):
    res = tutor_answer("r_chem", "unrelated question")
    assert res["source"] == "m2" and res["grounded"] is False and res["citations"] == []


def test_topics_and_pages_come_from_m2(m2_up):
    assert m2.topics("r_chem") == ([{"topic_id": "t_bond", "name": "الروابط الكيميائية"}], "m2")
    data, src = m2.source_page("r_chem", "chem.pdf", 2)
    assert src == "m2" and data["image_url"] == "/files/f_9/pages/2.png"


def test_a_page_m2_does_not_have_falls_back_to_the_passage_the_tutor_received(m2_up):
    m2.search("r_chem", "how do atoms bond")                           # remembers passages 1..3
    data, src = m2.source_page("r_chem", "chem.pdf", 3)
    assert src == "passage" and "passage 3" in data["text"]
    assert m2.source_page("r_chem", "nope.pdf", 99) == (None, "none")


def test_while_m2_is_missing_the_samples_are_used_and_labelled(m2_down):
    chunks, source = m2.search("r", "anything")
    assert source == "sample" and chunks and m2.topics("r")[1] == "sample"
    assert tutor_answer("r", "ما وظيفة الميتوكوندريا؟")["source"] == "sample"


def test_without_the_fallback_a_missing_m2_means_no_material(m2_down, monkeypatch):
    monkeypatch.setenv("TUTOR_SAMPLE_FALLBACK", "0")
    assert m2.search("r", "anything") == ([], "none")
    assert tutor_answer("r", "anything")["grounded"] is False

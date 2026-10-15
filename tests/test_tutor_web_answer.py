"""A question the course material cannot answer is never answered from sample passages: the tutor says so, searches the web and
gives about three lines with the sources. The study file stays one button away (Level 2)."""
import pytest

from rafeeq.tutor import llm, m2, safety, study_store, tutor_agent, web_search

PAGES = [{"title": "WWI", "url": "https://example.org/ww1", "content": "The First World War began in 1914 after the assassination of Archduke Franz Ferdinand. " * 3, "score": 1}]


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path / "study"))
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.delenv("TUTOR_SAMPLE_FALLBACK", raising=False)
    monkeypatch.setattr(tutor_agent, "notify_m4", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(llm, "ask_json", lambda prompt, max_tokens=None: {"answer": "It began in 1914. [1]"} if "web sources" in prompt else None)
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: PAGES)
    monkeypatch.setattr(safety, "check", lambda q: None, raising=False)


def test_m2_being_down_does_not_fall_back_to_sample_passages(world, monkeypatch):
    def down(*a, **k):
        raise ConnectionError("no M2")
    monkeypatch.setattr(m2.requests, "post", down)
    monkeypatch.delenv("RAFEEQ_KNOWLEDGE", raising=False)
    chunks, source = m2.search("r1", "what caused the first world war?", 3)
    assert chunks == [] and source == "none"


def test_a_question_outside_the_course_is_answered_from_the_web_with_a_lead_line_and_sources(world, monkeypatch):
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    r = tutor_agent.act("r1", "ask", "what caused the first world war?", None, "en")["result"]
    assert r["material_source"] == "web" and r["web_sources"] and r["study_file"] is None and r["offer_level2"] is True
    assert r["answer"].startswith("I did not find this in the course material, so I searched the web:") and "1914" in r["answer"]
    assert study_store.list_study_files("r1") == []


def test_the_lead_line_is_in_the_language_of_the_answer(world, monkeypatch):
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    r = tutor_agent.act("r1", "ask", "ما أسباب الحرب العالمية الأولى؟", None, "ar")["result"]
    assert r["answer"].startswith("لم أجد هذا في المنهج، فبحثتُ في النت:")


# ----------------------------------------------------------------------------- "in detail"
from rafeeq.tutor import answer as answer_mod


@pytest.mark.parametrize("text", ["اشرح الميتوكوندريا بالتفصيل", "اشرح الموضوع بالتفصيل الممل", "explain osmosis in detail", "give me a detailed explanation of ATP",
                                  "explain it in depth", "شرح مفصل للخلية"])
def test_the_student_asking_for_detail_is_noticed(text):
    assert answer_mod.wants_detail(text)


@pytest.mark.parametrize("text", ["what does the mitochondria do?", "ما وظيفة الميتوكوندريا؟", "what is a cell?"])
def test_a_plain_question_is_not_a_request_for_detail(text):
    assert not answer_mod.wants_detail(text)


def test_in_detail_gives_more_passages_a_longer_prompt_and_room_to_write(world, monkeypatch):
    seen = {}
    chunks = [{"file_name": "bio.pdf", "page": i, "text": f"Mitochondria fact number {i}. " * 20, "topic_id": "t"} for i in range(1, 9)]

    def search(room, q, top_k=3, topic_id=None, detail=False):
        seen["top_k"] = top_k
        return chunks[:top_k], "m2"
    monkeypatch.setattr(m2, "search", search)

    def ask(prompt, max_tokens=None):
        seen["prompt"], seen["max"] = prompt, max_tokens
        return {"grounded": True, "answer": "long answer", "steps": [], "used": [1]}
    monkeypatch.setattr(llm, "ask_json", ask)
    answer_mod.tutor_answer("r1", "explain the mitochondria in detail", "en")
    assert seen["top_k"] == 8 and "IN DETAIL" in seen["prompt"] and seen["max"] == 2000 and "[8]" in seen["prompt"]
    answer_mod.tutor_answer("r1", "explain the mitochondria", "en")
    assert seen["top_k"] == 3 and "IN DETAIL" not in seen["prompt"] and seen["max"] is None


def test_in_detail_from_the_web_asks_for_a_full_explanation_and_more_sources(world, monkeypatch):
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    asked, prompts = [], []
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: asked.append((q, n)) or PAGES)
    monkeypatch.setattr(llm, "ask_json", lambda prompt, max_tokens=None: prompts.append((prompt, max_tokens)) or {"answer": "A long explanation. [1]"})
    tutor_agent.act("r1", "ask", "explain the first world war in detail", None, "en")
    assert "IN DETAIL" in prompts[-1][0] and prompts[-1][1] == 2000
    assert "in detail" not in asked[0][0].lower()                                                  # the wish for length is not searched

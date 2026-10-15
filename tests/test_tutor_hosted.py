"""M3 runs the way the rest of the team runs: the hosted model of M1 (LLM_API_KEY) and the in-process M2. Nothing local is needed."""
import pytest

from rafeeq.snap import readers
from rafeeq.tutor import llm, m2


class Reply:
    def __init__(self, status=200, content='{"ok": true}'):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    sent = []
    replies = []

    def post(url, headers=None, json=None, timeout=None):
        sent.append((url, headers, json))
        return replies.pop(0) if replies else Reply()
    monkeypatch.setattr(llm.requests, "post", post)
    return sent, replies


def test_the_tutor_uses_the_hosted_model_of_m1_when_a_key_is_set(hosted):
    sent, _ = hosted
    assert llm.ask_json("hi", max_tokens=50) == {"ok": True}
    url, headers, body = sent[0]
    assert url == "https://llm.example/v1/chat/completions" and headers["Authorization"] == "Bearer k"
    assert body["model"] == "some-model" and body["max_tokens"] == 50 and body["messages"][0]["content"] == "hi"


def test_a_busy_model_is_asked_again_and_thinking_text_is_dropped(hosted):
    sent, replies = hosted
    replies += [Reply(429), Reply(200, '<think>hmm</think>{"answer": "yes"}')]
    assert llm.ask_json("hi") == {"answer": "yes"} and len(sent) == 2


def test_a_dead_model_gives_none_instead_of_breaking_the_page(hosted):
    _, replies = hosted
    replies += [Reply(401)]
    assert llm.ask_json("hi") is None


def test_ollama_is_only_used_when_asked_for(hosted, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    sent, _ = hosted
    monkeypatch.setattr(llm.requests, "post", lambda url, timeout=None, json=None: sent.append(url) or type("R", (), {"status_code": 200, "raise_for_status": lambda s: None, "json": lambda s: {"message": {"content": '{"a": 1}'}}})())
    assert llm.ask_json("hi") == {"a": 1} and sent[0].endswith("/api/chat")


def test_the_photo_reader_is_the_hosted_one_when_a_key_is_set(monkeypatch):
    monkeypatch.delenv("SNAP_READER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert readers.default_reader() == "hosted"
    monkeypatch.setenv("LLM_API_KEY", "put-your-key-here")                                       # the .env.example placeholder is not a key
    assert readers.default_reader() == "ollama" and not llm.hosted_ready()
    monkeypatch.setenv("GROQ_API_KEY", "g")                                                       # the team's Groq key reads photos too
    assert readers.default_reader() == "hosted" and readers._endpoint()[0].startswith("https://api.groq.com")
    monkeypatch.delenv("LLM_API_KEY")
    monkeypatch.delenv("GROQ_API_KEY")
    assert readers.default_reader() == "ollama"
    monkeypatch.setenv("SNAP_READER", "gemini")
    assert readers.default_reader() == "gemini"


def test_level_1_reads_m2_in_process_when_the_knowledge_base_is_on(monkeypatch):
    calls = []

    class Service:
        def search(self, room, query, top_k):
            calls.append(("search", room, query, top_k))
            return {"chunks": [{"file_name": "bio.pdf", "page": 3, "text": "Mitochondria make ATP.", "topic_id": "t_cell"}]}

        def list_topics(self, room):
            return {"topics": [{"topic_id": "t_cell", "name": "Cell"}]}

        def source_page(self, room, file_name, page):
            return {"file_name": file_name, "page": page, "text": "Mitochondria make ATP."}

    monkeypatch.delenv("M2_BASE_URL", raising=False)
    monkeypatch.delenv("M2_SEARCH_URL", raising=False)
    monkeypatch.setenv("RAFEEQ_KNOWLEDGE", "1")
    monkeypatch.setattr("rafeeq.knowledge.service.get_service", lambda: Service())
    monkeypatch.setattr(m2.requests, "post", lambda *a, **k: pytest.fail("no HTTP round trip to ourselves"))
    chunks, source = m2.search("r1", "what do mitochondria do?", 3)
    assert source == "m2" and chunks[0]["page"] == 3 and calls == [("search", "r1", "what do mitochondria do?", 3)]
    assert m2.topics("r1") == ([{"topic_id": "t_cell", "name": "Cell"}], "m2")
    assert m2.source_page("r1", "bio.pdf", 3)[1] == "m2"
    assert m2.status() == {"connected": True, "detail": "in-process"}


def test_without_the_knowledge_base_the_tutor_still_talks_http(monkeypatch):
    monkeypatch.setenv("RAFEEQ_KNOWLEDGE", "0")
    assert m2._direct() is None

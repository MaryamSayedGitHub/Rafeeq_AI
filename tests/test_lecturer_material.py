"""The lecturer assistant prepares a room's session FROM the room's uploaded material (not from the topic name alone)."""
from rafeeq.teacher.lecturer import material


class FakeKB:
    def __init__(self, chunks):
        self.chunks, self.queries = chunks, []

    def search(self, room_id, query, top_k):
        self.queries.append((room_id, query, top_k))
        return {"chunks": self.chunks}


TWT = [{"file_name": "TWT_Study_Guide.pdf", "page": 2, "text": "A traveling-wave tube (TWT) is a microwave amplifier.", "topic_id": "t1"},
       {"file_name": "TWT_Study_Guide.pdf", "page": 5, "text": "The helix is a slow-wave structure.", "topic_id": "t2"}]


def test_room_passages_become_the_course_material(monkeypatch):
    kb = FakeKB(TWT)
    monkeypatch.setattr(material, "_service", lambda: kb)
    text = material.room_material({"room_id": "r_1", "topic": "TWT", "learning_objectives": ["helix"]})
    assert "traveling-wave tube" in text and "[TWT_Study_Guide.pdf, p. 2]" in text
    assert text.count("slow-wave") == 1                                # the same passage from two queries is kept once
    assert [q[1] for q in kb.queries] == ["TWT", "helix"] and all(q[0] == "r_1" for q in kb.queries)


def test_no_room_or_no_knowledge_base_means_no_material(monkeypatch):
    monkeypatch.setattr(material, "_service", lambda: FakeKB(TWT))
    assert material.room_material({"room_id": "", "topic": "TWT"}) == ""
    monkeypatch.setattr(material, "_service", lambda: None)
    assert material.room_material({"room_id": "r_1", "topic": "TWT"}) == ""


def test_a_broken_knowledge_base_never_breaks_the_draft(monkeypatch):
    class Broken:
        def search(self, *a):
            raise RuntimeError("index missing")
    monkeypatch.setattr(material, "_service", lambda: Broken())
    assert material.room_material({"room_id": "r_1", "topic": "TWT"}) == ""


def test_agents_reuse_the_research_step_material(monkeypatch):
    monkeypatch.setattr(material, "room_material", lambda brief: (_ for _ in ()).throw(AssertionError("searched again")))
    assert material.material_for({"material": "[f.pdf, p. 1]\ntext", "brief": {}}) == "[f.pdf, p. 1]\ntext"
    assert material.material_for({"material": "", "brief": {}}) == material.NO_MATERIAL     # research found nothing: no second search

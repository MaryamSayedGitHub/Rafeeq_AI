"""Topics learned per room (rafeeq/knowledge/topics/room_topics.py): fake LLM, fake embedder, no network."""
from pathlib import Path

import numpy as np

from kb_fakes import HashingEmbedder
from rafeeq.knowledge import service as knowledge
from rafeeq.knowledge.topics.room_topics import MAX_ROOM_TOPICS, RoomTopics

BIOLOGY = [
    "Mitochondria produce ATP for the cell through cellular respiration and the electron transport chain.",
    "Photosynthesis in chloroplasts turns light energy, water and carbon dioxide into glucose and oxygen.",
    "DNA replication copies the double helix before cell division, using DNA polymerase.",
]


class FakeLLM:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append(user)
        return self.answers.pop(0) if self.answers else {"topics": []}


TOPICS_1 = {"topics": [
    {"name": "Cellular respiration", "description": "mitochondria ATP cellular respiration electron transport chain"},
    {"name": "Photosynthesis", "description": "chloroplasts light energy glucose oxygen carbon dioxide"},
    {"name": "Introduction", "description": "generic, must be dropped"},
]}


def chunks(room="r_bio"):
    return [{"room_id": room, "text": t, "page": i + 1} for i, t in enumerate(BIOLOGY)]


def test_topics_are_learned_from_the_material_and_chunks_go_to_the_closest_one(tmp_path):
    rt, emb = RoomTopics(tmp_path, min_similarity=0.05), HashingEmbedder()
    topics = rt.ensure_topics("r_bio", chunks(), FakeLLM(TOPICS_1), emb)
    assert [t["name"] for t in topics] == ["Cellular respiration", "Photosynthesis"]  # generic one dropped
    ids = rt.assign("r_bio", emb.embed_documents(BIOLOGY))
    assert ids[0] == topics[0]["topic_id"] and ids[1] == topics[1]["topic_id"]
    assert ids[2] in {topics[0]["topic_id"], topics[1]["topic_id"], "topic_general"}


def test_existing_topics_are_kept_stable_and_only_new_ones_are_added(tmp_path):
    rt, emb = RoomTopics(tmp_path), HashingEmbedder()
    first = rt.ensure_topics("r_bio", chunks(), FakeLLM(TOPICS_1), emb)
    llm = FakeLLM({"topics": [{"name": "Photosynthesis", "description": "again"},
                              {"name": "DNA replication", "description": "DNA polymerase double helix"}]})
    second = rt.ensure_topics("r_bio", chunks(), llm, emb)
    assert [t["topic_id"] for t in second[:2]] == [t["topic_id"] for t in first]
    assert [t["name"] for t in second] == ["Cellular respiration", "Photosynthesis", "DNA replication"]
    assert "Existing topics: Cellular respiration, Photosynthesis" in llm.prompts[0]
    again = RoomTopics(tmp_path)  # survives a restart
    assert again.names("r_bio")[second[2]["topic_id"]] == "DNA replication"


def test_rooms_are_separate_and_the_cap_is_respected(tmp_path):
    rt, emb = RoomTopics(tmp_path), HashingEmbedder()
    many = {"topics": [{"name": f"Topic {i}", "description": f"words {i}"} for i in range(30)]}
    for _ in range(4):
        rt.ensure_topics("r_a", chunks("r_a"), FakeLLM(many), emb)
        many = {"topics": [{"name": f"More {i}", "description": f"x {i}"} for i in range(30)]}
    assert len(rt.topics("r_a")) == MAX_ROOM_TOPICS
    assert rt.topics("r_b") == [] and rt.assign("r_b", np.ones((1, 4))) is None


def test_no_llm_or_a_failing_llm_learns_nothing(tmp_path):
    rt, emb = RoomTopics(tmp_path), HashingEmbedder()
    assert rt.ensure_topics("r_bio", chunks(), None, emb) == []

    def broken(system, user):
        raise RuntimeError("503")

    assert rt.ensure_topics("r_bio", chunks(), broken, emb) == []
    assert rt.ensure_topics("r_bio", chunks(), lambda s, u: {"nope": 1}, emb) == []
    assert not (Path(tmp_path) / "room_topics.json").exists()


def _service(tmp_path, propose, on=True):
    return knowledge.KnowledgeService(
        Path(tmp_path) / "kb", embedder=HashingEmbedder(), status_callback=lambda *a: None,
        parse=lambda path: [{"page": i + 1, "text": t} for i, t in enumerate(BIOLOGY)],
        room_topics=on, topic_propose=propose)


def _upload(svc, tmp_path, file_id="f_bio", room="r_bio"):
    path = Path(tmp_path) / f"{file_id}__biology.pdf"
    path.write_bytes(b"%PDF-1.4 fake")
    return svc.index_file(room, file_id, path).result(timeout=30)


def test_the_service_lists_the_rooms_own_topics_after_indexing(tmp_path):
    svc = _service(tmp_path, FakeLLM(TOPICS_1))
    assert _upload(svc, tmp_path) == 3
    names = [t["name"] for t in svc.list_topics("r_bio")["topics"]]
    assert set(names) <= {"Cellular respiration", "Photosynthesis", "General"}
    assert "Cellular respiration" in names or "Photosynthesis" in names
    assert "Language Models & Text Generation" not in names
    chunk_topics = {c["topic_id"] for c in svc.search("r_bio", "mitochondria ATP", 3)["chunks"]}
    assert chunk_topics <= {"topic_01", "topic_02", "topic_general"}


def test_room_topics_off_keeps_the_fixed_taxonomy_and_never_calls_the_llm(tmp_path):
    llm = FakeLLM(TOPICS_1)
    svc = _service(tmp_path, llm, on=False)
    _upload(svc, tmp_path)
    assert llm.prompts == []
    assert not (Path(tmp_path) / "kb" / "room_topics.json").exists()

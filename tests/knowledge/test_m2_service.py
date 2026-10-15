"""rafeeq/knowledge/service.py: the M2 contract operations, with a fake embedder and a fake PDF parser.

No BGE-M3, no PyMuPDF and no network: this checks the plumbing (upload hook -> status -> index -> search /
topics / source page / tutor answer) and that every returned value matches the schemas in contracts/openapi.yaml.
It says nothing about the retrieval quality of the real model (see rafeeq/knowledge/eval/ for that).
"""
import os
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from kb_fakes import HashingEmbedder
from rafeeq.knowledge import service as knowledge
from rafeeq.knowledge.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[2]
SPEC = yaml.safe_load((ROOT / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))

PAGES = [
    "Self-attention allows each token to attend to other tokens in the sequence of the input text.",
    "Convolutional neural networks are useful for image processing and for many vision tasks.",
    "Gradient descent is used to optimize neural networks by following the gradient of the loss.",
]


def check(schema_name, value):
    schema = {"$ref": f"#/components/schemas/{schema_name}", "components": SPEC["components"]}
    errors = [e.message for e in Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value)]
    assert not errors, errors


def parse_pages(path):
    return [{"page": i + 1, "text": t} for i, t in enumerate(PAGES)]


class Env:
    def __init__(self, tmp_path, parse=parse_pages, **kw):
        self.store_dir = Path(tmp_path) / "knowledge"
        self.embedder = HashingEmbedder()
        self.statuses = []
        self.service = knowledge.KnowledgeService(
            self.store_dir, embedder=self.embedder, parse=parse,
            status_callback=lambda room, fid, status: self.statuses.append((room, fid, status)), **kw)

    def upload(self, room="r_1", file_id="f_a1", name="course.pdf"):
        """What M1 does: save `<file_id>__<name>` and call the hook. Waits for the background worker."""
        path = Path(self.store_dir).parent / f"{file_id}__{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4 fake")
        return self.service.index_file(room, file_id, path).result(timeout=30)

    def fresh_service(self, **kw):
        """A new process looking at the same saved store."""
        return knowledge.KnowledgeService(self.store_dir, embedder=HashingEmbedder(), parse=parse_pages,
                                          status_callback=lambda *a: None, **kw)


def test_default_configuration_is_hybrid_with_exclusion_and_no_rerank(tmp_path):
    d = Env(tmp_path).service.describe()
    assert d["retrieval"] == "hybrid" and d["exclude_non_content"] is True and d["rerank"] is False and d["candidate_pool"] == 50


def test_upload_hook_indexes_in_the_background_and_reports_status(tmp_path):
    env = Env(tmp_path)
    chunks = env.upload()
    assert chunks == 3
    assert env.statuses == [("r_1", "f_a1", "indexing"), ("r_1", "f_a1", "ready")]


def test_search_returns_contract_chunks_with_the_uploaded_name_not_the_stored_name(tmp_path):
    env = Env(tmp_path)
    env.upload()
    out = env.service.search("r_1", "What does self-attention do?", top_k=3)
    check("SearchMaterial", out)
    first = out["chunks"][0]
    assert first["file_name"] == "course.pdf" and first["page"] == 1
    assert set(first) == {"file_name", "page", "text", "topic_id"}
    assert len(out["chunks"]) == 3
    assert isinstance(env.service._retriever(), HybridRetriever)


def test_rooms_are_isolated(tmp_path):
    env = Env(tmp_path)
    env.upload(room="r_1")
    assert env.service.search("r_2", "self-attention", 5) == {"chunks": []}
    assert env.service.list_topics("r_2") == {"topics": []}


def test_nothing_indexed_yet_means_empty_results_and_no_model_is_touched(tmp_path):
    env = Env(tmp_path)
    assert env.service.search("r_1", "anything", 5) == {"chunks": []}
    assert env.service.list_topics("r_1") == {"topics": []}
    assert env.embedder.document_calls == [] and env.embedder.query_calls == []
    assert not env.store_dir.exists()


def test_topics_are_only_those_present_in_the_room_and_match_the_contract(tmp_path):
    env = Env(tmp_path)
    env.upload()
    topics = env.service.list_topics("r_1")
    check("RoomTopics", topics)
    ids = [t["topic_id"] for t in topics["topics"]]
    assert ids and len(ids) == len(set(ids))
    stored = {c["topic_id"] for c in env.service.search("r_1", "neural networks", 5)["chunks"]}
    assert stored <= set(ids)


def test_source_page_returns_the_page_text_or_404(tmp_path):
    env = Env(tmp_path)
    env.upload()
    page = env.service.source_page("r_1", "course.pdf", 2)
    check("SourcePage", page)
    assert page["text"] == PAGES[1] and page["page"] == 2
    for args in (("r_1", "course.pdf", 99), ("r_1", "other.pdf", 1), ("r_2", "course.pdf", 1)):
        try:
            env.service.source_page(*args)
        except knowledge.NotFound as e:
            assert e.status == 404
        else:
            raise AssertionError(f"expected NotFound for {args}")


def test_the_index_survives_a_restart(tmp_path):
    env = Env(tmp_path)
    env.upload()
    again = env.fresh_service()
    assert again.search("r_1", "self-attention", 1)["chunks"][0]["page"] == 1
    assert again.list_topics("r_1")["topics"]


def test_a_second_file_is_added_to_the_same_store(tmp_path):
    env = Env(tmp_path)
    env.upload(file_id="f_a1", name="one.pdf")
    env.upload(file_id="f_b2", name="two.pdf")
    names = {c["file_name"] for c in env.service.search("r_1", "neural networks gradient", 10)["chunks"]}
    assert names == {"one.pdf", "two.pdf"}


def test_indexing_the_same_upload_twice_changes_nothing_and_stays_ready(tmp_path):
    env = Env(tmp_path)
    env.upload()
    assert env.upload() == 0                                  # nothing was added the second time
    assert env.statuses[-1] == ("r_1", "f_a1", "ready")        # a working file is never flipped to failed
    assert len(env.service.search("r_1", "self-attention", 10)["chunks"]) == 3


def test_two_uploads_with_the_same_name_in_one_room_get_distinct_citation_names(tmp_path):
    env = Env(tmp_path)
    env.upload(file_id="f_a1", name="course.pdf")
    assert env.upload(file_id="f_b2", name="course.pdf") == 3
    assert env.upload(file_id="f_c3", name="course.pdf") == 3
    names = {c["file_name"] for c in env.service.search("r_1", "self-attention", 20)["chunks"]}
    assert names == {"course.pdf", "course (2).pdf", "course (3).pdf"}
    assert env.service.source_page("r_1", "course (2).pdf", 1)["text"] == PAGES[0]      # not doubled
    assert [s for _, _, s in env.statuses].count("ready") == 3


def test_a_non_pdf_is_marked_failed_without_loading_anything(tmp_path):
    env = Env(tmp_path)
    assert env.upload(name="slides.pptx") == 0
    assert env.statuses == [("r_1", "f_a1", "failed")]
    assert env.embedder.document_calls == []


def test_a_parser_error_marks_the_file_failed_and_leaves_no_store(tmp_path):
    def broken(path):
        raise RuntimeError("corrupt pdf")

    env = Env(tmp_path, parse=broken)
    assert env.upload() == 0
    assert [s for _, _, s in env.statuses] == ["indexing", "failed"]
    assert env.service.search("r_1", "anything", 3) == {"chunks": []}


def test_input_validation(tmp_path):
    env = Env(tmp_path)
    for call in (lambda: env.service.search("", "q"), lambda: env.service.search("r", " "),
                 lambda: env.service.search("r", "q", 0), lambda: env.service.search("r", "q", 21),
                 lambda: env.service.search("r", "q", True), lambda: env.service.source_page("r", "f.pdf", 0)):
        try:
            call()
        except knowledge.Invalid as e:
            assert e.status == 422
        else:
            raise AssertionError("expected Invalid")


# ------------------------------------------------------------------ tutor_answer

class CitingLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, system, prompt):
        self.prompts.append(prompt)
        return self.reply


def test_tutor_answer_is_grounded_with_citations_and_matches_the_contract(tmp_path):
    env = Env(tmp_path, llm=CitingLLM("Self-attention lets each token attend to the other tokens [1]."))
    env.upload()
    out = env.service.tutor_answer("r_1", PAGES[0])
    check("TutorAnswer", out)
    assert out["grounded"] is True and out["steps"] == []
    assert out["citations"] and out["citations"][0] == {"file_name": "course.pdf", "page": 1}
    assert "[1]" not in out["answer"] and "Sources" not in out["answer"]


def test_tutor_answer_says_not_covered_instead_of_guessing(tmp_path):
    env = Env(tmp_path, llm=CitingLLM("NOT_COVERED"))
    env.upload()
    out = env.service.tutor_answer("r_1", PAGES[1])
    check("TutorAnswer", out)
    assert out["grounded"] is False and out["citations"] == []
    unrelated = env.service.tutor_answer("r_1", "zebra giraffe savanna migration")
    assert unrelated["grounded"] is False and unrelated["citations"] == []


def test_tutor_answer_on_an_empty_knowledge_base_is_not_grounded_and_never_calls_the_llm(tmp_path):
    llm = CitingLLM("should not be used [1]")
    out = Env(tmp_path, llm=llm).service.tutor_answer("r_1", "anything at all")
    assert out["grounded"] is False and llm.prompts == []


def test_tutor_answer_without_an_llm_is_unavailable_not_invented(tmp_path):
    saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL")}
    try:
        try:
            Env(tmp_path).service.tutor_answer("r_1", "anything")
        except knowledge.Unavailable as e:
            assert e.status == 503
        else:
            raise AssertionError("expected Unavailable")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_llm_from_env_uses_the_gemini_openai_endpoint():
    keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        assert knowledge.llm_from_env() is None
        os.environ["GOOGLE_API_KEY"], os.environ["GEMINI_MODEL"] = "k", "some-model"
        llm = knowledge.llm_from_env()
        assert llm.describe()["base_url"] == knowledge.GEMINI_OPENAI_BASE_URL and llm.api_key_env == "GOOGLE_API_KEY"
        os.environ["GEMINI_API_KEY"] = "k2"
        assert knowledge.llm_from_env().api_key_env == "GEMINI_API_KEY"
        assert "k2" not in str(knowledge.llm_from_env().describe())
    finally:
        for k in keys:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_a_deleted_file_disappears_from_search_and_stays_gone_after_a_restart(tmp_path):
    env = Env(tmp_path)
    env.upload(file_id="f_a1", name="one.pdf")
    env.upload(file_id="f_b2", name="two.pdf")
    assert env.service.remove_file("r_1", "f_a1").result(timeout=30) == 3
    names = {c["file_name"] for c in env.service.search("r_1", "neural networks gradient", 10)["chunks"]}
    assert names == {"two.pdf"}
    again = env.fresh_service()
    assert {c["file_name"] for c in again.search("r_1", "self-attention", 10)["chunks"]} == {"two.pdf"}


def test_removing_a_file_that_was_never_indexed_is_harmless(tmp_path):
    env = Env(tmp_path)
    assert env.service.remove_file("r_1", "f_none").result(timeout=30) == 0  # no store yet
    env.upload()
    assert env.service.remove_file("r_1", "f_none").result(timeout=30) == 0
    assert len(env.service.search("r_1", "self-attention", 5)["chunks"]) > 0


def test_room_chunks_give_m4_the_rooms_material_with_topic_names(tmp_path):
    env = Env(tmp_path)
    assert env.service.room_chunks("r_1") == []
    env.upload()
    chunks = env.service.room_chunks("r_1")
    assert len(chunks) == 3 and {c["file_name"] for c in chunks} == {"course.pdf"}
    assert all(c["topic_name"] and c["topic_id"] for c in chunks)
    assert env.service.room_chunks("r_2") == []


def test_overview_explains_the_newest_file_and_cites_its_pages(tmp_path):
    env = Env(tmp_path)
    assert env.service.overview("r_1", "explain the lesson", propose=lambda s, u: {"answer": "x"}) is None  # nothing indexed
    env.upload(file_id="f_a1", name="old.pdf")
    env.upload(file_id="f_b2", name="new.pdf")
    seen = []

    def propose(system, user):
        seen.append(user)
        return {"answer": "The lesson is about attention, CNNs and gradient descent (p. 1).", "pages": [1, 99]}

    out = env.service.overview("r_1", "اشرح الدرس الأخير", propose=propose)
    check("TutorAnswer", out)
    assert out["grounded"] is True and out["citations"] == [{"file_name": "new.pdf", "page": 1}]
    assert "File: new.pdf" in seen[0] and "old.pdf" not in seen[0]
    assert env.service.overview("r_1", "q", propose=lambda s, u: {"nope": 1}) is None


def test_topic_for_citations_finds_the_topic_of_the_cited_pages(tmp_path):
    env = Env(tmp_path)
    env.upload()
    topic = env.service.topic_for_citations("r_1", [{"file_name": "course.pdf", "page": 1}])
    assert topic == env.service.room_chunks("r_1")[0]["topic_id"]
    assert env.service.topic_for_citations("r_1", []) is None
    assert env.service.topic_for_citations("r_1", [{"file_name": "other.pdf", "page": 1}]) is None

"""Answerer tests: fake retriever + fake LLM. No model, no FAISS, no network."""

import pytest

from rafeeq.knowledge.answering import NOT_COVERED_MESSAGE, Answerer, answer, set_default_answerer
from rafeeq.knowledge.answering.answerer import SYSTEM_PROMPT

ROOM = "llm_101"
BOOK = "hands_on_llm.pdf"


def make_chunk(n, page, score, text=None, file_name=BOOK):
    return {"chunk_id": f"c{n}", "text": text or f"excerpt text number {n}", "file_name": file_name,
            "page": page, "topic_id": "topic_attention", "score": score}


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def retrieve(self, room_id, query, k=6):
        self.calls.append((room_id, query, k))
        return list(self.chunks)


class FakeLLM:
    def __init__(self, reply="Self-attention relates tokens to each other [1]."):
        self.reply, self.calls = reply, []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return self.reply


def build(chunks, reply="Self-attention relates tokens to each other [1].", **kwargs):
    retriever, llm = FakeRetriever(chunks), FakeLLM(reply)
    return Answerer(retriever, llm, **kwargs), retriever, llm


GOOD = [make_chunk(1, 21, 0.82), make_chunk(2, 22, 0.61), make_chunk(3, 90, 0.20)]


# ---------- grounded answers ----------

def test_grounded_answer_has_the_expected_shape():
    answerer, _, _ = build(GOOD)
    result = answerer.answer(ROOM, "What is self-attention?")
    assert set(result) == {"answer_md", "grounded", "citations"}
    assert result["grounded"] is True
    assert "Self-attention relates tokens" in result["answer_md"]
    assert result["citations"] == [{"file_name": BOOK, "page": 21}]


def test_citation_markers_map_to_the_right_chunks_in_order_of_first_use():
    answerer, _, _ = build(GOOD, reply="First point [2]. Second point [1].")
    result = answerer.answer(ROOM, "q")
    assert result["citations"] == [{"file_name": BOOK, "page": 22}, {"file_name": BOOK, "page": 21}]


def test_citations_are_deduplicated_by_file_and_page():
    chunks = [make_chunk(1, 21, 0.9), make_chunk(2, 21, 0.8), make_chunk(3, 30, 0.7)]
    answerer, _, _ = build(chunks, reply="A [1]. B [2]. C [3].")
    assert answerer.answer(ROOM, "q")["citations"] == [
        {"file_name": BOOK, "page": 21}, {"file_name": BOOK, "page": 30}]


def test_only_cited_chunks_become_citations():
    answerer, _, _ = build(GOOD, reply="Only the first excerpt matters [1].")
    assert answerer.answer(ROOM, "q")["citations"] == [{"file_name": BOOK, "page": 21}]


def test_combined_markers_are_understood():
    answerer, _, _ = build(GOOD, reply="Both agree [1, 2].")
    result = answerer.answer(ROOM, "q")
    assert [c["page"] for c in result["citations"]] == [21, 22]
    assert "[1][2]" in result["answer_md"]


def test_sources_footer_is_built_from_retrieved_metadata():
    answerer, _, _ = build(GOOD, reply="Yes [1].")
    assert f"- [1] {BOOK}, page 21" in answerer.answer(ROOM, "q")["answer_md"]


def test_citations_from_multiple_files_keep_their_file_names():
    chunks = [make_chunk(1, 3, 0.9, file_name="lecture1.pdf"), make_chunk(2, 7, 0.8, file_name=BOOK)]
    answerer, _, _ = build(chunks, reply="A [1]. B [2].")
    assert answerer.answer(ROOM, "q")["citations"] == [
        {"file_name": "lecture1.pdf", "page": 3}, {"file_name": BOOK, "page": 7}]


# ---------- citation correctness: only from retrieved chunks ----------

def test_markers_to_nonexistent_excerpts_are_removed_and_never_cited():
    answerer, _, _ = build(GOOD, reply="Real [1] and invented [9] and [0].")
    result = answerer.answer(ROOM, "q")
    assert result["citations"] == [{"file_name": BOOK, "page": 21}]
    assert "[9]" not in result["answer_md"] and "[0]" not in result["answer_md"]


def test_below_threshold_chunks_cannot_be_cited_even_if_the_llm_tries():
    # GOOD has 2 relevant chunks (0.82, 0.61); the 3rd (0.20) was never shown to the LLM.
    answerer, _, _ = build(GOOD, reply="Claim [1]. Another claim [3].")
    result = answerer.answer(ROOM, "q")
    assert result["citations"] == [{"file_name": BOOK, "page": 21}]


def test_file_names_or_pages_typed_by_the_llm_never_become_citations():
    answerer, _, _ = build(GOOD, reply="See fake_book.pdf, page 999 [1].")
    result = answerer.answer(ROOM, "q")
    assert result["citations"] == [{"file_name": BOOK, "page": 21}]


def test_every_citation_exists_in_the_retrieved_set():
    answerer, retriever, _ = build(GOOD, reply="A [1]. B [2].")
    retrieved = {(c["file_name"], c["page"]) for c in retriever.chunks}
    for citation in answerer.answer(ROOM, "q")["citations"]:
        assert (citation["file_name"], citation["page"]) in retrieved


# ---------- not covered ----------

def test_nothing_retrieved_means_not_covered_and_the_llm_is_not_called():
    answerer, _, llm = build([])
    result = answerer.answer(ROOM, "What is the capital of France?")
    assert result == {"answer_md": "This topic is not covered in the provided course material.",
                      "grounded": False, "citations": []}
    assert llm.calls == []


def test_exact_not_covered_message_constant():
    assert NOT_COVERED_MESSAGE == "This topic is not covered in the provided course material."


def test_all_chunks_below_min_score_means_not_covered_without_calling_the_llm():
    answerer, _, llm = build([make_chunk(1, 5, 0.10), make_chunk(2, 6, 0.20)])
    result = answerer.answer(ROOM, "q")
    assert result["grounded"] is False and result["citations"] == []
    assert llm.calls == []


def test_min_score_is_configurable():
    chunks = [make_chunk(1, 5, 0.50)]
    strict, _, strict_llm = build(chunks, min_score=0.6)
    lenient, _, lenient_llm = build(chunks, min_score=0.4)
    assert strict.answer(ROOM, "q")["grounded"] is False and strict_llm.calls == []
    assert lenient.answer(ROOM, "q")["grounded"] is True and lenient_llm.calls


def test_llm_saying_not_covered_gives_not_covered():
    answerer, _, llm = build(GOOD, reply="NOT_COVERED")
    result = answerer.answer(ROOM, "q")
    assert result == {"answer_md": NOT_COVERED_MESSAGE, "grounded": False, "citations": []}
    assert llm.calls


def test_llm_answer_without_any_valid_citation_is_rejected():
    answerer, _, _ = build(GOOD, reply="Self-attention is great, trust me.")
    result = answerer.answer(ROOM, "q")
    assert result == {"answer_md": NOT_COVERED_MESSAGE, "grounded": False, "citations": []}


def test_llm_answer_citing_only_invalid_numbers_is_rejected():
    answerer, _, _ = build(GOOD, reply="Made up claim [7].")
    assert answerer.answer(ROOM, "q")["grounded"] is False


def test_empty_llm_reply_is_not_an_answer():
    answerer, _, _ = build(GOOD, reply="   ")
    assert answerer.answer(ROOM, "q")["grounded"] is False


# ---------- the LLM sees only retrieved context ----------

def test_prompt_contains_only_relevant_retrieved_text_and_the_question():
    chunks = [make_chunk(1, 21, 0.9, text="ALPHA excerpt"), make_chunk(2, 22, 0.8, text="BETA excerpt"),
              make_chunk(3, 23, 0.1, text="GAMMA irrelevant excerpt")]
    answerer, _, llm = build(chunks, reply="Answer [1].")
    answerer.answer(ROOM, "What is attention?")
    system, prompt = llm.calls[0]
    assert "ALPHA excerpt" in prompt and "BETA excerpt" in prompt
    assert "GAMMA" not in prompt
    assert "What is attention?" in prompt
    assert f"({BOOK}, page 21)" in prompt


def test_system_prompt_forbids_general_knowledge_and_defines_the_sentinel():
    answerer, _, llm = build(GOOD)
    answerer.answer(ROOM, "q")
    system = llm.calls[0][0]
    assert system == SYSTEM_PROMPT
    assert "ONLY" in system and "general knowledge" in system and "NOT_COVERED" in system


def test_retrieval_is_asked_for_the_right_room_question_and_k():
    answerer, retriever, _ = build(GOOD, k=4)
    answerer.answer("room_x", "my question")
    assert retriever.calls == [("room_x", "my question", 4)]


def test_arabic_questions_are_passed_through_unchanged():
    question = "يعني إيه Self-Attention؟"
    answerer, retriever, llm = build(GOOD, reply="الانتباه الذاتي يربط الكلمات ببعضها [1].")
    result = answerer.answer(ROOM, question)
    assert retriever.calls[0][1] == question and question in llm.calls[0][1]
    assert result["grounded"] is True and "الانتباه" in result["answer_md"]


# ---------- errors and validation ----------

class ExplodingLLM:
    def generate(self, system, prompt):
        raise ConnectionError("provider down")


def test_llm_errors_propagate_instead_of_producing_an_invented_answer():
    answerer = Answerer(FakeRetriever(GOOD), ExplodingLLM())
    with pytest.raises(ConnectionError):
        answerer.answer(ROOM, "q")


def test_non_string_llm_reply_is_rejected():
    answerer, _, _ = build(GOOD, reply=None)
    with pytest.raises(TypeError, match="string"):
        answerer.answer(ROOM, "q")


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_invalid_question_rejected(bad):
    answerer, _, _ = build(GOOD)
    with pytest.raises(ValueError, match="question"):
        answerer.answer(ROOM, bad)


@pytest.mark.parametrize("bad", ["", None])
def test_invalid_room_rejected(bad):
    answerer, _, _ = build(GOOD)
    with pytest.raises(ValueError, match="room_id"):
        answerer.answer(bad, "q")


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_invalid_k_rejected(bad):
    with pytest.raises(ValueError, match="k must"):
        Answerer(FakeRetriever([]), FakeLLM(), k=bad)


@pytest.mark.parametrize("bad", [1.5, -2, "0.3", None])
def test_invalid_min_score_rejected(bad):
    with pytest.raises(ValueError, match="min_score"):
        Answerer(FakeRetriever([]), FakeLLM(), min_score=bad)


# ---------- module-level contract ----------

def test_module_level_answer_requires_configuration_then_works():
    set_default_answerer(None)
    try:
        with pytest.raises(RuntimeError, match="no answerer configured"):
            answer(ROOM, "q")
        answerer, _, _ = build(GOOD)
        set_default_answerer(answerer)
        assert answer(ROOM, "q")["grounded"] is True
    finally:
        set_default_answerer(None)


# ---------- citation marker escaping regression ----------

@pytest.mark.parametrize("citation_marker", ["[1]", r"\[1\]"])
def test_plain_and_escaped_citation_markers_are_grounded(citation_marker):
    answerer, _, _ = build(GOOD, reply=f"Supported claim {citation_marker}.")
    result = answerer.answer(ROOM, "q")

    assert result["grounded"] is True
    assert result["citations"] == [{"file_name": BOOK, "page": 21}]
    assert "Supported claim [1]." in result["answer_md"]
    assert r"\\[1\\]" not in result["answer_md"]

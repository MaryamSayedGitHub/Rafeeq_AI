"""M6 question bank: generation, safe editing, ownership, search, and the hand-off to M4."""
import json
import random

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import rafeeq.quiz.quiz.models  # noqa: F401  (registers the quiz tables on Base)
from rafeeq.quiz.contracts import QuizItem
from rafeeq.quiz.profile.models import Base as QuizBase
from rafeeq.quiz.quiz import service as quiz_service
from rafeeq.teacher.question_bank import difficulty as qb_difficulty
from rafeeq.teacher.question_bank import nlu
from rafeeq.teacher.question_bank import router as qb_router
from rafeeq.teacher.question_bank import stt
from rafeeq.teacher.question_bank import service as qb_service
from rafeeq.teacher.question_bank.service import QuestionBankService

ROOM = "room-1"

GOOD = {
    "stem": "What does a tokenizer do?",
    "options": ["Splits text into tokens", "Trains the model", "Stores vectors", "Draws images"],
    "right_choice": 0,
    "explanation": "It converts raw text into tokens.",
}


def fake_llm(prompt):
    return json.dumps(GOOD)


def chunks():
    return [
        {"chunk_id": f"b-{t}{p}", "text": "x" * 500, "file_name": "book.pdf",
         "page": p, "topic_id": t, "topic_name": t}
        for t in ("a", "b") for p in range(1, 6)
    ]


@pytest.fixture
def svc(tmp_path, monkeypatch):
    service = QuestionBankService(f"sqlite:///{tmp_path}/qb.db")
    monkeypatch.setattr(qb_service, "_service", service)
    return service


@pytest.fixture
def client(svc, monkeypatch):
    """The real router; the teacher is whoever the Authorization header says (no login needed)."""
    monkeypatch.setattr(qb_router, "_require_teacher", lambda authorization: authorization or "t1")
    monkeypatch.setattr(qb_router, "_chunks", lambda room_id: chunks())
    monkeypatch.setattr(qb_router, "_llm", lambda: fake_llm)
    app = FastAPI()
    app.include_router(qb_router.router)
    return TestClient(app)


def url(item_id="", room=ROOM, tail=""):
    base = f"/teacher/rooms/{room}/question-bank"
    return base + (f"/{item_id}" if item_id else "") + tail


def generate(client, teacher="t1", n=3, kinds=("mcq",)):
    r = client.post(url(tail="/generate"), json={"n": n, "kinds": list(kinds)}, headers={"authorization": teacher})
    assert r.status_code == 201, r.text
    return [b["item"]["item_id"] for b in r.json()["items"]]


def H(teacher="t1"):
    return {"authorization": teacher}


# ---------- generate / list ----------

def test_generate_saves_unapproved_items(client):
    ids = generate(client, n=3)
    assert len(ids) == 3
    listed = client.get(url(), headers=H()).json()
    assert listed["total"] == 3
    assert all(b["approved"] is False for b in listed["items"])


def test_generate_without_material_is_422(client, monkeypatch):
    monkeypatch.setattr(qb_router, "_chunks", lambda room_id: [])
    r = client.post(url(tail="/generate"), json={"n": 2}, headers=H())
    assert r.status_code == 422


def test_generate_asks_for_at_most_20(client):
    r = client.post(url(tail="/generate"), json={"n": 21}, headers=H())
    assert r.status_code == 422


def test_generate_avoids_stems_already_in_the_bank(svc):
    seen = []

    def spy(prompt):
        seen.append(prompt)
        return json.dumps(GOOD)

    svc.generate(room_id=ROOM, teacher_id="t1", chunks=chunks(), llm=spy, n=1)
    seen.clear()
    svc.generate(room_id=ROOM, teacher_id="t1", chunks=chunks(), llm=spy, n=1)
    assert GOOD["stem"] in seen[0]   # the second run is told not to repeat the first stem


def test_generate_skips_a_question_that_repeats_one_in_the_bank(svc):
    def llm(stem):
        return lambda prompt: json.dumps({**GOOD, "stem": stem})

    first = svc.generate(room_id=ROOM, teacher_id="t1", chunks=chunks(), llm=llm("In the document intelligence pipeline, what is the specific function of the Vision-Language Model (VLM) stage?"), n=1)
    assert len(first) == 1
    again = svc.generate(room_id=ROOM, teacher_id="t1", chunks=chunks(), llm=llm("In the document intelligence pipeline, what is the primary function of the Vision Language Model (VLM) step?"), n=1)
    assert again == []                                        # the same idea in other words
    other = svc.generate(room_id=ROOM, teacher_id="t1", chunks=chunks(), llm=llm("Why does attention need positional encoding?"), n=1)
    assert len(other) == 1


# ---------- editing is validated ----------

def test_edit_that_breaks_the_question_is_rejected_and_nothing_is_saved(client):
    item_id = generate(client, n=1)[0]
    before = client.get(url(item_id), headers=H()).json()["item"]["right_choice"]   # shuffled, so not always 0

    r = client.put(url(item_id), json={"right_choice": 7}, headers=H())
    assert r.status_code == 422

    # the list still works and the item is unchanged
    listed = client.get(url(), headers=H())
    assert listed.status_code == 200
    assert listed.json()["items"][0]["item"]["right_choice"] == before


def test_valid_edit_is_saved(client):
    item_id = generate(client, n=1)[0]
    before = client.get(url(item_id), headers=H()).json()["item"]["right_choice"]
    new_choice = (before + 1) % 4
    r = client.put(url(item_id), json={"stem": "Why tokenize?", "right_choice": new_choice}, headers=H())
    assert r.status_code == 200
    assert r.json()["item"]["stem"] == "Why tokenize?"
    assert r.json()["item"]["right_choice"] == new_choice


# ---------- ownership ----------

def test_another_teacher_cannot_approve_edit_or_delete(client):
    item_id = generate(client, teacher="t1", n=1)[0]
    assert client.patch(url(item_id, tail="/approve"), json={"approved": True}, headers=H("t2")).status_code == 403
    assert client.put(url(item_id), json={"stem": "hijacked"}, headers=H("t2")).status_code == 403
    assert client.delete(url(item_id), headers=H("t2")).status_code == 403
    assert client.get(url(item_id), headers=H("t2")).status_code == 403


def test_item_from_another_room_is_not_found(client):
    item_id = generate(client, n=1)[0]
    assert client.get(url(item_id, room="other-room"), headers=H()).status_code == 404
    assert client.delete(url(item_id, room="other-room"), headers=H()).status_code == 404


def test_owner_can_delete(client):
    item_id = generate(client, n=1)[0]
    assert client.delete(url(item_id), headers=H()).status_code == 204
    assert client.get(url(item_id), headers=H()).status_code == 404


# ---------- search ----------

def test_search_falls_back_to_word_overlap(client):
    generate(client, n=2)
    r = client.get(url(tail="/search"), params={"q": "what does a tokenizer do"}, headers=H())
    assert r.status_code == 200
    assert r.json()["total"] == 2
    assert client.get(url(tail="/search"), params={"q": "photosynthesis"}, headers=H()).json()["total"] == 0


def test_search_only_returns_my_questions(client):
    generate(client, teacher="t1", n=1)
    r = client.get(url(tail="/search"), params={"q": "tokenizer"}, headers=H("t2"))
    assert r.json()["total"] == 0


def test_search_via_m2_puts_the_retrieved_page_first(svc, monkeypatch):
    from rafeeq.knowledge import integration as ki
    from rafeeq.knowledge import service as ks_module

    for page, stem in ((1, "Question on page 1"), (4, "Question on page 4")):
        item = QuizItem(item_id=f"i{page}", topic_id="a", stem=stem, options=["w", "x", "y", "z"],
                        right_choice=0, explanation="e", file_name="book.pdf", page=page)
        with svc._session() as db:
            db.add(qb_service._item_to_row(item, ROOM, "t1"))
            db.commit()

    class FakeKnowledge:
        def search(self, room_id, query, top_k=5):
            assert top_k <= 20                      # M2 rejects anything above its MAX_TOP_K
            return {"chunks": [{"file_name": "book.pdf", "page": 4, "text": "t", "topic_id": "a"}]}

    monkeypatch.setattr(ki, "enabled", lambda: True)
    monkeypatch.setattr(ks_module, "get_service", lambda: FakeKnowledge())

    found = svc.search(ROOM, "anything", teacher_id="t1", top_k=10)
    assert [e.item.item_id for e in found] == ["i4", "i1"]   # exact page, then the same topic


# ---------- hand-off to M4 ----------

def test_only_approved_items_reach_m4(client, svc):
    ids = generate(client, n=2)
    assert svc.approved_items(ROOM, []) == []
    client.patch(url(ids[0], tail="/approve"), json={"approved": True}, headers=H())
    approved = svc.approved_items(ROOM, [])
    assert [it.item_id for it in approved] == [ids[0]]
    assert approved[0].source == "lecturer_approved"


def test_deps_returns_the_bank_function_when_the_service_exists(svc):
    from rafeeq.quiz.deps import get_approved_items

    assert get_approved_items() == svc.approved_items


def test_deps_returns_empty_when_the_service_is_missing(monkeypatch):
    from rafeeq.quiz.deps import get_approved_items

    monkeypatch.setattr(qb_service, "_service", None)
    assert get_approved_items()(ROOM, []) == []


@pytest.fixture
def quiz_session():
    engine = create_engine("sqlite://")
    QuizBase.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def make_quiz(session, approved, kinds):
    return quiz_service.create_quiz(
        session, user_id="u1", room_id=ROOM, n=3, mode="screen", llm=fake_llm,
        chunks=chunks(), approved_items=approved, rng=random.Random(0), kinds=kinds)


def test_student_quiz_uses_an_approved_mcq(client, svc, quiz_session):
    ids = generate(client, n=1)
    client.patch(url(ids[0], tail="/approve"), json={"approved": True}, headers=H())
    quiz = make_quiz(quiz_session, svc.approved_items, ("mcq",))
    assert quiz.items[0].source == "lecturer_approved"


def test_student_quiz_ignores_approved_items_of_a_kind_not_asked_for(svc, quiz_session):
    essay = QuizItem(item_id="e1", topic_id="a", stem="Explain attention.", explanation="e", file_name="b.pdf",
                     page=1, kind="essay", answer="Attention weighs tokens.", rubric=["weights", "tokens"])
    with svc._session() as db:
        db.add(qb_service._item_to_row(essay, ROOM, "t1", approved=True))
        db.commit()
    quiz = make_quiz(quiz_session, svc.approved_items, ("mcq",))
    assert "e1" not in [i.item_id for i in quiz.items]          # an essay is not slipped into an mcq quiz


# ---------- grading guide (correct / partial / wrong) ----------

ESSAY = {
    "stem": "Explain how self-attention weighs the tokens of a sentence.",
    "answer": "Each token is compared with every other token; the scores are normalised with softmax and used to mix the values.",
    "rubric": ["compares every token with every other", "scores normalised with softmax", "values mixed by the scores"],
    "explanation": "This is the core of the transformer.",
}
GUIDE = {
    "correct": "Says tokens are compared pairwise, scores go through softmax, and the values are mixed.",
    "partial": "Mentions the comparison of tokens but misses softmax or the mixing of values.",
    "wrong": "Describes recurrence or convolution instead of attention.",
}


def essay_llm(prompt):
    if prompt.startswith("You write a grading guide"):
        return json.dumps(GUIDE)
    return json.dumps(ESSAY)


def test_mcq_guide_is_built_by_code_from_the_right_option(client):
    generate(client, n=1)
    entry = client.get(url(), headers=H()).json()["items"][0]
    right = entry["item"]["options"][entry["item"]["right_choice"]]
    assert right in entry["grading"]["correct"]
    assert "No partial credit" in entry["grading"]["partial"]


def test_essay_guide_comes_from_the_llm(client, monkeypatch):
    monkeypatch.setattr(qb_router, "_llm", lambda: essay_llm)
    r = client.post(url(tail="/generate"), json={"n": 1, "kinds": ["essay"]}, headers=H())
    assert r.status_code == 201, r.text
    entry = r.json()["items"][0]
    assert entry["item"]["kind"] == "essay"
    assert entry["grading"] == GUIDE


def test_a_bad_llm_reply_falls_back_to_the_rubric_guide(client, monkeypatch):
    def bad_guide_llm(prompt):
        return "sorry, no JSON here" if prompt.startswith("You write a grading guide") else json.dumps(ESSAY)

    monkeypatch.setattr(qb_router, "_llm", lambda: bad_guide_llm)
    r = client.post(url(tail="/generate"), json={"n": 1, "kinds": ["essay"]}, headers=H())
    guide = r.json()["items"][0]["grading"]
    assert "softmax" in guide["correct"] and "softmax" in guide["partial"]     # built from the rubric
    assert guide["wrong"]


def test_editing_the_options_updates_an_mcq_guide(client):
    item_id = generate(client, n=1)[0]
    r = client.put(url(item_id), json={"options": ["new a", "new b", "new c", "new d"], "right_choice": 2}, headers=H())
    assert r.status_code == 200
    assert "new c" in r.json()["grading"]["correct"]


def test_teacher_can_write_the_guide_by_hand(client):
    item_id = generate(client, n=1)[0]
    mine = {"correct": "my correct", "partial": "my partial", "wrong": "my wrong"}
    r = client.put(url(item_id), json={"grading": mine}, headers=H())
    assert r.status_code == 200
    assert r.json()["grading"] == mine


def test_a_guide_with_an_empty_level_is_rejected(client):
    item_id = generate(client, n=1)[0]
    r = client.put(url(item_id), json={"grading": {"correct": "x", "partial": "", "wrong": "y"}}, headers=H())
    assert r.status_code == 422


def test_regenerate_fills_a_missing_guide_and_checks_the_owner(client, svc):
    item_id = generate(client, n=1)[0]
    with svc._session() as db:
        row = db.query(qb_service.BankItemRow).filter_by(item_id=item_id).one()
        row.grading = {}
        db.commit()
    assert client.get(url(item_id), headers=H()).json()["grading"] is None

    assert client.post(url(item_id, tail="/grading"), headers=H("t2")).status_code == 403
    r = client.post(url(item_id, tail="/grading"), headers=H())
    assert r.status_code == 200
    assert r.json()["grading"]["correct"]


def test_an_old_database_without_the_grading_column_still_works(tmp_path):
    import sqlalchemy as sa

    db_url = f"sqlite:///{tmp_path}/old.db"
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(sa.text(
            "CREATE TABLE m6_question_bank (id INTEGER PRIMARY KEY, item_id VARCHAR(80), room_id VARCHAR(64), "
            "teacher_id VARCHAR(64), kind VARCHAR(16), stem TEXT, options JSON, right_choice INTEGER, answer TEXT, "
            "rubric JSON, explanation TEXT, file_name VARCHAR(256), page INTEGER, topic_id VARCHAR(64), "
            "approved BOOLEAN, created_at DATETIME)"))
        conn.execute(sa.text(
            "INSERT INTO m6_question_bank VALUES (1, 'old-1', 'room-1', 't1', 'mcq', 'An old question here?', "
            "'[\"a\", \"b\", \"c\", \"d\"]', 0, '', '[]', 'why', 'book.pdf', 1, 'a', 0, '2026-01-01 00:00:00.000000')"))

    old = QuestionBankService(db_url)                     # adds the columns instead of crashing
    entry = old.list_items("room-1", teacher_id="t1")[0]
    assert entry.item.item_id == "old-1" and entry.grading is None and entry.difficulty is None
    assert old.backfill("room-1", "t1", None) == 1        # no LLM: the plain guide and a level from the question type
    done = old.get_item("old-1")
    assert done.grading is not None and done.difficulty == "medium"
    assert old.backfill("room-1", "t1", None) == 0        # nothing left to fill


# ---------- difficulty ----------

def make_item(kind="mcq", item_id="d1"):
    if kind == "essay":
        return QuizItem(item_id=item_id, topic_id="a", stem="Explain how self-attention works.", explanation="e",
                        file_name="b.pdf", page=1, kind="essay", answer="a" * 50, rubric=["x", "y", "z"])
    return QuizItem(item_id=item_id, topic_id="a", stem="What does a tokenizer do?", options=["w", "x", "y", "z"],
                    right_choice=0, explanation="e", file_name="b.pdf", page=1)


def test_difficulty_uses_the_llm_rating():
    items = [make_item(item_id="a"), make_item(item_id="b"), make_item(item_id="c")]
    levels = qb_difficulty.classify_difficulties(items, lambda p: json.dumps({"levels": ["easy", "hard", "medium"]}))
    assert levels == {"a": "easy", "b": "hard", "c": "medium"}


def test_difficulty_fills_the_gaps_from_the_question_type():
    items = [make_item("mcq", "a"), make_item("essay", "b")]
    # the LLM rates the first one with an unknown word and forgets the second
    levels = qb_difficulty.classify_difficulties(items, lambda p: json.dumps({"levels": ["very hard"]}))
    assert levels == {"a": "medium", "b": "hard"}
    assert qb_difficulty.classify_difficulties(items, None) == {"a": "medium", "b": "hard"}
    assert qb_difficulty.classify_difficulties(items, lambda p: "no json") == {"a": "medium", "b": "hard"}


def test_difficulty_rates_big_batches_in_groups_of_ten():
    calls = []

    def llm(prompt):
        calls.append(prompt)
        return json.dumps({"levels": ["easy"] * 10})

    items = [make_item(item_id=f"i{i}") for i in range(23)]
    levels = qb_difficulty.classify_difficulties(items, llm)
    assert len(calls) == 3 and len(levels) == 23


def test_generated_questions_are_rated(client):
    generate(client, n=2)
    for entry in client.get(url(), headers=H()).json()["items"]:
        assert entry["difficulty"] == "medium"          # fake llm gives no ratings: an mcq falls back to medium


def test_teacher_can_change_the_difficulty(client):
    item_id = generate(client, n=1)[0]
    r = client.put(url(item_id), json={"difficulty": "hard"}, headers=H())
    assert r.json()["difficulty"] == "hard"
    assert client.put(url(item_id), json={"difficulty": "impossible"}, headers=H()).status_code == 422


# ---------- reading a request (NLU) ----------

@pytest.mark.parametrize("text, n, difficulty, kinds, topic", [
    ("give me 5 medium questions about photosynthesis", 5, "medium", [], "photosynthesis"),
    ("هاتلي ٥ اسئلة صعبة عن البناء الضوئي", 5, "hard", [], "البناء الضوئي"),
    ("3 easy true/false questions on gravity", 3, "easy", ["true_false"], "gravity"),
    ("عايز سؤالين مقالي سهل عن الشبكات العصبية", 2, "easy", ["essay"], "الشبكات العصبية"),
    ("اديني اسئلة اختيار من متعدد متوسطة حول الخلية", None, "medium", ["mcq"], "الخلية"),
    ("five hard math problems on integrals", 5, "hard", ["math", "problem"], "integrals"),
    ("10 mcq", 10, None, ["mcq"], ""),
    ("عايزة خمس أسئلة اختيار من متعدد معتمدة", 5, None, ["mcq"], ""),
    ("هاتلي تلات اسئلة صعبة عن الخلايا", 3, "hard", [], "الخلايا"),
    ("عايزة سؤال واحد بس مقالي صعب", 1, "hard", ["essay"], ""),
    ("كام سؤال عندك في الميتوكوندريا", None, None, [], "الميتوكوندريا"),
])
def test_rules_read_the_request(text, n, difficulty, kinds, topic):
    intent = nlu.parse_with_rules(text)
    assert (intent.n, intent.difficulty, intent.kinds, intent.topic_query) == (n, difficulty, kinds, topic)


@pytest.mark.parametrize("text, n, levels, topic", [
    ("عايزة سؤالين عن كليب واحد صعب وواحد سهل", 2, ["hard", "easy"], "كليب"),
    ("give me 2 question about clip one of them hard and the other easy", 2, ["hard", "easy"], "clip"),
    ("one easy one medium and one hard question about cells", None, ["easy", "medium", "hard"], "cells"),
])
def test_rules_read_a_mix_of_levels(text, n, levels, topic):
    intent = nlu.parse_with_rules(text)
    assert intent.difficulty is None and intent.difficulties == levels and intent.topic_query == topic
    if n:
        assert intent.n == n


def test_a_word_like_simple_inside_the_subject_is_not_a_mix():
    intent = nlu.parse_with_rules("hard questions about simple harmonic motion")
    assert intent.difficulty == "hard" and intent.difficulties == []


def test_the_llm_cannot_cancel_a_mix():
    text = "2 questions about clip, one hard and one easy"
    intent = nlu.parse_query(text, _llm_reply(difficulty="hard", difficulty_word="hard", topic_query="clip"))
    assert intent.difficulty is None and intent.difficulties == ["hard", "easy"] and intent.topic_query == "clip"


def test_the_answer_line_says_exact_related_and_missing():
    assert nlu.result_message("clip", 2, 2, related=1) == "Found 1 question on the subject, plus 1 related."
    assert nlu.result_message("clip", 1, 1, related=1) == "No exact match. Here is 1 related question."
    assert nlu.result_message("clip", 2, 1, missing=["easy"]) == "Found 1 of the 2 questions you asked for. No easy question available."
    assert nlu.result_message("عن كليب", 2, 2, related=1) == "وجدت 1 سؤال عن الموضوع، و1 سؤال قريب منه."


def test_rules_read_approved_only():
    intent = nlu.parse_with_rules("show approved essay questions about attention")
    assert intent.approved_only and intent.kinds == ["essay"] and intent.topic_query == "attention"


def test_the_llm_reading_wins_and_the_rules_fill_the_gaps():
    reply = json.dumps({"n": None, "difficulty": "hard", "difficulty_word": "hard", "kinds": ["essay", "banana"],
                        "topic_query": "self attention", "approved_only": False})
    intent = nlu.parse_query("give me 4 hard questions on attention", lambda p: reply)
    assert intent.difficulty == "hard"                 # from the LLM, with the word that says it
    assert intent.kinds == ["essay"]                   # the unknown type is dropped
    assert intent.topic_query == "self attention"      # from the LLM
    assert intent.n == 4                               # the LLM said nothing, the rules found the 4


def test_a_json_reply_that_is_not_a_reading_is_ignored():
    intent = nlu.parse_query("hard questions about tokenizer", lambda p: json.dumps({"question": "x", "answer": "y"}))
    assert intent.difficulty == "hard" and intent.topic_query == "tokenizer"


def _llm_reply(**fields):
    base = {"n": None, "difficulty": None, "difficulty_word": "", "kinds": [], "topic_query": "", "approved_only": False}
    return lambda prompt: json.dumps({**base, **fields})


def test_the_llm_cannot_guess_a_difficulty_nobody_asked_for():
    # a noisy transcript of "5 approved multiple choice": the LLM invents "medium" and finds no subject
    text = "أعزة خمس أسئلات اختيار من متعدد ومعتقدة"
    intent = nlu.parse_query(text, _llm_reply(n=5, difficulty="medium", kinds=["mcq"]))
    assert intent.difficulty is None                   # no word in the request says "medium"
    assert intent.topic_query == ""                    # the rules' leftover words are not a subject
    assert intent.n == 5 and intent.kinds == ["mcq"]


def test_the_llm_can_read_a_difficulty_word_the_rules_do_not_know():
    intent = nlu.parse_query("هاتلي أسئلة تقيلة عن الخلية", _llm_reply(difficulty="hard", difficulty_word="تقيلة", topic_query="الخلية"))
    assert intent.difficulty == "hard" and intent.topic_query == "الخلية"


def test_simple_inside_a_subject_is_not_a_difficulty_when_the_llm_reads_it():
    intent = nlu.parse_query("2 questions about simple harmonic motion", _llm_reply(topic_query="simple harmonic motion"))
    assert intent.difficulty is None and intent.topic_query == "simple harmonic motion"


def test_a_broken_llm_reply_falls_back_to_the_rules():
    rules = nlu.parse_with_rules("2 hard questions about gravity")
    assert nlu.parse_query("2 hard questions about gravity", lambda p: "I cannot do that") == rules
    assert nlu.parse_query("2 hard questions about gravity", None) == rules

    def boom(prompt):
        raise RuntimeError("down")

    assert nlu.parse_query("2 hard questions about gravity", boom) == rules


def test_the_llm_cannot_make_up_a_difficulty_or_a_huge_number():
    reply = json.dumps({"n": 5000, "difficulty": "very hard", "kinds": [], "topic_query": "gravity"})
    intent = nlu.parse_query("questions about gravity", lambda p: reply)
    assert intent.n is None and intent.difficulty is None and intent.topic_query == "gravity"


# ---------- asking the bank in your own words ----------

def seed_three(client):
    """3 tokenizer questions: one hard, two medium; the first medium one is approved."""
    ids = generate(client, n=3)
    client.put(url(ids[0]), json={"difficulty": "hard"}, headers=H())
    client.put(url(ids[1]), json={"difficulty": "medium"}, headers=H())
    client.put(url(ids[2]), json={"difficulty": "medium"}, headers=H())
    client.patch(url(ids[1], tail="/approve"), json={"approved": True}, headers=H())
    return ids


def ask(client, text, teacher="t1"):
    r = client.get(url(tail="/search"), params={"q": text}, headers=H(teacher))
    assert r.status_code == 200, r.text
    return r.json()


def test_ask_for_fewer_than_exist(client):
    seed_three(client)
    body = ask(client, "give me 2 medium questions about tokenizer")
    assert body["total"] == 2 and body["requested"] == 2
    assert body["understood"]["difficulty"] == "medium"
    assert all(i["difficulty"] == "medium" for i in body["items"])


def test_ask_for_more_than_exist_says_how_many_were_found(client):
    seed_three(client)
    body = ask(client, "5 medium questions about tokenizer")
    assert body["total"] == 2
    assert body["message"] == "Found 2 of the 5 questions you asked for."
    ar = ask(client, "هاتلي ٥ اسئلة متوسطة عن tokenizer")
    assert ar["message"] == "وجدت 2 من 5 أسئلة طلبتها."


def test_ask_filters_by_difficulty_type_and_approval(client):
    seed_three(client)
    assert ask(client, "hard questions about tokenizer")["total"] == 1
    assert ask(client, "hard essay questions about tokenizer")["total"] == 0
    assert ask(client, "approved questions about tokenizer")["total"] == 1
    assert ask(client, "mcq questions about tokenizer")["total"] == 3
    assert ask(client, "hard questions about photosynthesis")["total"] == 0


def test_ask_without_a_number_returns_up_to_ten(client):
    generate(client, n=3)
    assert ask(client, "questions about tokenizer")["total"] == 3


def test_ask_only_sees_my_questions(client):
    seed_three(client)
    assert ask(client, "questions about tokenizer", teacher="t2")["total"] == 0


def test_ask_works_when_there_is_no_llm(client, monkeypatch):
    seed_three(client)

    def no_llm():
        raise qb_router.HTTPException(503, "no llm")

    monkeypatch.setattr(qb_router, "_llm", no_llm)
    body = ask(client, "2 hard questions about tokenizer")
    assert body["total"] == 1 and body["understood"]["n"] == 2


def test_questions_without_a_rating_are_left_out_of_a_difficulty_search(client, svc):
    ids = generate(client, n=2)
    with svc._session() as db:
        for row in db.query(qb_service.BankItemRow).all():
            row.difficulty = ""
        db.commit()
    assert ask(client, "medium questions about tokenizer")["total"] == 0
    assert ask(client, "questions about tokenizer")["total"] == 2
    assert client.post(url(tail="/backfill"), headers=H()).json() == {"updated": 2}
    assert ask(client, "medium questions about tokenizer")["total"] == 2


# ---------- voice ----------

def audio():
    return {"audio": ("request.webm", b"fake-audio-bytes", "audio/webm")}


def test_voice_search_reads_the_spoken_request(client, monkeypatch):
    seed_three(client)
    heard = {}

    def fake_transcribe(data, filename, content_type, language):
        heard.update(data=data, language=language)
        return "give me 1 hard question about tokenizer"

    monkeypatch.setattr(qb_router, "transcribe", fake_transcribe)
    r = client.post(url(tail="/search/voice"), files=audio(), data={"language": "en"}, headers=H())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["transcript"] == "give me 1 hard question about tokenizer"
    assert body["total"] == 1 and body["items"][0]["difficulty"] == "hard"
    assert heard == {"data": b"fake-audio-bytes", "language": "en"}


def test_voice_search_errors(client, monkeypatch):
    def unavailable(*a):
        raise stt.SttUnavailable("no key")

    def broken(*a):
        raise stt.SttError("down")

    monkeypatch.setattr(qb_router, "transcribe", unavailable)
    assert client.post(url(tail="/search/voice"), files=audio(), headers=H()).status_code == 503
    monkeypatch.setattr(qb_router, "transcribe", broken)
    assert client.post(url(tail="/search/voice"), files=audio(), headers=H()).status_code == 502
    monkeypatch.setattr(qb_router, "transcribe", lambda *a: "   ")
    assert client.post(url(tail="/search/voice"), files=audio(), headers=H()).status_code == 422
    empty = {"audio": ("request.webm", b"", "audio/webm")}
    assert client.post(url(tail="/search/voice"), files=empty, headers=H()).status_code == 422


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {"text": "  hello  "}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def clear_stt_env(monkeypatch):
    for name in ("M6_STT_API_KEY", "GROQ_API_KEY", "LLM_API_KEY", "M6_STT_BASE_URL", "LLM_BASE_URL", "M6_STT_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_stt_calls_whisper_on_groq(monkeypatch):
    clear_stt_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    seen = {}

    def fake_post(url_, headers, files, data, timeout):
        seen.update(url=url_, headers=headers, data=data, filename=files["file"][0])
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    assert stt.transcribe(b"abc", "a.webm", "audio/webm", "ar") == "hello"
    assert seen["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert seen["headers"] == {"Authorization": "Bearer secret"}
    assert seen["data"]["model"] == "whisper-large-v3-turbo" and seen["data"]["language"] == "ar"


def test_stt_without_a_key_or_when_the_service_fails(monkeypatch):
    clear_stt_env(monkeypatch)
    with pytest.raises(stt.SttUnavailable):
        stt.transcribe(b"abc")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setattr("httpx.post", lambda *a, **k: FakeResponse(429, {"error": "rate limit"}))
    with pytest.raises(stt.SttError):
        stt.transcribe(b"abc")


# ---------- backfill ----------

def test_backfill_only_touches_my_questions(client, svc):
    generate(client, teacher="t1", n=1)
    with svc._session() as db:
        for row in db.query(qb_service.BankItemRow).all():
            row.grading = {}
            row.difficulty = ""
        db.commit()
    assert client.post(url(tail="/backfill"), headers=H("t2")).json() == {"updated": 0}
    assert client.post(url(tail="/backfill"), headers=H("t1")).json() == {"updated": 1}
    entry = client.get(url(), headers=H("t1")).json()["items"][0]
    assert entry["grading"] and entry["difficulty"] == "medium"


# ---------- exact matches vs related questions, and "one hard and one easy" ----------

def test_a_question_from_the_same_topic_is_marked_related(svc, monkeypatch):
    from rafeeq.knowledge import integration as ki
    from rafeeq.knowledge import service as ks_module

    for page, stem in ((1, "Question on page 1"), (4, "Question on page 4")):
        item = QuizItem(item_id=f"i{page}", topic_id="a", stem=stem, options=["w", "x", "y", "z"],
                        right_choice=0, explanation="e", file_name="book.pdf", page=page)
        with svc._session() as db:
            db.add(qb_service._item_to_row(item, ROOM, "t1", difficulty="medium"))
            db.commit()

    class FakeKnowledge:
        def search(self, room_id, query, top_k=5):
            return {"chunks": [{"file_name": "book.pdf", "page": 4, "text": "t", "topic_id": "a"}]}

    monkeypatch.setattr(ki, "enabled", lambda: True)
    monkeypatch.setattr(ks_module, "get_service", lambda: FakeKnowledge())

    result = svc.nlu_search(ROOM, "2 questions about clip", teacher_id="t1")
    assert [e.item.item_id for e in result.entries] == ["i4", "i1"]
    assert result.related_ids == frozenset({"i1"})          # i4 is on the retrieved page, i1 only shares the topic


def test_one_hard_and_one_easy_gives_one_of_each(svc):
    for number, level in enumerate(["hard", "easy", "easy", "medium"], 1):
        item = QuizItem(item_id=f"m{number}", topic_id="a", stem=f"tokenizer question {number}", options=["w", "x", "y", "z"],
                        right_choice=0, explanation="e", file_name="book.pdf", page=number)
        with svc._session() as db:
            db.add(qb_service._item_to_row(item, ROOM, "t1", difficulty=level))
            db.commit()

    result = svc.nlu_search(ROOM, "2 questions about tokenizer, one hard and one easy", teacher_id="t1")
    assert sorted(e.difficulty for e in result.entries) == ["easy", "hard"] and result.missing == ()


def test_a_missing_level_of_a_mix_is_reported(client):
    seed_three(client)                                       # one hard, two medium, no easy
    body = ask(client, "2 questions about tokenizer, one hard and one easy")
    assert body["total"] == 1 and body["items"][0]["difficulty"] == "hard"
    assert body["missing_levels"] == ["easy"]
    assert body["message"].endswith("No easy question available.")

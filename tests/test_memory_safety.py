"""Chat history per account, the short memory of a chat, the long memory that follows the student, the web search narrowed by what they study, and the
refusal to search for anything illegal or 18+. Real app and routes; only the model, the web, M2 and M4 are replaced."""
import re

import pytest
from fastapi.testclient import TestClient
from test_tutor_agent import HX, csrf_of, model, room, world  # noqa: F401  (fixtures and helpers)
from test_web import register

from rafeeq.main import app
from rafeeq.tutor import chats, memory, m2, safety, study_store, tutor_agent, web_search
from rafeeq.tutor import llm as llm_mod


def chat_say(student, rid, chat, text, **extra):
    page = student.get(f"/app/rooms/{rid}/study", params={"chat": chat})
    return student.post(f"/app/rooms/{rid}/study/message", data={"text": text, **extra},
                        headers={**HX, "X-CSRF-Token": csrf_of(page.text), "X-Chat-Id": chat})


def owner_of(chat_id):
    with study_store.db() as con:
        return con.execute("SELECT user_id FROM chats WHERE id=?", (chat_id,)).fetchone()["user_id"]


def second_student(teacher, rid, email="sara2@x.com"):
    code = re.search(r'join-code-value" dir="ltr">([A-Z0-9-]+)<', teacher.get(f"/app/rooms/{rid}/overview").text).group(1)
    sara = TestClient(app)
    register(sara, "student", "Sara", email)
    sara.post("/app/rooms/join", data={"csrf_token": csrf_of(sara.get("/app/rooms").text), "code": code})
    return sara


# ----------------------------------------------------------------------------- chat history, one account at a time
def test_every_chat_is_saved_for_the_account_and_can_be_reopened_or_deleted(room):
    _, omar, rid = room
    c1, c2 = chats.new_id(), chats.new_id()
    chat_say(omar, rid, c1, "What does the mitochondria do?")
    chat_say(omar, rid, c1, "and where is it?")
    chat_say(omar, rid, c2, "ما وظيفة الميتوكوندريا؟")
    page = omar.get(f"/app/rooms/{rid}/study").text                                       # a plain visit is a NEW chat; the old ones are in the list
    assert f"chat={c1}" in page and f"chat={c2}" in page and 'class="tq-chatlist"' in page and 'id="chat-empty"' in page
    reopened = omar.get(f"/app/rooms/{rid}/study", params={"chat": c1}).text
    assert "What does the mitochondria do?" in reopened and "and where is it?" in reopened and 'id="chat-empty"' not in reopened
    log = reopened.split('id="chat-log"')[1].split('id="chat-indicator"')[0]
    assert "ما وظيفة" not in log                                                           # the other chat's turns are not mixed in
    assert [c["turns"] for c in chats.list_chats(owner_of(c1), rid) if c["id"] == c1] == [2]
    assert omar.post(f"/app/rooms/{rid}/study/chats/{c1}/delete", data={}).status_code == 403
    assert omar.post(f"/app/rooms/{rid}/study/chats/{c1}/delete", data={"csrf_token": csrf_of(reopened)}).status_code == 200
    assert "What does the mitochondria do?" not in omar.get(f"/app/rooms/{rid}/study", params={"chat": c1}).text
    assert f"chat={c2}" in omar.get(f"/app/rooms/{rid}/study").text


def test_a_chat_belongs_to_its_account_only(room):
    teacher, omar, rid = room
    sara = second_student(teacher, rid)
    c = chats.new_id()
    chat_say(omar, rid, c, "What does the mitochondria do?")
    assert "What does the mitochondria do?" not in sara.get(f"/app/rooms/{rid}/study", params={"chat": c}).text    # Sara gets a blank chat
    chat_say(sara, rid, c, "my private question")                                                                  # and cannot write into Omar's chat
    assert "my private question" not in omar.get(f"/app/rooms/{rid}/study", params={"chat": c}).text
    sara.post(f"/app/rooms/{rid}/study/chats/{c}/delete", data={"csrf_token": csrf_of(sara.get(f"/app/rooms/{rid}/study").text)})
    assert "What does the mitochondria do?" in omar.get(f"/app/rooms/{rid}/study", params={"chat": c}).text         # still there


def test_a_level_2_study_file_turn_is_saved_in_the_chat_too(room):
    _, omar, rid = room
    c = chats.new_id()
    page = omar.get(f"/app/rooms/{rid}/study", params={"chat": c})
    omar.post(f"/app/rooms/{rid}/study/agent", data={"action": "research", "text": "what is ATP", "pages": "1"},
              headers={**HX, "X-CSRF-Token": csrf_of(page.text), "X-Chat-Id": c})
    assert "/study-files/" in omar.get(f"/app/rooms/{rid}/study", params={"chat": c}).text


# ----------------------------------------------------------------------------- short memory
def test_the_short_memory_understands_a_follow_up_inside_the_same_chat_only(room, monkeypatch):
    _, omar, rid = room
    prompts = []
    monkeypatch.setattr(llm_mod, "ask_json", lambda p, max_tokens=None: prompts.append(p) or model(p, max_tokens))
    c = chats.new_id()
    chat_say(omar, rid, c, "What does the mitochondria do?")
    prompts.clear()
    chat_say(omar, rid, c, "and where is it?")
    assert any("Conversation so far" in p and "mitochondria" in p for p in prompts)         # the follow-up is rewritten with the chat so far
    prompts.clear()
    chat_say(omar, rid, chats.new_id(), "and where is it?")                                 # a NEW chat has no short memory of that one
    assert not any("Conversation so far" in p for p in prompts)


def test_the_short_memory_keeps_only_the_last_turns(room):
    _, omar, rid = room
    c = chats.new_id()
    for i in range(7):
        chat_say(omar, rid, c, f"question number {i}")
    hist = chats.history(owner_of(c), rid, c)
    assert len(hist) == 2 * chats.HISTORY_TURNS and hist[-2]["text"] == "question number 6"


# ----------------------------------------------------------------------------- long memory: important things are kept by themselves
FACTS = {"facts": [{"kind": "study", "text": "first year medicine student", "field_en": "medicine"}, {"kind": "name", "text": "Omar"},
                   {"kind": "difficulty", "text": "finds biochemistry hard"}]}


def remembering_model(prompts=None):
    def ask(p, max_tokens=None):
        if prompts is not None:
            prompts.append(p)
        if "worth remembering" in p:
            return FACTS if "medicine" in p else {"facts": []}
        return model(p, max_tokens)
    return ask


def test_important_things_the_student_says_are_kept_without_being_asked(room, monkeypatch):
    _, omar, rid = room
    monkeypatch.setattr(llm_mod, "ask_json", remembering_model())
    c = chats.new_id()
    chat_say(omar, rid, c, "I'm a first year medicine student and biochemistry is hard for me. What does the mitochondria do?")
    kept = {n["kind"]: n["note"] for n in memory.notes(owner_of(c))}
    assert kept["name"] == "Omar" and "medicine" in kept["study"] and "biochemistry" in kept["difficulty"]
    assert memory.profile(owner_of(c))["field_en"] == "medicine"
    panel = omar.get(f"/app/rooms/{rid}/study").text
    assert "tq-notes" in panel and "finds biochemistry hard" in panel                                # the student sees what was kept


def test_a_plain_question_keeps_nothing_and_does_not_even_ask_the_model(room, monkeypatch):
    _, omar, rid = room
    prompts = []
    monkeypatch.setattr(llm_mod, "ask_json", remembering_model(prompts))
    c = chats.new_id()
    chat_say(omar, rid, c, "What does the mitochondria do?")                                          # nothing about the student: no extraction call at all
    assert not any("worth remembering" in p for p in prompts) and memory.notes(owner_of(c)) == []
    chat_say(omar, rid, c, "I like short answers please")                                           # about the student, but the model finds nothing important in it
    assert memory.notes(owner_of(c)) == []


def test_the_notes_follow_the_student_into_a_new_chat(room, monkeypatch):
    _, omar, rid = room
    prompts = []
    monkeypatch.setattr(llm_mod, "ask_json", remembering_model(prompts))
    chat_say(omar, rid, chats.new_id(), "I'm a first year medicine student. What does the mitochondria do?")
    prompts.clear()
    chat_say(omar, rid, chats.new_id(), "What does the mitochondria do?")                           # a brand new chat
    assert any("Remembered from earlier chats" in p and "medicine" in p and "biochemistry" in p for p in prompts)
    assert any("Lately they asked about" in p for p in prompts)                                      # the simple long memory of the history


def test_a_newer_study_fact_replaces_the_old_one_but_other_notes_pile_up(world):
    memory.add_note("u1", "study", "first year medicine", "medicine")
    memory.add_note("u1", "study", "second year medicine", "medicine")
    memory.add_note("u1", "goal", "exam on Friday")
    memory.add_note("u1", "goal", "Exam on friday")                                                  # the same note again: kept once
    assert [(n["kind"], n["note"]) for n in memory.notes("u1")] == [("study", "second year medicine"), ("goal", "exam on Friday")]
    for i in range(40):
        memory.add_note("u1", "other", f"fact {i}")
    assert len(memory.notes("u1")) == memory.MAX_NOTES


def test_the_student_can_delete_a_note_or_forget_everything(room, monkeypatch):
    _, omar, rid = room
    monkeypatch.setattr(llm_mod, "ask_json", remembering_model())
    c = chats.new_id()
    chat_say(omar, rid, c, "I'm a first year medicine student. What does the mitochondria do?")
    uid = owner_of(c)
    gone = next(n for n in memory.notes(uid) if n["kind"] == "difficulty")
    token = csrf_of(omar.get(f"/app/rooms/{rid}/study").text)
    assert omar.post(f"/app/rooms/{rid}/study/memory/{gone['id']}/delete", data={}).status_code == 403
    r = omar.post(f"/app/rooms/{rid}/study/memory/{gone['id']}/delete", data={"csrf_token": token})
    assert r.status_code == 200 and "finds biochemistry hard" not in r.text
    assert gone["id"] not in [n["id"] for n in memory.notes(uid)] and len(memory.notes(uid)) == 2
    r = omar.post(f"/app/rooms/{rid}/study/memory/forget", data={"csrf_token": token})
    assert memory.notes(uid) == [] and "tq-notes" not in r.text


def test_the_notes_are_per_account(room, monkeypatch):
    teacher, omar, rid = room
    sara = second_student(teacher, rid)
    monkeypatch.setattr(llm_mod, "ask_json", remembering_model())
    c = chats.new_id()
    chat_say(omar, rid, c, "I'm a first year medicine student. what is a cell?")
    assert "first year medicine student" in omar.get(f"/app/rooms/{rid}/study").text
    assert "first year medicine student" not in sara.get(f"/app/rooms/{rid}/study").text
    omar_note = memory.notes(owner_of(c))[0]["id"]
    token = csrf_of(sara.get(f"/app/rooms/{rid}/study").text)
    sara.post(f"/app/rooms/{rid}/study/memory/{omar_note}/delete", data={"csrf_token": token})        # Sara tries to delete Omar's note
    assert any(n["id"] == omar_note for n in memory.notes(owner_of(c)))


def test_what_the_student_says_is_only_checked_when_it_is_about_them():
    assert memory.looks_personal("I'm a medicine student") and memory.looks_personal("انا طالب في كلية الطب") and memory.looks_personal("اسمي محمد")
    assert not memory.looks_personal("what does the mitochondria do?") and not memory.looks_personal("ما وظيفة الميتوكوندريا؟")


def test_the_web_search_is_narrowed_by_what_the_student_studies(world, monkeypatch):
    queries = []
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: queries.append(q) or [{"title": "T", "url": "https://e.org/1", "content": "Osmosis is the movement of water. " * 4, "score": 1}])
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    memory.add_note("u_med", "study", "طالب طب", "medicine")
    token = memory.CURRENT.set(memory.context_for("u_med", "r", None))
    try:
        tutor_agent.act("r", "auto", "what is osmosis?", None, "en")
        assert queries == ["osmosis medicine"]
        queries.clear()
        tutor_agent.act("r", "research", "how does osmosis work", None, "en", pages=1)
        assert queries and all("medicine" in q.lower() for q in queries[:1])                         # the study file's queries carry the field
    finally:
        memory.CURRENT.reset(token)


# ----------------------------------------------------------------------------- nothing illegal or 18+ is searched
@pytest.mark.parametrize("q", ["free porn videos", "how to make a bomb at home", "شراء مخدرات", "سكس", "hack someone's instagram account", "تصنيع قنبلة"])
def test_illegal_and_adult_searches_are_blocked_by_the_rules(world, monkeypatch, q):
    monkeypatch.setattr(web_search, "_tavily_http", lambda *a: pytest.fail("a blocked search must never reach Tavily"))
    with pytest.raises(safety.Blocked):
        web_search.multi_search([q])


@pytest.mark.parametrize("q", ["reproductive system anatomy", "criminal law: what is fraud", "history of the opium wars", "how does a firewall work", "ما وظيفة الجهاز التناسلي"])
def test_study_questions_about_sensitive_subjects_are_not_blocked_by_the_rules(world, q):
    assert safety.check([q]) is None


def test_the_model_judges_what_the_rules_cannot(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda p, max_tokens=None: {"block": True, "category": "illegal"} if "must be BLOCKED" in p else None)
    assert safety.check(["some cleverly worded request"]) == "illegal"
    monkeypatch.setattr(llm_mod, "ask_json", lambda p, max_tokens=None: {"block": False, "category": "none"})
    assert safety.check(["some cleverly worded request"]) is None


def test_a_blocked_study_file_is_refused_in_the_students_language_and_nothing_is_searched(world, monkeypatch):
    monkeypatch.setattr(web_search, "_tavily_http", lambda *a: pytest.fail("never"))
    out = tutor_agent.act("r_bio101", "research", "how to make a bomb at home", None, "ar", pages=1)
    assert "غير قانوني" in (out["result"]["error"] or "") and out["result"]["study_file"] is None
    out = tutor_agent.act("r_bio101", "research", "how to make a bomb at home", None, "en", pages=1)
    assert "illegal" in out["result"]["error"] and study_store.list_study_files("r_bio101") == []


def test_a_photographed_term_that_is_not_allowed_is_refused_not_searched(world, monkeypatch):
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    monkeypatch.setattr(web_search, "_tavily_http", lambda *a: pytest.fail("never"))
    out = tutor_agent.act("r_bio101", "explain", "free porn videos", None, "en")
    assert out["result"]["grounded"] is False and "illegal or not appropriate" in out["result"]["error"]

"""The tutor agent under M1's Copilot, in M1's green app: Level 1 and 2, the 3 MCQ, summary, flashcards, study files, hints, snap, and what students
ask again and again reaching the teacher's class insights. Real app, real routes; only the model, the web, M2 and M4 are replaced."""
import re

import pytest
from fastapi.testclient import TestClient
from test_member3 import FakeReader, _photo, assert_contract  # noqa: F401
from test_web import csrf_of, student_in_room, teacher_with_room  # noqa: F401  (M1's helpers for driving the real forms)

from rafeeq.copilot import clients
from rafeeq.main import app
from rafeeq.platform import service
from rafeeq.snap import service as snap_service
from rafeeq.tutor import anonymous, connect, extras, m2, study_store, tutor_agent, web_search
from rafeeq.tutor import llm as llm_mod

HX = {"HX-Request": "true"}


def model(prompt, max_tokens=None):
    """A stand-in for the LLM that answers the prompts of the tools by what they ask for."""
    if "SAME subject" in prompt:                                  # the relevance check on a photo's text
        return {"related": "Westphalia" not in prompt}
    if "Summarise" in prompt:
        return {"points": ["Mitochondria make ATP", "ATP is the cell's energy"]}
    if "flashcards" in prompt:
        return {"cards": [{"front": "What makes ATP?", "back": "Mitochondria"}]}
    if "Explain it again" in prompt or "Give ONE concrete" in prompt or "Compare it with" in prompt:
        return {"text": "Think of the mitochondria as a power plant."}
    if "multiple-choice" in prompt.lower() or "mcq" in prompt.lower():
        return {"quiz": [{"question": f"Q{i}?", "options": ["a", "b", "c"], "answer": 1, "explanation": "because"} for i in range(3)]}
    return None


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("RAFEEQ_DATA_DIR", str(tmp_path / "m1"))
    monkeypatch.setenv("RAFEEQ_WEB_SECRET", "test-secret")
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path / "study"))
    monkeypatch.setenv("SNAP_DATA_DIR", str(tmp_path / "snap"))
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.delenv("RAFEEQ_API_BASE", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    service.set_store(service.PlatformStore(tmp_path / "platform", seed_demo=True))
    down = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down"))
    monkeypatch.setattr(m2.requests, "post", down)
    monkeypatch.setattr(m2.requests, "get", down)
    monkeypatch.setattr(llm_mod, "ask_json", model)
    pages = [{"title": f"Source {i}", "url": f"https://example.org/{i}", "content": "A cell makes energy in the mitochondria as ATP. " * 4, "score": 1 - i / 10} for i in range(6)]
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: pages)
    connect.connect_m3(force=True)
    yield
    connect.disconnect_m3()
    service.set_store(None)


@pytest.fixture
def room(world):
    teacher, rid, code = teacher_with_room(app)
    student, _ = student_in_room(app, code)
    return teacher, student, rid


def say(student, rid, text):
    page = student.get(f"/app/rooms/{rid}/study")
    return student.post(f"/app/rooms/{rid}/study/message", data={"text": text}, headers={**HX, "X-CSRF-Token": csrf_of(page.text)})


def agent(student, rid, **form):
    page = student.get(f"/app/rooms/{rid}/study")
    return student.post(f"/app/rooms/{rid}/study/agent", data=form, headers={**HX, "X-CSRF-Token": csrf_of(page.text)})


# ----------------------------------------------------------------------------- the agent itself
def test_auto_sends_a_problem_to_hints_and_a_question_to_the_material(world):
    p = tutor_agent.act("r_bio101", "auto", "2x + 6 = 14", None, "en")
    assert p["action"] == "help" and p["ok"] and p["result"]["key"]
    q = tutor_agent.act("r_bio101", "auto", "What does the mitochondria do?", None, "en")
    assert q["action"] == "ask" and q["result"]["grounded"] and q["result"]["citations"]


def test_every_action_answers_in_one_shape_and_never_raises(world, monkeypatch):
    for action in tutor_agent.ACTIONS:
        out = tutor_agent.act("r", action, "text", None, "en")
        assert set(out) == {"action", "ok", "error", "result"}
    assert tutor_agent.act("r", "bogus")["ok"] is False
    monkeypatch.setattr(tutor_agent, "run_qa", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tutor_agent.act("r", "ask", "q")
    assert out["ok"] is False and out["error"] == "RuntimeError"


def test_the_act_route_matches_the_contract_and_is_strict(world):
    c = TestClient(app)
    body = c.post("/rooms/r_bio101/tutor/act", json={"action": "ask", "text": "What does the mitochondria do?", "language": "en"}).json()
    assert_contract("TutorActResponse", body)
    assert_contract("RoomAskResponse", body["result"])
    assert c.post("/rooms/r/tutor/act", json={"action": "ask", "nope": 1}).status_code == 422
    assert c.post("/rooms/r/tutor/act", json={"action": "fly"}).status_code == 422


def test_a_study_file_can_be_quizzed_summarised_and_turned_into_cards(world):
    asked = tutor_agent.act("r_bio101", "research", "what is ATP", None, "en", pages=1)
    fid = asked["result"]["study_file"]["id"]
    assert asked["ok"] and asked["result"]["study_file"]["pages"] >= 1 and asked["result"]["study_file"]["sources"]
    assert tutor_agent.act("r_bio101", "summary", "", None, "en", study_file_id=fid)["result"]["points"]
    assert tutor_agent.act("r_bio101", "cards", "", None, "en", study_file_id=fid)["result"]["cards"][0]["front"]
    assert len(tutor_agent.act("r_bio101", "quiz", "", None, "en", study_file_id=fid)["result"]["questions"]) == 3
    assert [f["id"] for f in tutor_agent.act("r_bio101", "files")["result"]["files"]] == [fid]
    assert tutor_agent.act("other_room", "files")["result"]["files"] == []                 # a room only sees its own files
    assert tutor_agent.act("other_room", "summary", "", None, "en", study_file_id=fid)["ok"] is False


def test_hints_come_one_at_a_time_then_the_solution(world):
    key = tutor_agent.act("r", "help", "2x + 6 = 14", None, "en")["result"]["key"]
    hints = [tutor_agent.act("r", "hint", "", None, "en", key=key, n=n)["result"] for n in (1, 2, 3)]
    assert [h["n"] for h in hints] == [1, 2, 3] and [h["next"] for h in hints] == ["hint", "hint", "solution"]
    assert tutor_agent.act("r", "solution", "", None, "en", key=key)["ok"]
    assert tutor_agent.act("r", "hint", "", None, "en", key="0" * 16)["ok"] is False


# ----------------------------------------------------------------------------- in M1's green Ask Rafeeq page
def test_an_answer_in_ask_rafeeq_offers_the_quiz_the_rewrites_and_go_deeper(room):
    _, student, rid = room
    html = say(student, rid, "What does the mitochondria do?").text
    assert 'class="tq-quiz"' not in html and 'class="tq-q"' not in html                                # no questions until the student asks
    assert '"action": "quiz"' in html and "tq-quizbar" in html                                # the button is there
    assert "tq-offer" in html and "/study/agent" in html and 'name="pages"' in html          # Level 2 offer with the page count
    assert "tq-rewrite-form" in html and 'value="simpler"' in html and 'value="example"' in html and 'value="compare"' in html
    assert "biology_ch3.pdf" in html                                                         # the citation chip
    quiz = agent(student, rid, action="quiz", topic_id="t_energy", question="What does the mitochondria do?").text
    assert quiz.count('class="tq-q"') == 3


def test_level_2_writes_a_study_file_with_its_tools_and_downloads(room):
    _, student, rid = room
    html = agent(student, rid, action="research", text="what is ATP", pages="1").text
    m = re.search(rf"/app/rooms/{rid}/study-files/(\d+)", html)
    assert m and "example.org" in html
    fid = m.group(1)
    assert 'hx-vals=\'{"action": "quiz"' in html and '"action": "summary"' in html and '"action": "cards"' in html
    listing = student.get(f"/app/rooms/{rid}/study-files").text
    assert f"/study-files/{fid}" in listing
    view = student.get(f"/app/rooms/{rid}/study-files/{fid}")
    assert view.status_code == 200 and "<h2>" in view.text
    assert student.get(f"/app/rooms/{rid}/study-files/{fid}/download", params={"format": "md"}).status_code == 200
    docx = student.get(f"/app/rooms/{rid}/study-files/{fid}/download", params={"format": "docx"})
    assert docx.status_code == 200 and docx.content[:2] == b"PK"
    assert student.get(f"/app/rooms/{rid}/study-files/99999").status_code == 404
    assert student.get(f"/app/rooms/{rid}/study-files/{fid}/download", params={"format": "exe"}).status_code == 404
    quiz = agent(student, rid, action="quiz", study_file_id=fid).text
    assert quiz.count('class="tq-q"') == 3
    assert "Mitochondria make ATP" in agent(student, rid, action="summary", study_file_id=fid).text
    assert "What makes ATP?" in agent(student, rid, action="cards", study_file_id=fid).text


def test_level_2_that_cannot_search_still_answers_from_the_material(room, monkeypatch):
    _, student, rid = room
    monkeypatch.delenv("TAVILY_API_KEY")
    monkeypatch.setattr(web_search, "multi_search", lambda *a, **k: (_ for _ in ()).throw(web_search.SearchError("TAVILY_API_KEY is not set.")))
    html = agent(student, rid, action="research", text="What does the mitochondria do?", pages="1").text
    # the student sees a plain sentence, never the key name (the reason is in the server log)
    assert "TAVILY_API_KEY" not in html and "البحث في الويب غير متاح الآن" in html
    assert "biology_ch3.pdf" in html and "study-files/" not in html


def test_rewrite_under_an_answer(room):
    _, student, rid = room
    html = agent(student, rid, action="rewrite", mode="simpler", question="q", text="Mitochondria make ATP.").text
    assert "power plant" in html
    assert agent(student, rid, action="rewrite", mode="bogus", text="x").status_code in (200, 422)


def test_a_problem_gets_the_hints_panel_and_each_click_one_piece(room):
    _, student, rid = room
    html = say(student, rid, "2x + 6 = 14").text
    key = re.search(r'help-([0-9a-f]{16})-step', html).group(1)
    assert "tq-help" in html and "tq-quiz" not in html
    h1 = agent(student, rid, action="hint", key=key, n="1").text
    assert "tq-hint" in h1 and '"n": 2' in h1
    h3 = agent(student, rid, action="hint", key=key, n="3").text
    assert '"action": "solution"' in h3
    assert "tq-solution" in agent(student, rid, action="solution", key=key).text
    assert "tq-hint" not in agent(student, rid, action="hint", key="not-a-key", n="1").text


def test_a_wrong_photo_gets_coloured_steps_the_hint_panel_and_the_real_box(room, monkeypatch):
    _, student, rid = room
    monkeypatch.setattr(snap_service, "get_reader", lambda name: FakeReader(["2 + 3 = 5", "4 × 2 = 9", "x = [?]"]))
    page = student.get(f"/app/rooms/{rid}/study")
    csrf = csrf_of(page.text)
    snap = student.post(f"/app/rooms/{rid}/study/photo", files={"photo": ("w.png", _photo(("2 + 3 = 5", "4 x 2 = 9", "x = 4")), "image/png")}, headers={**HX, "X-CSRF-Token": csrf})
    assert snap.status_code == 200, snap.text[:300]
    fid = re.search(r'name="file_id" value="([^"]+)"', snap.text).group(1)
    out = student.post(f"/app/rooms/{rid}/study/message", data={"file_id": fid, "mode": "check_work", "text": "2 + 3 = 5\n4 × 2 = 9\nx = [?]"}, headers={**HX, "X-CSRF-Token": csrf})
    html = out.text
    assert "tq-ok" in html and "tq-error" in html and "tq-check" in html                    # the three colours
    assert html.count("tq-step-hint") == 2 and "tq-help" in html and 'data-boxes=' in html
    assert '"status": "error"' in html or '"status":"error"' in html


# ----------------------------------------------------------------------------- what students ask again and again reaches the teacher
def test_the_teacher_sees_what_is_asked_again_and_again_without_names(room):
    teacher, student, rid = room
    for q in ("What does the mitochondria do?", "what does the mitochondria do", "Who won the 1998 world cup?"):
        say(student, rid, q)
    rows = anonymous.frequent(rid)
    assert len(rows) == 1 and rows[0]["count"] == 2 and rows[0]["students"] == 1 and set(rows[0]) == {"topic_id", "question", "count", "students", "unanswered", "deeper"}
    page = teacher.get(f"/app/rooms/{rid}/insights").text
    assert "tq-frequent" in page and "mitochondria" in page.lower() and "Omar" not in page
    api = teacher.get(f"/rooms/{rid}/instructor/frequent")
    assert api.status_code == 200 and "Omar" not in api.text and "u_" not in api.text
    assert_contract("FrequentQuestions", api.json())


def test_frequent_questions_group_by_topic_and_ignore_the_rare_ones(world):
    for q in ("what is ATP", "what is ATP?", "what is atp"):
        study_store.log_question("r", q, "question", True, "t_a", 1, "u_1")
    study_store.log_question("r", "what is ATP", "question", False, "t_b", 2, "u_2")      # same words, another topic: a different group
    study_store.log_question("r", "something asked once", "question", True, "t_a", 1, "u_3")
    rows = anonymous.frequent("r")
    assert [(r["topic_id"], r["count"]) for r in rows] == [("t_a", 3)] and anonymous.frequent("r", 1)[-1]["count"] == 1
    assert rows[0]["students"] == 1


def test_the_blue_site_is_gone(world):
    c = TestClient(app, follow_redirects=False)
    for path in ("/chat", "/snap/ui/read", "/quiz", "/review", "/plan", "/lab", "/study"):
        assert c.get(path).status_code in (404, 405), path
    assert c.get("/").status_code in (307, 303)


# ----------------------------------------------------------------------------- a photo to explain: the material first, then the web right away
def explain(student, rid, fid, question="Explain this photo", text="Mitochondria produce energy as ATP"):
    page = student.get(f"/app/rooms/{rid}/study")
    return student.post(f"/app/rooms/{rid}/study/message", data={"file_id": fid, "mode": "ask", "text": text, "question": question},
                        headers={**HX, "X-CSRF-Token": csrf_of(page.text)})


def snapped(student, rid, monkeypatch, lines):
    monkeypatch.setattr(snap_service, "get_reader", lambda name: FakeReader(lines))
    page = student.get(f"/app/rooms/{rid}/study")
    r = student.post(f"/app/rooms/{rid}/study/photo", files={"photo": ("p.png", _photo(("a", "b")), "image/png")}, headers={**HX, "X-CSRF-Token": csrf_of(page.text)})
    return re.search(r'name="file_id" value="([^"]+)"', r.text).group(1)


def test_a_photo_that_the_material_explains_is_answered_from_it_and_never_touches_the_web(room, monkeypatch):
    _, student, rid = room
    fid = snapped(student, rid, monkeypatch, ["Mitochondria produce energy as ATP", "in the cell"])
    monkeypatch.setattr(web_search, "multi_search", lambda *a, **k: pytest.fail("the material had the answer: no web search"))
    html = explain(student, rid, fid).text
    assert "biology_ch3.pdf" in html and "study-files/" not in html and "tq-quizbar" in html and "tq-quiz\"" not in html


def test_a_photo_the_material_cannot_explain_goes_to_the_web_without_a_button(room, monkeypatch):
    _, student, rid = room
    fid = snapped(student, rid, monkeypatch, ["The Treaty of Westphalia", "ended the war"])
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))                       # M2 answers: nothing matches
    searched = []
    pages = [{"title": "Westphalia", "url": "https://example.org/w", "content": "The Peace of Westphalia ended the Thirty Years War in 1648. " * 4, "score": 1}]
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: searched.append(q) or pages)
    html = explain(student, rid, fid, "Explain what is in the photo", "The Treaty of Westphalia ended the war").text
    assert searched and re.search(rf"/study-files/\d+", html) and "example.org/w" in html            # sent from the web straight away
    assert "Explain what is in the photo" in html and "tq-quiz\"" not in html
    assert '"action": "quiz"' in html                                                             # the study file's quiz is a button too
    logged = study_store.list_questions(rid)
    assert len(logged) == 1 and logged[0]["level"] == 2 and logged[0]["study_file_id"]            # one question, not two


def test_the_quiz_after_a_web_answer_comes_from_the_study_file(room, monkeypatch):
    _, student, rid = room
    asked = tutor_agent.act(rid, "explain", "The Treaty of Westphalia", None, "en")
    assert asked["result"]["answered_by"] in ("web", "m2", "none")
    fid = tutor_agent.act(rid, "research", "what is ATP", None, "en")["result"]["study_file"]["id"]
    out = tutor_agent.act(rid, "quiz", "", None, "en", study_file_id=fid)
    assert len(out["result"]["questions"]) == 3


def test_a_problem_in_the_photo_gets_hints_not_a_web_search(room, monkeypatch):
    _, student, rid = room
    fid = snapped(student, rid, monkeypatch, ["2x + 6 = 14", "x = ?"])
    monkeypatch.setattr(web_search, "multi_search", lambda *a, **k: pytest.fail("a problem is not searched on the web"))
    page = student.get(f"/app/rooms/{rid}/study")
    html = student.post(f"/app/rooms/{rid}/study/message", data={"file_id": fid, "mode": "ask", "text": "2x + 6 = 14", "question": "help"},
                        headers={**HX, "X-CSRF-Token": csrf_of(page.text)}).text
    assert "tq-help" in html


def test_questions_are_not_made_unless_asked_but_the_api_can_include_them(world):
    c = TestClient(app)
    plain = c.post("/rooms/r_bio101/ask", json={"question": "What does the mitochondria do?", "language": "en"}).json()
    assert plain["grounded"] and plain["quiz"] == []
    with_quiz = c.post("/rooms/r_bio101/ask", json={"question": "What does the mitochondria do?", "language": "en", "quiz": True}).json()
    assert_contract("RoomAskResponse", with_quiz) and len(with_quiz["quiz"]) == 3


def test_the_nearest_passages_that_are_about_something_else_do_not_answer_a_photo(room, monkeypatch):
    _, student, rid = room
    fid = snapped(student, rid, monkeypatch, ["The Peace of Westphalia 1648", "ended the Thirty Years War"])
    pages = [{"title": "Westphalia", "url": "https://example.org/w", "content": "The Peace of Westphalia ended the Thirty Years War in 1648. " * 4, "score": 1}]
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: pages)
    html = explain(student, rid, fid, "", "The Peace of Westphalia 1648 ended the Thirty Years War").text
    assert "biology_ch3.pdf" not in html and re.search(r"/study-files/\d+", html)                 # M2 returned its nearest page (about mitochondria): not used


# ----------------------------------------------------------------------------- the language of answers, quizzes and study files; "what is X" from the web
from rafeeq.tutor.answer import answer_language, keeps_english_terms  # noqa: E402
from rafeeq.tutor.lookup import term_of  # noqa: E402


@pytest.mark.parametrize("q,lang,terms", [
    ("ما وظيفة الميتوكوندريا بالانجليزي", "en", False), ("اشرح الخلية بالإنجليزية", "en", False), ("explain this in English please", "en", False),
    ("what does the mitochondria do in Arabic", "ar", True), ("What is ATP? answer in arabic", "ar", True), ("ما هو ATP بالعربي", "ar", False),
    ("ما وظيفة الميتوكوندريا؟", "ar", False), ("what is the cell", "en", False)])
def test_the_answer_language_is_what_was_asked_for(q, lang, terms):
    assert answer_language(q) == lang and keeps_english_terms(q, lang) is terms


def test_an_english_question_answered_in_arabic_keeps_the_terms_in_english(world, monkeypatch):
    prompts = []
    monkeypatch.setattr(llm_mod, "ask_json", lambda p, max_tokens=None: prompts.append(p) or {"grounded": True, "answer": "الـ mitochondria تنتج ATP", "steps": [], "used": [1]})
    out = tutor_agent.act("r_bio101", "ask", "What does the mitochondria do in Arabic?", None, "en")["result"]
    assert out["language"] == "ar" and any("Write in Arabic" in p and "technical term in English" in p for p in prompts)
    prompts.clear()
    out = tutor_agent.act("r_bio101", "ask", "ما وظيفة الميتوكوندريا بالانجليزي", None, "ar")["result"]
    assert out["language"] == "en" and any("Write in English" in p for p in prompts) and not any("technical term" in p for p in prompts)


def test_the_site_language_does_not_change_the_language_of_an_answer(world):
    ar = tutor_agent.act("r_bio101", "ask", "ما وظيفة الميتوكوندريا؟", None, "en")["result"]
    en = tutor_agent.act("r_bio101", "ask", "What does the mitochondria do?", None, "ar")["result"]
    assert (ar["language"], en["language"]) == ("ar", "en")


def test_the_quiz_of_an_answer_follows_the_answer_and_the_quiz_of_a_file_follows_the_site(world, monkeypatch):
    prompts = []
    real = model
    monkeypatch.setattr(llm_mod, "ask_json", lambda p, max_tokens=None: prompts.append(p) or real(p, max_tokens))
    tutor_agent.act("r_bio101", "quiz", "", None, "en", topic_id="t_energy", question="What does the mitochondria do?")       # an English answer
    assert any("write everything in English" in p for p in prompts)
    prompts.clear()
    fid = tutor_agent.act("r_bio101", "research", "ما هي الخلية؟", None, "en", pages=1)["result"]["study_file"]["id"]
    assert any("Write in English" in p for p in prompts)                                      # the file: the site language, not the question's
    prompts.clear()
    tutor_agent.act("r_bio101", "quiz", "", None, "ar", study_file_id=fid)
    assert any("write everything in Arabic" in p for p in prompts)


def test_what_is_a_term_is_answered_from_the_web_in_the_chat_without_a_file(world, monkeypatch):
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    calls = []
    pages = [{"title": "Photosynthesis - Britannica", "url": "https://example.org/p", "content": "Photosynthesis is the process plants use to turn light into chemical energy. " * 3, "score": 1}]
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: calls.append(q) or pages)
    out = tutor_agent.act("r_bio101", "auto", "what is photosynthesis?", None, "en")
    r = out["result"]
    assert out["ok"] and r["grounded"] and r["material_source"] == "web" and r["web_sources"][0]["url"] == "https://example.org/p" and r["study_file"] is None
    assert calls == ["photosynthesis"] and r["offer_level2"] is True                        # a short answer now, the deeper study file is still one button away
    assert study_store.list_study_files("r_bio101") == []                                    # no study file
    [row] = study_store.list_questions("r_bio101")
    assert row["grounded"] == 1 and row["engine"] == "web"
    assert_contract("RoomAskResponse", r)


def test_any_question_the_material_cannot_answer_gets_a_short_web_answer_and_no_file(world, monkeypatch):
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    queries = []
    pages = [{"title": "WWI causes", "url": "https://example.org/ww1", "content": "The First World War began after the assassination of Archduke Franz Ferdinand in 1914. " * 3, "score": 1}]
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: queries.append(q) or pages)
    r = tutor_agent.act("r_bio101", "ask", "explain how the causes of the first world war led to the treaty", None, "en")["result"]
    assert r["grounded"] and r["material_source"] == "web" and r["web_sources"] and r["study_file"] is None and r["offer_level2"] is True
    assert queries == ["explain how the causes of the first world war led to the treaty"]
    assert study_store.list_study_files("r_bio101") == []


def test_when_the_web_has_nothing_either_it_says_so_and_offers_level_2(world, monkeypatch):
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    monkeypatch.setattr(web_search, "_tavily_http", lambda *a: [])
    r = tutor_agent.act("r_bio101", "ask", "some question nobody wrote about", None, "en")["result"]
    assert r["grounded"] is False and r["offer_level2"] is True and r["web_sources"] == []


@pytest.mark.parametrize("text,photo,term", [("ما هو الـ osmosis؟", False, "الـ osmosis"), ("يعني ايه entropy", False, "entropy"), ("define entropy", False, "entropy"),
                                             ("Westphalia", True, "Westphalia"), ("Westphalia", False, None), ("2x + 6 = 14", True, None),
                                             ("ما وظيفة الميتوكوندريا؟", False, None), ("a b c d e f g", True, None)])
def test_which_texts_are_a_term_lookup(text, photo, term):
    assert term_of(text, photo) == term


def test_a_photographed_word_is_looked_up_on_the_web_and_answered_without_a_file(room, monkeypatch):
    _, student, rid = room
    fid = snapped(student, rid, monkeypatch, ["Osmosis"])
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    pages = [{"title": "Osmosis", "url": "https://example.org/o", "content": "Osmosis is the movement of water across a semipermeable membrane. " * 3, "score": 1}]
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: pages)
    html = explain(student, rid, fid, "", "Osmosis").text
    assert "example.org/o" in html and "/study-files/" not in html and '"action": "quiz"' not in html
    assert study_store.list_study_files(rid) == []


# ----------------------------------------------------------------------------- deleting a study file
def test_a_student_can_delete_a_study_file_from_the_list_and_from_its_page(room):
    _, student, rid = room
    html = agent(student, rid, action="research", text="what is ATP", pages="1").text
    fid = re.search(r"/study-files/(\d+)", html).group(1)
    for ex in ("summary", "cards"):
        agent(student, rid, action=ex, study_file_id=fid)
    listing = student.get(f"/app/rooms/{rid}/study-files").text
    assert f"/study-files/{fid}/delete" in listing and "tq-delete" in listing
    assert f"/study-files/{fid}/delete" in student.get(f"/app/rooms/{rid}/study-files/{fid}").text
    path = study_store.get_study_file(int(fid))["path"]
    token = csrf_of(listing)
    assert student.post(f"/app/rooms/{rid}/study-files/{fid}/delete", data={}).status_code == 403              # no token, no delete
    r = student.post(f"/app/rooms/{rid}/study-files/{fid}/delete", data={"csrf_token": token})
    assert r.status_code == 200 and r.url.path.endswith("/study-files")
    assert study_store.get_study_file(int(fid)) is None and not __import__("pathlib").Path(path).exists()
    assert not list((study_store.data_dir() / "summaries").glob(f"{fid}-*.json")) and not list((study_store.data_dir() / "flashcards").glob(f"{fid}-*.json"))
    assert student.get(f"/app/rooms/{rid}/study-files/{fid}").status_code == 404
    assert student.get(f"/app/rooms/{rid}/study-files/{fid}/download", params={"format": "md"}).status_code == 404
    assert student.post(f"/app/rooms/{rid}/study-files/{fid}/delete", data={"csrf_token": token}).status_code == 404       # already gone
    assert all(q["study_file_id"] is None for q in study_store.list_questions(rid))                          # the question stays in the log


def test_a_file_of_another_room_cannot_be_deleted(room, world):
    teacher, student, rid = room
    other = tutor_agent.act("r_other", "research", "what is ATP", None, "en", pages=1)["result"]["study_file"]["id"]
    token = csrf_of(student.get(f"/app/rooms/{rid}/study-files").text)
    assert student.post(f"/app/rooms/{rid}/study-files/{other}/delete", data={"csrf_token": token}).status_code == 404
    assert study_store.get_study_file(other) is not None


# ----------------------------------------------------------------------------- study files are the student's own
def test_a_study_file_is_visible_only_to_the_student_who_asked_for_it(room, world):
    teacher, omar, rid = room
    from test_web import student_in_room  # noqa: F811
    code = re.search(r"join-code-value\" dir=\"ltr\">([A-Z0-9-]+)<", teacher.get(f"/app/rooms/{rid}/overview").text).group(1)
    sara = TestClient(app)
    from test_web import register
    register(sara, "student", "Sara", "sara@x.com")
    sara.post("/app/rooms/join", data={"csrf_token": csrf_of(sara.get("/app/rooms").text), "code": code})

    html = agent(omar, rid, action="research", text="what is ATP", pages="1").text        # Omar searches something outside the course
    fid = re.search(r"/study-files/(\d+)", html).group(1)
    assert f"/study-files/{fid}" in omar.get(f"/app/rooms/{rid}/study-files").text
    assert f"/study-files/{fid}" not in sara.get(f"/app/rooms/{rid}/study-files").text      # Sara's list does not have it
    assert sara.get(f"/app/rooms/{rid}/study-files/{fid}").status_code == 404
    assert sara.get(f"/app/rooms/{rid}/study-files/{fid}/download", params={"format": "md"}).status_code == 404
    token = csrf_of(sara.get(f"/app/rooms/{rid}/study-files").text)
    assert sara.post(f"/app/rooms/{rid}/study-files/{fid}/delete", data={"csrf_token": token}).status_code == 404
    for tool in ("quiz", "summary", "cards"):                                             # and cannot use it through the agent either
        assert "tq-q" not in agent(sara, rid, action=tool, study_file_id=fid).text
    assert study_store.get_study_file(int(fid)) is not None                                # Omar's file is untouched
    assert omar.get(f"/app/rooms/{rid}/study-files/{fid}").status_code == 200


def test_the_api_lists_only_the_callers_files(world):
    c = TestClient(app)
    a = {"X-User-Id": "u_a"}
    b = {"X-User-Id": "u_b"}
    fid = c.post("/rooms/r_bio101/tutor/act", json={"action": "research", "text": "what is ATP", "pages": 1}, headers=a).json()["result"]["study_file"]["id"]
    assert [f["id"] for f in c.post("/rooms/r_bio101/tutor/act", json={"action": "files"}, headers=a).json()["result"]["files"]] == [fid]
    assert c.post("/rooms/r_bio101/tutor/act", json={"action": "files"}, headers=b).json()["result"]["files"] == []
    assert c.post("/rooms/r_bio101/tutor/act", json={"action": "file", "study_file_id": fid}, headers=b).json()["ok"] is False


def test_an_old_database_gets_the_owner_column(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path))
    con = sqlite3.connect(tmp_path / "study.db")
    con.execute("CREATE TABLE study_files(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, room_id TEXT, question TEXT, title TEXT, markdown TEXT, sources_json TEXT, path TEXT)")
    con.commit(); con.close()
    study_store.save_study_file("r", "q", "t", "# t", [], "u_1")
    assert [f["user_id"] for f in study_store.list_study_files("r", "u_1")] == ["u_1"] and study_store.list_study_files("r", "u_2") == []

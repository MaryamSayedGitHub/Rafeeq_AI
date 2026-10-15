"""Questions only the teacher can answer go to the teacher (never to the web); the web answer can also be sent to the teacher with one button;
and the teacher's reply comes back to the student. No model, no network."""
import pytest

from rafeeq.copilot import clients
from rafeeq.copilot.supervisor import clear_history, handle_message
from rafeeq.tutor import connect, llm, lookup, m2, study_store, tutor_agent, web_search
from test_inbox import Store  # noqa: F401  (the in-memory store of the inbox tests)
from rafeeq.teacher import inbox

PAGES = [{"title": "WWI", "url": "https://example.org/ww1", "content": "The First World War began in 1914. " * 6, "score": 1}]


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path / "study"))
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    monkeypatch.setattr(m2, "search", lambda *a, **k: ([], "m2"))
    searched = []
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: searched.append(q) or PAGES)
    monkeypatch.setattr(llm, "ask_json", lambda prompt, max_tokens=None: {"teacher_only": False} if "teacher_only" in prompt else
                        ({"answer": "It began in 1914. [1]"} if "web sources" in prompt else None))
    st = Store()
    clear_history()
    clients.connect("escalate", lambda room_id, q, user_id: inbox.escalate(room_id, q, user_id, store=st))
    clients.connect("find_verified", lambda room_id, q: inbox.find_verified(room_id, q, store=st))
    connect.connect_m3(force=True)
    yield st, searched
    connect.disconnect_m3()
    clients.disconnect("escalate")
    clients.disconnect("find_verified")


@pytest.mark.parametrize("q", ["when is the final exam?", "متى ميعاد الامتحان؟", "what did the doctor mean in lecture 3", "my grades are low, what now?"])
def test_questions_about_the_course_itself_are_recognised(q):
    assert lookup._COURSE.search(q)


@pytest.mark.parametrize("q", ["what caused the first world war?", "what is a final state machine?", "ما أسباب الحرب العالمية الأولى؟"])
def test_subject_questions_are_not_taken_for_course_questions(q):
    assert not lookup._COURSE.search(q)


def test_a_course_question_goes_to_the_teacher_and_the_web_is_not_searched(world):
    st, searched = world
    out = handle_message("r1", "when is the final exam?", user_id="u_omar")
    assert out["payload"]["grounded"] is False and out["payload"]["escalated"] is True and out["payload"]["question_id"]
    assert searched == []                                                                           # nothing was sent to Tavily
    assert [q["text"] for q in inbox.list_questions("r1", store=st)] == ["when is the final exam?"]


def test_a_subject_question_the_material_misses_is_answered_from_the_web(world):
    st, searched = world
    out = handle_message("r1", "what caused the first world war?", user_id="u_omar")
    assert out["payload"]["grounded"] is True and out["payload"]["material_source"] == "web" and searched
    assert inbox.list_questions("r1", store=st) == []


def test_when_the_web_has_nothing_either_the_question_goes_to_the_teacher(world, monkeypatch):
    st, _ = world
    monkeypatch.setattr(web_search, "_tavily_http", lambda q, n, d: [])
    out = handle_message("r1", "what caused the first world war?", user_id="u_omar")
    assert out["payload"]["grounded"] is False and out["payload"]["escalated"] is True


def test_the_ask_my_teacher_button_files_the_question(world):
    st, _ = world
    r = tutor_agent.act("r1", "teacher", "what caused the first world war?", "u_omar", "en")
    assert r["ok"] and r["result"]["escalated"] and r["result"]["question_id"]
    assert [q["user_id"] for q in inbox.list_questions("r1", store=st)] == ["u_omar"]


def test_the_teachers_reply_reaches_the_student_and_is_reused(world):
    st, _ = world
    first = handle_message("r1", "when is the final exam?", user_id="u_omar")
    inbox.answer("r1", first["payload"]["question_id"], "Sunday 10am, hall 3.", store=st)
    mine = inbox.list_questions("r1", user_id="u_omar", store=st)                                    # what the student's page lists
    assert mine[0]["status"] == "answered" and mine[0]["answer"] == "Sunday 10am, hall 3."
    again = handle_message("r1", "when is the final exam", user_id="u_sara")                         # another student asks the same
    assert again["reply"] == "Sunday 10am, hall 3." and again["payload"]["verified_by_teacher"] is True

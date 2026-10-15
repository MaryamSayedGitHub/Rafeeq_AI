"""M5 inbox: escalated questions, the teacher's answers, and reusing them (no LLM, no network)."""
from rafeeq.copilot import clients
from rafeeq.copilot.supervisor import clear_history, handle_message
from rafeeq.teacher import inbox


class Store:
    class Item:
        def __init__(self, key, value):
            self.key, self.value = key, value

    def __init__(self):
        self.data = {}

    def put(self, ns, key, value):
        self.data.setdefault(tuple(ns), {})[key] = dict(value)

    def search(self, ns, limit=10, offset=0, **kw):
        return [self.Item(k, dict(v)) for k, v in self.data.get(tuple(ns), {}).items()][offset:offset + limit]


def test_escalate_files_once_and_the_answer_is_found_for_a_similar_question(monkeypatch):
    monkeypatch.setattr(inbox, "_student_names", lambda room_id: {"u_omar": "عمر"})
    st = Store()
    first = inbox.escalate("r1", "What is the gain of a TWT amplifier?", "u_omar", store=st)
    again = inbox.escalate("r1", "what is the gain of a TWT amplifier", "u_omar", store=st)
    assert first == again
    listed = inbox.list_questions("r1", store=st)
    assert [q["status"] for q in listed] == ["open"] and listed[0]["student_name"] == "عمر"
    assert inbox.find_verified("r1", "gain of a TWT amplifier?", store=st) is None  # not answered yet
    inbox.answer("r1", first["question_id"], "Usually 30 to 60 dB.", store=st)
    found = inbox.find_verified("r1", "What is the gain of TWT amplifier?", store=st)
    assert found["answer"] == "Usually 30 to 60 dB."
    assert inbox.find_verified("r1", "What is a magnetron?", store=st) is None
    assert inbox.list_questions("r1", user_id="u_other", store=st) == []


def test_the_copilot_escalates_uncovered_questions_and_reuses_the_teachers_answer(monkeypatch):
    st = Store()
    clear_history()
    clients.connect("tutor_answer", lambda room_id, q: {"answer": "This topic is not covered in the provided course material.",
                                                        "citations": [], "grounded": False, "steps": []})
    clients.connect("escalate", lambda room_id, q, user_id: inbox.escalate(room_id, q, user_id, store=st))
    clients.connect("find_verified", lambda room_id, q: inbox.find_verified(room_id, q, store=st))
    try:
        out = handle_message("r1", "What is the gain of a TWT amplifier?", user_id="u_omar")
        assert out["payload"]["escalated"] is True and out["payload"]["question_id"]
        inbox.answer("r1", out["payload"]["question_id"], "Usually 30 to 60 dB.", store=st)
        again = handle_message("r1", "what's the gain of a TWT amplifier", user_id="u_sara")
        assert again["reply"] == "Usually 30 to 60 dB." and again["payload"]["verified_by_teacher"] is True
    finally:
        for name in ("tutor_answer", "escalate", "find_verified"):
            clients.disconnect(name)


def test_drafts_use_the_model_and_say_unavailable_without_one():
    assert inbox.draft("r1", "q", propose=lambda s, u: {"reply": "Check page 4."}) == "Check page 4."
    assert inbox.draft("r1", "q", propose=lambda s, u: {}) is None
    text = inbox.assistant("r1", "رسالة للفصل", store=Store(), propose=lambda s, u: {"reply": u})
    assert "(no quizzes yet)" in text   # the model only gets real class data, here: none

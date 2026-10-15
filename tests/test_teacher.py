"""M5 teacher: profile, past sessions, the session view shape, and the pages.

No LLM key, no database and no network are needed:
    uv run pytest tests/test_teacher.py
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import Draft202012Validator

from rafeeq.samples import load_sample
from rafeeq.teacher.lecturer.long_term_store import list_sessions, save_session
from rafeeq.teacher.lecturer.schemas import ApproveRequest, SessionRequest
from rafeeq.teacher.profile import TeacherProfileUpdate, load_profile, make_teacher_id, save_profile

ROOT = Path(__file__).resolve().parent.parent
SPEC = yaml.safe_load((ROOT / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))


def assert_matches(schema_name, data):
    schema = {"$ref": f"#/components/schemas/{schema_name}", "components": SPEC["components"]}
    errors = list(Draft202012Validator(schema).iter_errors(json.loads(json.dumps(data))))
    assert not errors, f"{schema_name}: " + "; ".join(f"{list(e.path)}: {e.message}" for e in errors)


class FakeStore:
    """Just enough of LangGraph's Store (get / put / search / list_namespaces)."""

    def __init__(self):
        self.data = {}

    def put(self, namespace, key, value):
        self.data[(tuple(namespace), key)] = value

    def get(self, namespace, key):
        value = self.data.get((tuple(namespace), key))
        return SimpleNamespace(key=key, value=value) if value is not None else None

    def search(self, namespace, limit=10):
        found = [SimpleNamespace(key=k, value=v) for (ns, k), v in self.data.items() if ns == tuple(namespace)]
        return found[:limit]

    def list_namespaces(self, prefix=(), max_depth=None):
        names = {ns[:max_depth] if max_depth else ns for ns, _ in self.data if ns[: len(prefix)] == tuple(prefix)}
        return sorted(names)


SARA = {"name": "سارة أحمد", "title": "د.", "subjects": "Python، تعلم الآلة", "bio": "أدرّس البرمجة."}


# ------------------------------------------------------------------- profile
def test_teacher_id_is_made_from_the_name():
    assert make_teacher_id("Dr. Sara Ahmed") == "dr-sara-ahmed"
    assert make_teacher_id("سارة أحمد") == "سارة-أحمد"
    assert make_teacher_id("  ...  ") == "teacher"


def test_profile_round_trip_matches_the_contract():
    store = FakeStore()
    teacher_id = save_profile(store, TeacherProfileUpdate(**SARA))
    profile = load_profile(store, teacher_id, ["Usually teaches in Arabic."])

    assert profile.display_name == "د. سارة أحمد"
    assert profile.subjects == ["Python", "تعلم الآلة"]
    assert profile.remembered == ["Usually teaches in Arabic."]
    assert_matches("TeacherProfile", profile.model_dump())


def test_a_new_teacher_never_takes_over_an_existing_profile():
    store = FakeStore()
    first = save_profile(store, TeacherProfileUpdate(**SARA))
    second = save_profile(store, TeacherProfileUpdate(**{**SARA, "bio": "another teacher"}))
    assert first != second
    assert load_profile(store, first).bio == "أدرّس البرمجة."


def test_editing_keeps_the_same_teacher_id():
    store = FakeStore()
    teacher_id = save_profile(store, TeacherProfileUpdate(**SARA))
    again = save_profile(store, TeacherProfileUpdate(**{**SARA, "teacher_id": teacher_id, "name": "سارة محمد"}))
    assert again == teacher_id
    assert load_profile(store, teacher_id).display_name == "د. سارة محمد"


def test_an_approved_session_updates_defaults_and_history():
    store = FakeStore()
    teacher_id = save_profile(store, TeacherProfileUpdate(**SARA))
    brief = {"session_id": "s1", "lecturer_id": teacher_id, "topic": "Python functions", "room_id": "r_bio101",
             "language": "English", "programming_language": "Python", "student_level": "advanced"}
    assert save_session(store, brief, ["Defining", "Calling"], [], ["functions.pptx"])

    profile = load_profile(store, teacher_id)
    assert profile.defaults.language == "English"           # learned from the approved session
    assert profile.defaults.student_level == "advanced"
    assert profile.sessions_count == 1

    [session] = list_sessions(store, teacher_id)
    assert session["files"] == ["functions.pptx"] and session["room_id"] == "r_bio101"


def test_unknown_teacher_has_no_profile():
    assert load_profile(FakeStore(), "nobody") is None


# ------------------------------------------------------------ request models
def test_contract_request_samples_are_accepted_by_the_code():
    SessionRequest(**load_sample("lecturer_session_request"))
    ApproveRequest(**load_sample("lecturer_decision"))


def test_revise_needs_feedback():
    with pytest.raises(ValueError):
        ApproveRequest(decision="revise", feedback="   ")


# ------------------------------------------------------- session view (graph)
def test_session_view_matches_the_contract():
    pytest.importorskip("langgraph")
    from rafeeq.teacher.lecturer.controller import GraphController

    sample = load_sample("lecturer_session")
    values = {
        "brief": {"room_id": "r_bio101", "topic": "Python functions"},
        "selected_agents": sample["selected_agents"], "reasoning": sample["reasoning"],
        "memory_context": sample["memory"], "draft": sample["draft"], "revision_count": 0,
    }
    waiting = SimpleNamespace(values=values, next=("human_approval",))
    view = GraphController(graph=None)._view("abc", waiting)
    assert view["status"] == "awaiting_approval"
    assert_matches("LecturerSession", view)

    done = SimpleNamespace(values={**values, "final_output": {"folder": "abc", "files": ["a.pptx"]}}, next=())
    view = GraphController(graph=None)._view("abc", done)
    assert view["status"] == "done"
    assert view["files"] == [{"name": "a.pptx", "url": "/teacher/sessions/abc/files/a.pptx"}]
    assert_matches("LecturerSession", view)


# -------------------------------------------------------------------- pages
@pytest.fixture
def client(monkeypatch, tmp_path):
    pytest.importorskip("langgraph")
    from fastapi.testclient import TestClient

    monkeypatch.setattr("rafeeq.teacher.lecturer.settings.DATABASE_URL", None)  # in-memory store
    from rafeeq.main import app

    with TestClient(app) as c:
        yield c


def test_teacher_page_asks_for_a_profile_first(client):
    page = client.get("/teacher")
    assert page.status_code == 200
    assert 'action="/teacher/ui/profile"' in page.text


def test_saving_the_profile_opens_the_dashboard(client):
    form = {"name": "سارة أحمد", "title": "د.", "subjects": "Python", "bio": "",
            "language": "Arabic", "student_level": "beginner", "programming_language": "Python",
            "duration_minutes": "60"}
    saved = client.post("/teacher/ui/profile", data=form, follow_redirects=False)
    assert saved.status_code == 303 and "rafeeq_teacher" in saved.cookies

    page = client.get("/teacher")
    assert "د. سارة أحمد" in page.text
    assert 'hx-post="/teacher/ui/sessions"' in page.text

    from urllib.parse import unquote

    teacher_id = unquote(saved.cookies["rafeeq_teacher"])
    assert teacher_id == "سارة-أحمد"  # Arabic ids survive the ASCII-only cookie
    api = client.get("/teacher/profile", params={"teacher_id": teacher_id})
    assert api.status_code == 200
    assert_matches("TeacherProfile", api.json())


def test_class_endpoints_return_contract_shapes(client):
    assert_matches("Insights", client.get("/teacher/rooms/r_bio101/insights").json())
    assert_matches("ClassReport", client.get("/teacher/rooms/r_bio101/report").json())

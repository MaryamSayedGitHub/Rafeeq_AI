"""Checks that every sample in contracts/samples/ matches its schema in contracts/openapi.yaml.

Run:  uv run pytest tests/test_contracts.py

If this test fails, read the message: it names the sample file and the field.
Fix the sample or the schema, in the same pull request.
"""
import copy
import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
SPEC = yaml.safe_load((ROOT / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))
SAMPLES_DIR = ROOT / "contracts" / "samples"

# sample file name (without .json) -> schema name in openapi.yaml
SAMPLE_TO_SCHEMA = {
    "login": "Login",
    "my_rooms": "MyRooms",
    "upload_file": "UploadFile",
    "register": "Login",
    "join_room": "Room",
    "create_room": "CreateRoomResult",
    "room_files": "RoomFiles",
    "room_students": "RoomStudents",
    "join_code": "JoinCode",
    "room_ask": "RoomAskResponse",
    "room_ask_level2": "RoomAskResponse",
    "vision_check": "VisionCheck",
    "logged_questions": "LoggedQuestions",
    "instructor_questions": "InstructorQuestions",
    "tutor_act": "TutorActResponse",
    "frequent_questions": "FrequentQuestions",
    "snap_result": "SnapResult",
    "copilot_reply": "CopilotReply",
    "search_material": "SearchMaterial",
    "room_topics": "RoomTopics",
    "source_page": "SourcePage",
    "tutor_answer": "TutorAnswer",
    "handwriting_check": "HandwritingCheck",
    "make_quiz": "Quiz",
    "submit_quiz": "QuizResult",
    "next_step": "NextStep",
    "student_summary": "StudentSummary",
    "question_event": "QuestionEvent",
    "insights": "Insights",
    "class_report": "ClassReport",
    "teacher_assistant": "TeacherAssistant",
    "teacher_profile": "TeacherProfile",
    "teacher_sessions": "TeacherSessions",
    "lecturer_session_request": "LecturerSessionRequest",
    "lecturer_decision": "LecturerDecision",
    "lecturer_session": "LecturerSession",
    "video_search": "VideoSearch",
    "video_job": "VideoJob",
    "video_job_status": "VideoJobStatus",
    "avatar_saved": "AvatarSaved",
    "asset": "Asset",
    "assets": "AssetList",
    "asset_create": "AssetCreate",
    "deck": "Deck",
    "decks": "DeckList",
    "themes": "ThemeList",
    "course": "Course",
    "courses": "CourseList",
    "room_courses": "RoomCourses",
    "media_capabilities": "MediaCapabilities",
    "visual_job": "VisualJob",
    "visual_status": "VisualStatus",
    "story": "StoryStatus",
}


def validator_for(schema_name):
    schema = {"$ref": f"#/components/schemas/{schema_name}", "components": SPEC["components"]}
    return Draft202012Validator(schema, format_checker=FormatChecker())


def load(name):
    return json.loads((SAMPLES_DIR / f"{name}.json").read_text(encoding="utf-8"))


def test_spec_is_openapi_31():
    assert SPEC["openapi"].startswith("3.1")


def test_every_sample_file_is_registered_and_every_registration_has_a_file():
    files = {p.stem for p in SAMPLES_DIR.glob("*.json")}
    assert files == set(SAMPLE_TO_SCHEMA), (
        f"Not registered: {sorted(files - set(SAMPLE_TO_SCHEMA))}. "
        f"Missing file: {sorted(set(SAMPLE_TO_SCHEMA) - files)}."
    )


@pytest.mark.parametrize("name", sorted(SAMPLE_TO_SCHEMA))
def test_sample_matches_schema(name):
    errors = sorted(validator_for(SAMPLE_TO_SCHEMA[name]).iter_errors(load(name)), key=lambda e: list(e.path))
    assert not errors, f"samples/{name}.json: " + "; ".join(
        f"{'/'.join(map(str, e.path)) or '(top level)'}: {e.message}" for e in errors
    )


def test_a_typo_is_caught():
    """Proof that the check works: a wrong field name must fail."""
    bad = copy.deepcopy(load("search_material"))
    bad["chunks"][0]["pages"] = bad["chunks"][0].pop("page")  # typo: pages instead of page
    assert list(validator_for("SearchMaterial").iter_errors(bad))


def test_quiz_answers_are_among_the_choices():
    for q in load("make_quiz")["questions"]:
        assert q["correct_answer"] in q["choices"]

"""M1 platform logic: the real student journey, without HTTP."""
import pytest
import yaml
from jsonschema import Draft202012Validator
from pathlib import Path

from rafeeq.platform import service
from rafeeq.platform.service import PlatformStore

SPEC = yaml.safe_load((Path(__file__).resolve().parent.parent / "contracts" / "openapi.yaml").read_text(encoding="utf-8"))


def assert_schema(name, data):
    schema = {"$ref": f"#/components/schemas/{name}", "components": SPEC["components"]}
    errors = list(Draft202012Validator(schema).iter_errors(data))
    assert not errors, [e.message for e in errors]


@pytest.fixture
def store(tmp_path):
    return PlatformStore(tmp_path, seed_demo=False)


def teacher_and_room(store):
    t = store.register("Dr. Mona", "mona@x.com", "secret1", "teacher")
    user = store.user_for_token(t["token"])
    created = store.create_room(user, "Biology 101")
    return user, created


def test_register_and_login_match_the_login_schema(store):
    reg = store.register("Sara", "Sara@X.com ", "secret1", "student")
    assert_schema("Login", reg)
    log = store.login("sara@x.com", "secret1")
    assert_schema("Login", log)
    assert log["user_id"] == reg["user_id"]


def test_wrong_password_and_unknown_email_fail(store):
    store.register("Sara", "sara@x.com", "secret1", "student")
    with pytest.raises(service.AuthError):
        store.login("sara@x.com", "nope")
    with pytest.raises(service.AuthError):
        store.login("ghost@x.com", "secret1")


def test_duplicate_email_and_bad_input(store):
    store.register("Sara", "sara@x.com", "secret1", "student")
    with pytest.raises(service.Conflict):
        store.register("Other", "SARA@x.com", "secret1", "student")
    with pytest.raises(service.Invalid):
        store.register("", "a@b.com", "secret1", "student")
    with pytest.raises(service.Invalid):
        store.register("A", "not-an-email", "secret1", "student")
    with pytest.raises(service.Invalid):
        store.register("A", "a@b.com", "123", "student")


def test_new_student_has_no_rooms_until_joining(store):
    _, created = teacher_and_room(store)
    assert_schema("CreateRoomResult", created)
    s = store.user_for_token(store.register("Sara", "sara@x.com", "secret1", "student")["token"])
    assert store.list_rooms(s) == {"rooms": []}

    code = created["join_code"].lower().replace("-", " ")  # typed loosely
    room = store.join_room(s, code)
    assert_schema("Room", room)
    rooms = store.list_rooms(s)
    assert_schema("MyRooms", rooms)
    assert [r["room_id"] for r in rooms["rooms"]] == [created["room"]["room_id"]]


def test_register_with_join_code_joins_the_room(store):
    _, created = teacher_and_room(store)
    reg = store.register("Omar", "omar@x.com", "secret1", "student", created["join_code"])
    s = store.user_for_token(reg["token"])
    assert len(store.list_rooms(s)["rooms"]) == 1
    with pytest.raises(service.Invalid):
        store.register("Ali", "ali@x.com", "secret1", "student", "ZZZ-ZZZ")


def test_roles_are_enforced(store):
    teacher, created = teacher_and_room(store)
    s = store.user_for_token(store.register("Sara", "sara@x.com", "secret1", "student")["token"])
    with pytest.raises(service.Forbidden):
        store.create_room(s, "My room")
    with pytest.raises(service.Forbidden):
        store.join_room(teacher, created["join_code"])
    with pytest.raises(service.NotFound):
        store.join_room(s, "AAA-AAA")
    rid = created["room"]["room_id"]
    with pytest.raises(service.Forbidden):
        store.list_files(s, rid)  # not a member yet
    with pytest.raises(service.Forbidden):
        store.join_code(s, rid)
    assert_schema("JoinCode", store.join_code(teacher, rid))


def test_upload_lists_file_as_uploaded_not_ready(store):
    teacher, created = teacher_and_room(store)
    rid = created["room"]["room_id"]
    up = store.add_file(teacher, rid, "../../lecture 1.pdf", b"%PDF-1.4")
    assert_schema("UploadFile", up)
    files = store.list_files(teacher, rid)
    assert_schema("RoomFiles", files)
    assert files["files"][0]["name"] == "lecture 1.pdf"  # path parts stripped
    assert files["files"][0]["status"] == "uploaded"
    assert store.list_rooms(teacher)["rooms"][0]["files_ready"] == 0
    store.set_file_status(rid, up["file_id"], "ready")
    assert store.list_rooms(teacher)["rooms"][0]["files_ready"] == 1
    assert store.file_path(rid, up["file_id"]).read_bytes() == b"%PDF-1.4"


def test_students_list_counts_questions_and_never_invents_mastery(store):
    teacher, created = teacher_and_room(store)
    rid = created["room"]["room_id"]
    s = store.user_for_token(store.register("Sara", "sara@x.com", "secret1", "student", created["join_code"])["token"])
    store.record_question(s["user_id"], rid)
    store.record_question(s["user_id"], rid)
    students = store.list_students(teacher, rid)
    assert_schema("RoomStudents", students)
    assert students["students"][0]["questions_this_week"] == 2
    assert students["students"][0]["mastery"] is None
    assert students["students"][0]["status"] is None


def test_data_survives_a_restart(tmp_path):
    a = PlatformStore(tmp_path, seed_demo=False)
    reg = a.register("Sara", "sara@x.com", "secret1", "student")
    b = PlatformStore(tmp_path, seed_demo=False)
    assert b.user_for_token(reg["token"])["name"] == "Sara"


def test_demo_accounts_keep_the_old_token(tmp_path):
    store = PlatformStore(tmp_path, seed_demo=True)
    assert store.user_for_token("demo-token-sara-123")["role"] == "student"
    assert store.login("mona@rafeeq.demo", service.DEMO_PASSWORD)["role"] == "teacher"


def test_token_header_parsing():
    assert service.token_from_header("Bearer abc") == "abc"
    assert service.token_from_header("bearer  abc ") == "abc"
    assert service.token_from_header("Basic abc") is None
    assert service.token_from_header(None) is None


def test_upload_hook_lets_m2_index_and_report_status(store):
    teacher, created = teacher_and_room(store)
    rid = created["room"]["room_id"]
    seen = []

    def fake_indexer(room_id, file_id, path):
        seen.append((room_id, path.name))
        store.set_file_status(room_id, file_id, "indexing")

    service.on_upload(fake_indexer)
    try:
        up = store.add_file(teacher, rid, "a.pdf", b"x")
        assert up["status"] == "indexing" and seen[0][0] == rid
        service.clear_upload_hooks()
        service.on_upload(lambda *a: 1 / 0)
        bad = store.add_file(teacher, rid, "b.pdf", b"y")
        assert bad["status"] == "failed"
    finally:
        service.clear_upload_hooks()


def test_teacher_deletes_a_file_record_disk_copy_and_tells_the_hooks(store):
    teacher, created = teacher_and_room(store)
    rid = created["room"]["room_id"]
    up = store.add_file(teacher, rid, "slides.pptx", b"pptx bytes")
    path = store.file_path(rid, up["file_id"])
    calls = []
    service.on_delete(lambda room_id, file_id: calls.append((room_id, file_id)))
    try:
        store.delete_file(teacher, rid, up["file_id"])
    finally:
        service.clear_delete_hooks()
    assert store.list_files(teacher, rid) == {"files": []}
    assert not path.exists()
    assert calls == [(rid, up["file_id"])]
    with pytest.raises(service.NotFound):
        store.delete_file(teacher, rid, up["file_id"])  # already gone


def test_only_the_room_teacher_can_delete_and_a_failing_hook_does_not_undo_it(store):
    teacher, created = teacher_and_room(store)
    rid, code = created["room"]["room_id"], created["join_code"]
    up = store.add_file(teacher, rid, "notes.pdf", b"%PDF-1.4")
    s = store.user_for_token(store.register("Sara", "sara@x.com", "secret1", "student", code)["token"])
    with pytest.raises((service.Forbidden, service.NotFound)):
        store.delete_file(s, rid, up["file_id"])
    assert len(store.list_files(teacher, rid)["files"]) == 1

    def broken(room_id, file_id):
        raise RuntimeError("index offline")

    service.on_delete(broken)
    try:
        store.delete_file(teacher, rid, up["file_id"])
    finally:
        service.clear_delete_hooks()
    assert store.list_files(teacher, rid) == {"files": []}


def test_a_login_token_expires_after_a_month(store):
    reg = store.register("Sara", "sara2@x.com", "secret1", "student")
    assert store.user_for_token(reg["token"])["name"] == "Sara"
    store.db["token_times"][reg["token"]] = "2000-01-01T00:00:00+00:00"
    assert store.user_for_token(reg["token"]) is None

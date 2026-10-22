"""Web UI end-to-end tests: real pages, real forms, through the contract.

Most tests run against tests/web_contract_double.py (M1 is real, M2-M5 are samples) and
fail if the UI sends a request body the contract does not allow. The last test runs the
student journey through the real FastAPI router when FastAPI is installed.
"""
import re
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from rafeeq.copilot import clients
from rafeeq.copilot.supervisor import clear_history
from rafeeq.platform import service
from rafeeq.samples import load_sample
from rafeeq.web import install

try:
    from tests.web_contract_double import ContractDouble
except ImportError:  # tests/ is not a package in this checkout
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from web_contract_double import ContractDouble

HX = {"HX-Request": "true"}
TEMPLATES = Path(__file__).resolve().parent.parent / "rafeeq" / "web" / "templates"


def png_bytes() -> bytes:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    img = np.full((120, 240, 3), 255, np.uint8)
    cv2.putText(img, "2x+3=11", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("RAFEEQ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAFEEQ_WEB_SECRET", "test-secret")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("RAFEEQ_API_BASE", raising=False)
    service.set_store(service.PlatformStore(tmp_path / "platform", seed_demo=True))
    clear_history()
    names = ("read_image", "tutor_answer", "check_work", "make_quiz", "next_step", "escalate", "find_verified", "record_event", "overview", "topic_of")
    for n in names:
        clients.disconnect(n)
    yield
    for n in names:
        clients.disconnect(n)
    service.set_store(None)


def make_app(disabled=None):
    double = ContractDouble(disabled)
    app = Starlette(routes=list(double.app.routes))
    install(app)
    return app, double


@pytest.fixture
def env():
    app, double = make_app()
    yield app, double
    assert double.violations == [], double.violations


def csrf_of(html: str) -> str:
    m = re.search(r'name="csrf_token" value="([^"]+)"', html) or re.search(r'"X-CSRF-Token": "([^"]+)"', html)
    assert m, "no CSRF token on the page"
    return m.group(1)


def register(client, role, name, email, join_code=""):
    page = client.get("/app/register")
    r = client.post("/app/register", data={"csrf_token": csrf_of(page.text), "role": role, "name": name,
                                           "email": email, "password": "secret1", "join_code": join_code})
    assert r.status_code == 200, r.text[:500]
    return r


def teacher_with_room(app):
    t = TestClient(app)
    register(t, "teacher", "Dr. Laila", "laila@x.com")
    page = t.get("/app/rooms")
    r = t.post("/app/rooms/create", data={"csrf_token": csrf_of(page.text), "name": "Physics 11"})
    room_id = re.search(r"/app/rooms/(r_[0-9a-f]+)/", str(r.url)).group(1)
    code = re.search(r'join-code-value" dir="ltr">([A-Z0-9-]+)<', r.text).group(1)
    return t, room_id, code


def student_in_room(app, code):
    s = TestClient(app)
    register(s, "student", "Omar", "omar@x.com")
    page = s.get("/app/rooms")
    r = s.post("/app/rooms/join", data={"csrf_token": csrf_of(page.text), "code": code})
    return s, r


# ---------------------------------------------------------------- shell, auth, language

def test_all_templates_compile():
    from rafeeq.web.common import templates
    for path in TEMPLATES.rglob("*.html"):
        templates.env.get_template(path.relative_to(TEMPLATES).as_posix())


def test_root_goes_to_login(env):
    app, _ = env
    r = TestClient(app).get("/")
    assert r.url.path == "/app/login"
    assert 'lang="ar" dir="rtl"' in r.text


def test_language_toggle_switches_to_english_ltr(env):
    app, _ = env
    c = TestClient(app)
    r = c.get("/app/lang/en", params={"next": "/app/login"})
    assert 'lang="en" dir="ltr"' in r.text and "Welcome to Rafeeq" in r.text
    r = c.get("/app/lang/ar", params={"next": "https://evil.example/"})  # open redirect is refused
    assert r.url.path.startswith("/app/")


def test_wrong_password_is_rejected(env):
    app, _ = env
    c = TestClient(app)
    page = c.get("/app/login")
    r = c.post("/app/login", data={"csrf_token": csrf_of(page.text), "email": "nobody@x.com", "password": "x"})
    assert r.status_code == 400 and "البريد أو كلمة المرور غير صحيحة" in r.text


def test_forms_need_csrf(env):
    app, _ = env
    r = TestClient(app).post("/app/login", data={"email": "a@b.com", "password": "secret1"})
    assert r.status_code == 403


def test_inserted_names_keep_their_own_direction():
    from rafeeq.web.i18n import strip_isolates, t
    text = t("en", "next.review_topic", topic="طاقة الخلية", n=3)
    assert "\u2068طاقة الخلية\u2069" in text and strip_isolates(text).startswith("Prepare a review lesson")
    assert t("ar", "insights.notes_questions", _iso=False, q="x") == "أسئلة الطلاب عن هذا الموضوع: x"


def test_no_awkward_phrase_anywhere():
    from rafeeq.web.i18n import S
    texts = " ".join(a for a, _ in S.values()) + " ".join(p.read_text(encoding="utf-8") for p in TEMPLATES.rglob("*.html"))
    assert "حوّل الارتباك" not in texts and "حول الارتباك" not in texts


# ---------------------------------------------------------------- the real student journey

def test_new_student_sees_empty_state_then_joins(env):
    app, _ = env
    _, room_id, code = teacher_with_room(app)
    s = TestClient(app)
    r = register(s, "student", "Omar", "omar@x.com")
    assert r.url.path == "/app/rooms"
    assert "لم تنضم إلى أي غرفة بعد" in r.text and "اطلب رمز الانضمام من معلّمك" in r.text

    page = s.get("/app/rooms")
    bad = s.post("/app/rooms/join", data={"csrf_token": csrf_of(page.text), "code": "ZZZ-ZZZ"})
    assert bad.status_code == 400 and "لم نجد غرفة بهذا الرمز" in bad.text

    r = s.post("/app/rooms/join", data={"csrf_token": csrf_of(page.text), "code": code.lower()})
    assert r.url.path == f"/app/rooms/{room_id}/study"
    assert "انضممت إلى \u2068Physics 11\u2069" in r.text  # name wrapped in direction isolates


def test_demo_student_also_starts_with_no_rooms(env):
    app, _ = env
    c = TestClient(app)
    page = c.get("/app/login")
    r = c.post("/app/login", data={"csrf_token": csrf_of(page.text), "demo": "student"})
    assert r.url.path == "/app/rooms" and "لم تنضم إلى أي غرفة بعد" in r.text


def test_student_cannot_open_teacher_pages(env):
    app, _ = env
    _, room_id, code = teacher_with_room(app)
    s, _ = student_in_room(app, code)
    assert s.get(f"/app/rooms/{room_id}/overview").status_code == 403
    assert s.get("/app/rooms/r_nope/study").status_code == 404


# ---------------------------------------------------------------- Ask Rafeeq

def test_copilot_answer_is_rendered_and_labelled_as_sample(env):
    app, double = env
    _, room_id, code = teacher_with_room(app)
    s, r = student_in_room(app, code)
    token = csrf_of(r.text)
    out = s.post(f"/app/rooms/{room_id}/study/message", data={"text": "ما وظيفة الميتوكوندريا؟", "mode": "auto"},
                 headers={**HX, "X-CSRF-Token": token})
    assert out.status_code == 200
    assert load_sample("tutor_answer")["answer"][:20] in out.text   # tutor sample answer
    assert "نتيجة تجريبية" in out.text                # honest label: M3 not connected
    assert f"/app/rooms/{room_id}/source?file_name=biology_ch3.pdf" in out.text
    assert any(c[1] == "/copilot/message" for c in double.calls)


def test_empty_question_is_refused(env):
    app, _ = env
    _, room_id, code = teacher_with_room(app)
    s, r = student_in_room(app, code)
    out = s.post(f"/app/rooms/{room_id}/study/message", data={"text": "  "}, headers={**HX, "X-CSRF-Token": csrf_of(r.text)})
    assert "اكتب سؤالك أولًا" in out.text


def test_photo_without_a_reader_says_so(env):
    app, _ = env
    _, room_id, code = teacher_with_room(app)
    s, r = student_in_room(app, code)
    out = s.post(f"/app/rooms/{room_id}/study/photo", files={"photo": ("w.png", png_bytes(), "image/png")},
                 headers={**HX, "X-CSRF-Token": csrf_of(r.text)})
    assert "قراءة الصور غير متاحة بعد" in out.text


def test_photo_then_check_my_solution(env):
    app, double = env
    clients.connect("read_image", lambda data: {"type": "student_work", "language": "en",
                                                "text": "2x + 3 = 11\n2x = 14", "confidence": 0.9})
    _, room_id, code = teacher_with_room(app)
    s, r = student_in_room(app, code)
    token = csrf_of(r.text)
    card = s.post(f"/app/rooms/{room_id}/study/photo", files={"photo": ("w.png", png_bytes(), "image/png")},
                  headers={**HX, "X-CSRF-Token": token})
    assert "قرأت الصورة" in card.text and "2x = 14" in card.text
    file_id = re.search(r'name="file_id" value="([^"]+)"', card.text).group(1)
    out = s.post(f"/app/rooms/{room_id}/study/message",
                 data={"file_id": file_id, "text": "2x + 3 = 11\n2x = 14", "mode": "check_work"},
                 headers={**HX, "X-CSRF-Token": token})
    assert "مراجعة حلّك" in out.text and "is-wrong" in out.text and "تلميح" in out.text
    sent = [c[2] for c in double.calls if c[1] == "/copilot/message"][-1]
    assert sent["mode"] == "check_work" and sent["file_id"] == file_id


# ---------------------------------------------------------------- practice and progress

def test_quiz_hides_answers_until_submitted(env):
    app, _ = env
    _, room_id, code = teacher_with_room(app)
    s, r = student_in_room(app, code)
    token = csrf_of(r.text)
    quiz = load_sample("make_quiz")
    qs = quiz["questions"]
    form = s.post(f"/app/rooms/{room_id}/practice/quiz", data={"topic_id": quiz["topic_id"], "count": "5"},
                  headers={**HX, "X-CSRF-Token": token})
    assert qs[0]["text"] in form.text
    assert qs[0]["explanation"] not in form.text      # explanation (and answer key) not sent before submit
    wrong = next(c for c in qs[1]["choices"] if c != qs[1]["correct_answer"])
    result = s.post(f"/app/rooms/{room_id}/practice/submit",
                    data={"quiz_id": quiz["quiz_id"], "topic_id": quiz["topic_id"], "n": str(len(qs)),
                          "q_0": "س1", "a_0": qs[0]["correct_answer"], "q_1": "س2", "a_1": wrong},
                    headers={**HX, "X-CSRF-Token": token})
    graded = load_sample("submit_quiz")
    topic = next(t["name"] for t in load_sample("room_topics")["topics"] if t["topic_id"] == quiz["topic_id"])
    assert f"نتيجتك: {graded['score']} من {graded['total']}" in result.text and topic in result.text


def test_progress_page_uses_topic_names(env):
    app, _ = env
    _, room_id, code = teacher_with_room(app)
    s, _ = student_in_room(app, code)
    r = s.get(f"/app/rooms/{room_id}/progress")
    assert r.status_code == 200 and "الوراثة" in r.text and "t_genetics" not in r.text.split("practice?topic=")[0][-200:]


# ---------------------------------------------------------------- teacher

def test_teacher_overview_shows_code_and_real_next_step(env):
    app, _ = env
    t, room_id, code = teacher_with_room(app)
    r = t.get(f"/app/rooms/{room_id}/overview")
    assert code in r.text
    assert "لم ترفع أي ملف بعد" in r.text and "ارفع أول ملف لتبدأ" in r.text
    assert "لم ينضم أي طالب بعد" in r.text


def test_upload_shows_file_as_uploaded_not_ready(env):
    app, _ = env
    t, room_id, _ = teacher_with_room(app)
    page = t.get(f"/app/rooms/{room_id}/materials")
    r = t.post(f"/app/rooms/{room_id}/materials/upload", data={"csrf_token": csrf_of(page.text)},
               files={"file": ("notes.pdf", b"%PDF-1.4 test", "application/pdf")})
    assert "notes.pdf" in r.text and "تم الرفع" in r.text and "جاهز</span>" not in r.text


def test_teacher_can_delete_a_file_and_students_see_no_delete_button(env):
    app, _ = env
    t, room_id, code = teacher_with_room(app)
    page = t.get(f"/app/rooms/{room_id}/materials")
    t.post(f"/app/rooms/{room_id}/materials/upload", data={"csrf_token": csrf_of(page.text)},
           files={"file": ("slides.pptx", b"pptx", "application/octet-stream")})
    page = t.get(f"/app/rooms/{room_id}/materials")
    assert "/materials/" in page.text and "/delete" in page.text
    action = page.text.split('action="')[1:]
    delete_url = next(a.split('"')[0] for a in action if a.split('"')[0].endswith("/delete"))
    s, _ = student_in_room(app, code)
    assert "/delete" not in s.get(f"/app/rooms/{room_id}/materials").text
    r = t.post(delete_url, data={"csrf_token": csrf_of(page.text)})
    assert "slides.pptx" not in r.text and "تم حذف الملف" in r.text


def test_students_page_lists_joined_student_without_invented_mastery(env):
    app, _ = env
    t, room_id, code = teacher_with_room(app)
    student_in_room(app, code)
    r = t.get(f"/app/rooms/{room_id}/students")
    assert "Omar" in r.text and "لا توجد بيانات بعد" in r.text


def test_insights_link_prefills_the_studio(env):
    app, _ = env
    t, room_id, _ = teacher_with_room(app)
    r = t.get(f"/app/rooms/{room_id}/insights")
    assert "طاقة الخلية" in r.text and "جهّز درس مراجعة" in r.text
    link = re.search(r'href="(/app/rooms/[^"]+/studio\?from=insights[^"]+)"', r.text).group(1).replace("&amp;", "&")
    studio = t.get(link)
    assert 'value="طاقة الخلية"' in studio.text and "ما وظيفة الميتوكوندريا" in studio.text


def test_studio_generate_review_approve_download(env):
    app, double = env
    t, room_id, _ = teacher_with_room(app)
    page = t.get(f"/app/rooms/{room_id}/studio")
    token = csrf_of(page.text)
    out = t.post(f"/app/rooms/{room_id}/studio/sessions",
                 data={"topic": "الوراثة", "needs": ["slides", "quiz"], "student_level": "intermediate",
                       "duration_minutes": "45", "language": "Arabic", "num_quiz_questions": "5",
                       "learning_objectives": "يفرّق بين الجين والأليل\n", "notes": ""},
                 headers={**HX, "X-CSRF-Token": token})
    sess = load_sample("lecturer_session")
    if sess.get("status") == "awaiting_approval":
        assert "بانتظار مراجعتك" in out.text and "اعتماد وإنشاء الملفات" in out.text
    outline = (sess.get("draft") or {}).get("outline") or []
    if outline:
        assert outline[0] in out.text
    sent = [c[2] for c in double.calls if c[1] == "/teacher/sessions"][-1]
    assert sent["room_id"] == room_id and sent["needs"] == ["slides", "quiz"]
    sid = sess["session_id"]
    no_fb = t.post(f"/app/rooms/{room_id}/studio/sessions/{sid}/decision", data={"decision": "revise"},
                   headers={**HX, "X-CSRF-Token": token})
    assert "اكتب ما تريد تعديله" in no_fb.text
    ok = t.post(f"/app/rooms/{room_id}/studio/sessions/{sid}/decision", data={"decision": "revise", "feedback": "أقصر"},
                headers={**HX, "X-CSRF-Token": token})
    assert ok.status_code == 200
    first = load_sample("teacher_sessions")["sessions"][0]
    f = t.get(f"/app/rooms/{room_id}/studio/sessions/{first['session_id']}/files/{first['files'][0]['name']}")
    assert f.content == b"PPTX-BYTES" and "attachment" in f.headers["content-disposition"]


def test_lesson_studio_stays_focused_and_the_other_studios_have_their_own_pages(env):
    app, _ = env
    t, room_id, _ = teacher_with_room(app)
    page = t.get(f"/app/rooms/{room_id}/studio")
    assert page.status_code == 200
    assert "ابحث عن شرح بالفيديو" not in page.text and "ولّد فيديو متحركًا" not in page.text   # moved out of the Studio
    for path in ("/app/video", "/app/presentations", "/app/media", "/app/learning", "/app/library"):
        assert f'href="{path}' in page.text, path                                             # one sidebar link each
    for path, heading in (("/app/video", "اكتشاف الفيديو"), ("/app/presentations", "العروض التقديمية"),
                          ("/app/media", "الصور والوسائط"), ("/app/learning", "منصتي التعليمية"), ("/app/library", "مكتبتي")):
        r = t.get(path)
        assert r.status_code == 200 and heading in r.text and 'aria-current="page"' in r.text, path
    assert t.get("/app/studio", follow_redirects=False).headers["location"].endswith("/studio")


def test_video_studio_search_opens_each_video_at_its_matching_second(env):
    app, double = env
    t, room_id, _ = teacher_with_room(app)
    page = t.get(f"/app/video?room={room_id}")
    assert "ابحث عن شرح بالفيديو" in page.text and "Physics 11" in page.text
    token = csrf_of(page.text)
    out = t.post("/app/video/search", data={"q": "ما معنى البناء الضوئي؟", "room_id": room_id},
                 headers={**HX, "X-CSRF-Token": token})
    first = load_sample("video_search")["results"][0]
    assert first["open_at"] in out.text and first["video_id"] in out.text
    assert "data-video-prev" in out.text and "data-video-next" in out.text and "data-cover-id" in out.text
    assert [c for c in double.calls if c[1] == "/teacher/videos/search"][-1][2]["q"] == "ما معنى البناء الضوئي؟"
    short = t.post("/app/video/search", data={"q": "a"}, headers={**HX, "X-CSRF-Token": token})
    assert "اكتب السؤال أولًا" in short.text


def test_media_studio_video_generation_polls_until_the_clip_is_ready(env):
    app, double = env
    t, room_id, _ = teacher_with_room(app)
    page = t.get("/app/media")
    assert "ولّد فيديو متحركًا" in page.text
    token = csrf_of(page.text)
    started = t.post("/app/media/video/generate", data={"prompt": "نبات يمتص الماء من التربة", "duration": "5"},
                     headers={**HX, "X-CSRF-Token": token})
    job = load_sample("video_job")
    assert f"/media/video/generate/{job['task_id']}" in started.text and 'hx-trigger="load delay:5s"' in started.text
    sent = [c[2] for c in double.calls if c[1] == "/teacher/videos/generate"][-1]
    assert sent["audio"] == "false" and sent["duration"] == "5" and sent["teacher_id"].startswith("u_")
    done = t.get(f"/app/media/video/generate/{job['task_id']}", headers=HX)
    assert load_sample("video_job_status")["video_url"] in done.text and "<video" in done.text
    gif = t.post("/app/media/video/generate", data={"prompt": "مشهد تعليمي بسيط"},
                 files={"image": ("x.gif", b"GIF89a", "image/gif")}, headers={**HX, "X-CSRF-Token": token})
    assert "JPG أو PNG أو WebP" in gif.text
    no_prompt = t.post("/app/media/video/generate", data={"prompt": ""}, headers={**HX, "X-CSRF-Token": token})
    assert "اكتب وصف المشهد أولًا" in no_prompt.text


def test_teacher_saves_a_video_moment_to_the_library(env):
    app, double = env
    t, room_id, _ = teacher_with_room(app)
    token = csrf_of(t.get(f"/app/video?room={room_id}").text)
    found = t.post("/app/video/search", data={"q": "ما معنى البناء الضوئي؟", "room_id": room_id},
                   headers={**HX, "X-CSRF-Token": token})
    assert "data-save-url" in found.text and "/app/video/moments" in found.text
    r = t.post("/app/video/moments", headers={**HX, "X-CSRF-Token": token},
               data={"title": "البناء الضوئي", "source_url": "https://www.youtube.com/watch?v=UPBMG5EYydo&t=92s",
                     "start_sec": "92.4", "end_sec": "138", "room_id": room_id})
    assert "تم الحفظ في مكتبتك" in r.text
    sent = [c[2] for c in double.calls if c[1] == "/teacher/assets" and isinstance(c[2], dict) and "title" in c[2]][-1]
    assert sent["room_id"] == room_id and sent["start_sec"] == 92.4 and sent["source_url"].startswith("https://")
    empty = t.post("/app/video/moments", data={"title": ""}, headers={**HX, "X-CSRF-Token": token})
    assert "تعذّر حفظ هذا العنصر" in empty.text


def test_a_saved_generated_video_is_played_from_the_library(env):
    app, double = env
    double.video_saved = True
    t, room_id, _ = teacher_with_room(app)
    job = load_sample("video_job")
    done = t.get(f"/app/media/video/generate/{job['task_id']}", headers=HX)
    assert "/app/assets/a_77aa10cd/file" in done.text and "حُفظ الفيديو في مكتبتك" in done.text
    assert "/app/media?asset=a_77aa10cd" in done.text                                        # edit it in the media studio
    file = t.get("/app/assets/a_77aa10cd/file")
    assert file.status_code == 200 and file.content == b"MP4-BYTES" and file.headers["content-type"] == "video/mp4"


def test_library_page_lists_the_teachers_assets(env):
    app, double = env
    t, room_id, _ = teacher_with_room(app)
    page = t.get("/app/library")
    assert page.status_code == 200 and "مكتبتي" in page.text
    samples = load_sample("assets")["assets"]
    video = next(a for a in samples if a["kind"] == "generated_video")
    moment = next(a for a in samples if a["kind"] == "source_video")
    assert video["title"] in page.text and moment["title"] in page.text
    assert f"/app/assets/{video['asset_id']}/file" in page.text and "<video" in page.text      # plays our own copy
    escaped = moment["source_url"].replace("&", "&amp;")                                       # HTML-escaped in href
    assert escaped in page.text and 'rel="noopener noreferrer"' in page.text                   # opens YouTube
    assert "01:15" in page.text                                                                # 75 s -> 01:15
    assert f"/app/assets/{video['asset_id']}/file?download=1" in page.text


def test_library_filters_are_sent_to_the_api_and_junk_is_dropped(env):
    app, double = env
    t, room_id, _ = teacher_with_room(app)
    t.get(f"/app/library?kind=generated_video&room={room_id}")
    sent = [c[2] for c in double.calls if c[0] == "GET" and c[1] == "/teacher/assets"][-1]
    assert sent == {"kind": "generated_video", "room_id": room_id}
    t.get("/app/library?kind=%3Cscript%3E")
    sent = [c[2] for c in double.calls if c[0] == "GET" and c[1] == "/teacher/assets"][-1]
    assert "kind" not in sent


def test_library_is_for_teachers_and_linked_from_the_sidebar(env):
    app, _ = env
    t, room_id, code = teacher_with_room(app)
    assert 'href="/app/library"' in t.get("/app/profile").text
    s, _ = student_in_room(app, code)
    assert s.get("/app/library", follow_redirects=False).status_code == 403
    assert s.get("/app/assets/a_77aa10cd/file", follow_redirects=False).status_code == 403


def test_library_says_when_the_api_is_not_there():
    app, _ = make_app(disabled={"assets"})
    t, _, _ = teacher_with_room(app)
    page = t.get("/app/library")
    assert page.status_code == 200 and "<video" not in page.text


def test_teacher_saves_a_presenter_image_on_the_profile(env):
    app, _ = env
    t, _, _ = teacher_with_room(app)
    page = t.get("/app/profile")
    assert "صورة المقدّم" in page.text
    r = t.post("/app/profile/avatar", data={"csrf_token": csrf_of(page.text)},
               files={"image": ("me.png", b"\x89PNG-ME", "image/png")})
    assert "تم حفظ الصورة" in r.text
    img = t.get("/app/profile/avatar")
    assert img.status_code == 200 and img.headers["content-type"].startswith("image/")


def test_studio_requires_topic_and_needs(env):
    app, _ = env
    t, room_id, _ = teacher_with_room(app)
    token = csrf_of(t.get(f"/app/rooms/{room_id}/studio").text)
    r = t.post(f"/app/rooms/{room_id}/studio/sessions", data={"topic": "x"}, headers={**HX, "X-CSRF-Token": token})
    assert "اكتب موضوع الدرس" in r.text
    r = t.post(f"/app/rooms/{room_id}/studio/sessions", data={"topic": "الوراثة"}, headers={**HX, "X-CSRF-Token": token})
    assert "اختر شيئًا واحدًا على الأقل" in r.text


def test_assistant_draft_and_profile_save(env):
    app, _ = env
    t, room_id, _ = teacher_with_room(app)
    token = csrf_of(t.get(f"/app/rooms/{room_id}/students").text)
    d = t.post(f"/app/rooms/{room_id}/students/draft", data={"request": "ذكّرهم بالمراجعة"}, headers={**HX, "X-CSRF-Token": token})
    assert "بانتظار مراجعتك" in d.text and "data-copy=" in d.text
    page = t.get("/app/profile")
    r = t.post("/app/profile", data={"csrf_token": csrf_of(page.text), "name": "ليلى", "title": "أ.", "subjects": "فيزياء، رياضيات",
                                     "bio": "", "language": "Arabic", "student_level": "advanced",
                                     "programming_language": "python", "duration_minutes": "50"})
    assert "تم الحفظ" in r.text


# ---------------------------------------------------------------- honesty when a teammate's part is missing

def test_missing_services_show_not_connected_instead_of_fake_data():
    app, double = make_app(disabled={"insights", "report", "summary", "next_step", "topics", "profile", "sessions"})
    t, room_id, code = teacher_with_room(app)
    for page in ("overview", "insights", "studio"):
        r = t.get(f"/app/rooms/{room_id}/{page}")
        assert r.status_code == 200, page
    assert "هذه البيانات غير متصلة بعد" in t.get(f"/app/rooms/{room_id}/insights").text
    s, _ = student_in_room(app, code)
    r = s.get(f"/app/rooms/{room_id}/progress")
    assert r.status_code == 200 and "هذه البيانات غير متصلة بعد" in r.text
    assert "هذه البيانات غير متصلة بعد" in s.get(f"/app/rooms/{room_id}/practice").text
    assert double.violations == []


# ---------------------------------------------------------------- the real FastAPI router (runs in the repo)

def test_journey_through_real_fastapi_router():
    fastapi = pytest.importorskip("fastapi")
    from rafeeq.platform.router import router

    app = fastapi.FastAPI()
    app.include_router(router)
    install(app)
    api = TestClient(app)

    teacher = api.post("/auth/register", json={"name": "T", "email": "t@x.com", "password": "secret1", "role": "teacher"}).json()
    created = api.post("/rooms", json={"name": "Chem"}, headers={"Authorization": f"Bearer {teacher['token']}"}).json()
    assert api.get(f"/rooms/{created['room']['room_id']}/join-code",
                   headers={"Authorization": f"Bearer {teacher['token']}"}).json() == {"join_code": created["join_code"]}
    assert api.get("/rooms").json()["rooms"]  # no token: legacy sample still served

    s = TestClient(app)
    r = register(s, "student", "Sara", "sara2@x.com")
    assert "لم تنضم إلى أي غرفة بعد" in r.text
    r = s.post("/app/rooms/join", data={"csrf_token": csrf_of(r.text), "code": created["join_code"]})
    assert r.url.path.endswith("/study")
    out = s.post(f"/app/rooms/{created['room']['room_id']}/study/message", data={"text": "Explain bonds please"},
                 headers={**HX, "X-CSRF-Token": csrf_of(r.text)})
    assert out.status_code == 200 and "bubble" in out.text


def test_a_crash_inside_a_teammates_route_shows_a_notice_not_a_500(env):
    app, _ = env
    def broken(room, question):
        raise RuntimeError("M3 exploded")
    clients.connect("tutor_answer", broken)
    _, room_id, code = teacher_with_room(app)
    s, r = student_in_room(app, code)
    out = s.post(f"/app/rooms/{room_id}/study/message", data={"text": "Explain friction please"},
                 headers={**HX, "X-CSRF-Token": csrf_of(r.text)})
    assert out.status_code == 200 and "حدث خطأ. حاول مرة أخرى." in out.text


@pytest.mark.parametrize("nxt", ["https://evil.example/app/x", "//evil.example/app/", "/app/\\evil.example", "/other/page"])
def test_language_switch_never_redirects_off_site(env, nxt):
    app, _ = env
    r = TestClient(app).get("/app/lang/en", params={"next": nxt}, follow_redirects=False)
    assert r.headers["location"] == "/app/"


def test_pages_send_a_referrer_so_embedded_youtube_players_work(env):
    app, _ = env
    r = TestClient(app).get("/app/login")
    assert r.headers["referrer-policy"] == "strict-origin-when-cross-origin"     # "same-origin" made YouTube refuse to play


def test_a_saved_youtube_moment_plays_only_its_clip():
    from rafeeq.web.common import yt_embed

    url = yt_embed("https://www.youtube.com/watch?v=abc12345678&t=75s", 75.0, 130.0)
    assert url == "https://www.youtube-nocookie.com/embed/abc12345678?rel=0&start=75&end=130"
    assert yt_embed("https://example.com/watch?v=abc12345678") == "" and yt_embed("") == ""


def test_teacher_opens_a_parent_report_and_the_share_link_works_without_login(env):
    app, double = env
    t, room_id, code = teacher_with_room(app)
    student_in_room(app, code)
    page = t.get(f"/app/rooms/{room_id}/students")
    link = re.search(r'href="(/app/rooms/' + room_id + r'/students/[^"]+/report)"', page.text).group(1)
    report = t.get(link)
    assert report.status_code == 200 and "تقرير وليّ الأمر" in report.text and "Omar" in report.text
    assert "لم يسجّل Omar نشاطًا" in report.text            # honest: no activity yet, nothing invented
    share = re.search(r'data-copy="(http[^"]+/app/p/[^"]+)"', report.text).group(1)
    public = TestClient(app).get("/" + share.split("://", 1)[1].split("/", 1)[1])
    assert public.status_code == 200 and "Omar" in public.text and "logout" not in public.text
    bad = TestClient(app).get("/app/p/not-a-real-token")
    assert bad.status_code == 404 and "انتهت صلاحية الرابط" in bad.text
    assert not double.violations


def test_teacher_answers_an_escalated_question_and_the_student_sees_the_reply(env):
    from rafeeq.teacher import inbox

    app, double = env
    t, room_id, code = teacher_with_room(app)
    s, _ = student_in_room(app, code)
    omar = next(uid for uid, u in service.get_store().db["users"].items() if u["name"] == "Omar")
    filed = inbox.escalate(room_id, "ما الفرق بين TWT و Klystron؟", omar, store=double.class_store)
    page = t.get(f"/app/rooms/{room_id}/inbox")
    assert "ما الفرق بين TWT و Klystron؟" in page.text and "بانتظار ردّك" in page.text
    r = t.post(f"/app/rooms/{room_id}/inbox/{filed['question_id']}/answer",
               data={"csrf_token": csrf_of(page.text), "answer": "الـ TWT يكبّر على نطاق ترددي واسع."})
    assert "وصل ردّك للطالب" in r.text and "تم الرد" in r.text
    study = s.get(f"/app/rooms/{room_id}/study")
    assert "ردود معلّمك" in study.text and "يكبّر على نطاق ترددي واسع" in study.text
    found = inbox.find_verified(room_id, "ايه الفرق بين TWT و Klystron؟", store=double.class_store)
    assert found and found["answer"].startswith("الـ TWT")
    assert not double.violations


def test_the_teacher_chooses_whether_the_tutor_may_answer_from_the_web(env):
    app, double = env
    t, room_id, _ = teacher_with_room(app)
    page = t.get(f"/app/rooms/{room_id}/overview")
    assert "المادة أولاً، ثم الويب" in page.text and 'value="material_then_web" checked' in page.text
    r = t.post(f"/app/rooms/{room_id}/settings", data={"csrf_token": csrf_of(page.text), "web_answers": "material_only"})
    assert r.status_code == 200 and 'value="material_only" checked' in r.text
    assert double.settings[room_id] == "material_only" and double.violations == []



def test_a_web_answer_waits_in_the_inbox_and_the_teacher_can_approve_it(env):
    from rafeeq.teacher import inbox

    app, double = env
    t, room_id, code = teacher_with_room(app)
    s, _ = student_in_room(app, code)
    omar = next(uid for uid, u in service.get_store().db["users"].items() if u["name"] == "Omar")
    kept = inbox.log_web_answer(room_id, "ما هو الـ magnetron؟", omar,
                                {"answer": "الماجنترون مذبذب ميكروويف [1]", "web_sources": [{"n": 1, "title": "Magnetron", "url": "https://example.org/m"}]},
                                store=double.class_store)
    page = t.get(f"/app/rooms/{room_id}/inbox")
    assert "أُجيب من الويب" in page.text and "الماجنترون مذبذب ميكروويف" in page.text and "اعتماد الإجابة" in page.text
    assert inbox.find_verified(room_id, "ما هو الـ magnetron؟", store=double.class_store) is None     # not reused before the teacher approves
    assert "ما هو الـ magnetron؟" not in s.get(f"/app/rooms/{room_id}/study").text                    # the student already got the web answer
    t.post(f"/app/rooms/{room_id}/inbox/{kept['question_id']}/answer",
           data={"csrf_token": csrf_of(page.text), "answer": "الماجنترون مذبذب ميكروويف [1]"})
    assert inbox.find_verified(room_id, "ما هو الـ magnetron؟", store=double.class_store)["answer"].startswith("الماجنترون")
    assert not double.violations

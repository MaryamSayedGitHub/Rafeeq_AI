"""Adversarial inputs for Member 3: blurry, rotated, cropped, low-light, crossed-out, ambiguous-sign, mixed Arabic/English, incomplete and
unsupported input; the agent's routing, retries and fallbacks; citations that must be real; hints that must never be the solution.
Every test runs the real code (OpenCV, grader, agent, API); only the model, the web and M2/M4 are replaced."""
import base64

import cv2
import numpy as np
import pytest
from test_member3 import SARA, FakeReader, _photo, assert_contract, reader, world  # noqa: F401  (fixtures)

from rafeeq.tutor import agent, citations, m2, uncertainty, web_search
from rafeeq.tutor import llm as llm_mod
from rafeeq.tutor.check_work import check_work_detail
from rafeeq.tutor.grading import _WORKING

PHOTO = {"photo": ("w.png", None, "image/png")}


def post_photo(client, data, lang="en", annotate=False):
    return client.post("/rooms/r_bio101/vision/check-work", files={"photo": ("w.png", data, "image/png")},
                       data={"language": lang, "annotate": str(annotate).lower()}, headers=SARA)


def png(img) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def draw(lines=("2 + 3 = 5", "4 x 2 = 9")):
    return cv2.imdecode(np.frombuffer(_photo(lines), np.uint8), cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------------- bad photos: a retake tip, never a verdict
@pytest.mark.parametrize("name,make", [
    ("blurry", lambda: cv2.GaussianBlur(draw(), (0, 0), 25)),
    ("low light", lambda: (draw() * 0.18).astype(np.uint8)),
    ("faded", lambda: cv2.addWeighted(draw(), 0.15, np.full((520, 900, 3), 255, np.uint8), 0.85, 0)),
])
def test_unusable_photos_get_a_retake_tip_and_no_judgement(world, reader, name, make):
    client, sent = world
    reader(["2 + 3 = 5", "4 × 2 = 9"])
    body = post_photo(client, png(make())).json()
    assert_contract("VisionCheck", body)
    assert body["quality_ok"] is False and body["retake_tip"] and body["lines"] == [] and not body["needs_help"] and sent == [], name


@pytest.mark.parametrize("name,make", [
    ("rotated 8 degrees", lambda: cv2.warpAffine(draw(), cv2.getRotationMatrix2D((450, 260), 8, 1.0), (900, 520), borderValue=(255, 255, 255))),
    ("rotated 90 degrees", lambda: cv2.rotate(draw(), cv2.ROTATE_90_CLOCKWISE)),
    ("upside down", lambda: cv2.rotate(draw(), cv2.ROTATE_180)),
    ("cropped to the top half", lambda: draw()[:200]),
    ("cropped to a corner", lambda: draw()[:300, :400]),
    ("tiny", lambda: cv2.resize(draw(), (90, 52))),
    ("all white", lambda: np.full((520, 900, 3), 255, np.uint8)),
])
def test_awkward_photos_never_crash_and_always_match_the_contract(world, reader, name, make):
    client, _ = world
    reader(["2 + 3 = 5", "4 × 2 = 9"])
    r = post_photo(client, png(make()))
    assert r.status_code == 200, (name, r.text)
    body = r.json()
    assert_contract("VisionCheck", body)
    assert all(ln["status"] in ("ok", "check", "error") for ln in body["lines"])
    assert all(ln["hint"] == "" for ln in body["lines"] if ln["status"] == "ok")


def test_unsupported_files_are_refused_politely(world):
    client, _ = world
    assert client.post("/rooms/r/vision/check-work", files={"photo": ("a.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/rooms/r/vision/check-work", files={"photo": ("a.png", b"", "image/png")}).status_code in (400, 422)
    assert client.post("/rooms/r/vision/check-work", files={"photo": ("a.png", b"\x89PNG\r\n\x1a\ntruncated", "image/png")}).status_code == 400
    assert client.post("/rooms/r/vision/check-work", files={"photo": ("big.png", b"0" * (12 * 1024 * 1024 + 1), "image/png")}).status_code == 413


def test_a_photo_with_no_readable_text_is_not_judged(world, reader):
    client, sent = world
    reader([""])
    body = post_photo(client, _photo(("2 + 3 = 5",))).json()
    assert_contract("VisionCheck", body)
    assert body["lines"] == [] and body["needs_help"] is False and sent == []


# ----------------------------------------------------------------------------- uncertainty: ask, do not judge
def test_a_crossed_out_line_is_asked_about_even_if_the_arithmetic_is_wrong(world, reader):
    client, _ = world
    reader(["2 + 3 = 5", "~~4 × 2 = 9~~"])                        # 4 x 2 = 9 is wrong, but it is crossed out: the student already dropped it
    body = post_photo(client, _photo(("2 + 3 = 5", "4 x 2 = 9"))).json()
    assert_contract("VisionCheck", body)
    ok, struck = body["lines"]
    assert ok["status"] == "ok" and struck["status"] == "check" and "crossed out" in struck["hint"]
    assert body["needs_clarification"] is True and body["clarification"] and not any(ln["status"] == "error" for ln in body["lines"])


@pytest.mark.parametrize("line", ["x ? 5 = 15", "x ± 5 = 15", "x +- 5 = 15"])
def test_an_ambiguous_sign_is_a_question_not_an_error(world, reader, line, monkeypatch):
    client, _ = world
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: {"steps": [{"i": 1, "status": "ok", "hint": ""}, {"i": 2, "status": "ok", "hint": ""}], "complete": True, "concepts": ""})
    reader(["x - 5 = 10", line])
    body = post_photo(client, _photo(("x - 5 = 10", "x + 5 = 15"))).json()
    assert [ln["status"] for ln in body["lines"]] == ["ok", "check"] and "sign" in body["lines"][1]["hint"] and body["needs_clarification"]


def test_an_unreadable_marker_from_the_reader_is_asked_about(world, reader):
    client, _ = world
    reader(["3x = 12", "x = [?]"])
    body = post_photo(client, _photo(("3x = 12", "x = 4")), lang="ar").json()
    assert body["lines"][1]["status"] == "check" and any("؀" <= ch <= "ۿ" for ch in body["lines"][1]["hint"])      # asked in the site language


def test_a_page_the_reader_was_unsure_about_is_not_judged_at_all(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: pytest.fail("the model must not be asked to judge an unreliable page"))
    det = check_work_detail("r", "", "2 + 3 = 5\n4 x 2 = 9\nx = 7", "en", confidence=0.3)
    assert [it["status"] for it in det["items"]] == ["check"] * 3 and all(it["hint"] for it in det["items"])
    assert det["needs_clarification"] and det["clarification"] and det["boxes"] == [] and det["reliable"] is False


def test_most_lines_unreliable_means_the_whole_page_is_not_judged(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: pytest.fail("no model call expected"))
    det = check_work_detail("r", "", "x = [?]\n~~y = 2~~\n3 + 4 = 7", "en")
    assert [it["status"] for it in det["items"]] == ["check"] * 3 and det["needs_clarification"]


def test_one_bad_line_of_two_does_not_void_the_other(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: None)
    det = check_work_detail("r", "", "2 + 3 = 5\nx = [?]", "en")
    assert [it["status"] for it in det["items"]] == ["ok", "check"]


def test_mixed_arabic_and_english_is_fine_but_a_scrambled_token_is_not():
    assert uncertainty.line_reasons("Let س = 4 then س + 3 = 7") == []                       # a normal bilingual line
    assert uncertainty.line_reasons("احسب the value of 2x + 6 = 14") == []
    assert "mixed_script" in uncertainty.line_reasons("xسy + 3 = 7")                          # letters of two scripts inside ONE word: a misread
    assert uncertainty.line_reasons("2 + 3 = 5") == [] and uncertainty.line_reasons("def f(n):") == [] and uncertainty.line_reasons("    return n * 2") == []


def test_garbage_is_noisy():
    assert "noisy" in uncertainty.line_reasons("¤§¶•◊∆≈¬¦¢£")
    assert "unreadable" in uncertainty.line_reasons("2x = �")


# ----------------------------------------------------------------------------- incomplete work
def test_incomplete_work_is_not_called_finished_and_the_hint_is_not_the_answer(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: {"steps": [{"i": 1, "status": "ok", "hint": ""}, {"i": 2, "status": "ok", "hint": ""}], "complete": False, "concepts": "square root"})
    det = check_work_detail("r", "", "x^2 - 9 = 0\nx^2 = 9", "en")
    assert det["needs_help"] and det["complete"] is False
    assert "x = 3" not in det["hint"] and "±3" not in det["hint"] and "next step" in det["hint"]


# ----------------------------------------------------------------------------- hints never reveal the solution
LEAKS = ["x = 4", "2x = 8", "Subtract 6 from both sides to get 2x = 8", "The answer is 4", "14 - 6 = 8, so x = 4", "x=4", "So 2x = 8. Then divide by 2."]


@pytest.mark.parametrize("hint", LEAKS)
def test_a_model_hint_that_contains_the_solution_is_replaced(monkeypatch, hint):
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: {"steps": [{"i": 1, "status": "ok", "hint": ""}, {"i": 2, "status": "error", "hint": hint}], "complete": True, "concepts": ""})
    det = check_work_detail("r", "", "2x + 6 = 14\n2x = 14 + 6", "en")
    for it in det["items"]:
        assert not _WORKING.search(it["hint"]), it
        for secret in ("x = 4", "2x = 8", "14 - 6", "x=4"):
            assert secret not in it["hint"]
    bad = det["items"][1]
    assert bad["status"] == "error" and bad["hint"] and bad["hint"].count(".") <= 1


def test_the_hint_for_the_arabic_site_is_arabic_and_never_the_line(monkeypatch):
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: {"steps": [{"i": 1, "status": "error", "hint": "الحل هو س = 4 لأن 2س = 8"}], "complete": True, "concepts": ""})
    det = check_work_detail("r", "", "2س + 6 = 14", "ar")
    hint = det["items"][0]["hint"]
    assert hint and "س = 4" not in hint and "2س = 8" not in hint and any("؀" <= c <= "ۿ" for c in hint)


# ----------------------------------------------------------------------------- citations: only passages M2 really returned
def test_verify_drops_invented_citations():
    got = [{"file_name": "biology_ch3.pdf", "page": 12, "text": "الميتوكوندريا هي العضية المسؤولة عن إنتاج الطاقة"}]
    v = citations.verify([{"file_name": "biology_ch3.pdf", "page": 12}, {"file_name": "chem.pdf", "page": 3}, {"file_name": "biology_ch3.pdf", "page": 99},
                          {"file_name": "", "page": 1}, {"file_name": "biology_ch3.pdf", "page": 0}], got, "الميتوكوندريا تنتج الطاقة")
    assert v["valid"] == [{"file_name": "biology_ch3.pdf", "page": 12}] and len(v["invalid"]) == 4 and v["ok"] and v["support"] == "supported"
    assert citations.verify([{"file_name": "x.pdf", "page": 1}], got, "text")["ok"] is False


def test_an_answer_its_own_citation_does_not_support_is_flagged_and_withheld(world, monkeypatch):
    client, _ = world
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: {"grounded": True, "answer": "باريس هي عاصمة فرنسا وبرج إيفل موجود هناك.", "steps": [], "used": [1]})
    body = client.post("/rooms/r_bio101/ask", json={"question": "ما هي عاصمة فرنسا؟", "language": "ar"}, headers=SARA).json()
    assert_contract("RoomAskResponse", body)
    assert body["answer_flagged"] is True and body["grounded"] is False and body["citations"] == [] and body["offer_level2"] is True


def test_a_citation_number_the_model_made_up_is_dropped(world, monkeypatch):
    client, _ = world
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: {"grounded": True, "answer": "الميتوكوندريا تنتج الطاقة", "steps": [], "used": [7, 0, -1]})
    body = client.post("/rooms/r_bio101/ask", json={"question": "ما وظيفة الميتوكوندريا؟", "language": "ar"}, headers=SARA).json()
    assert body["grounded"] is False and body["citations"] == []


def test_a_supported_answer_keeps_only_real_citations(world, monkeypatch):
    client, _ = world
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: {"grounded": True, "answer": "الميتوكوندريا هي العضية المسؤولة عن إنتاج الطاقة.", "steps": [], "used": [1, 99]})
    body = client.post("/rooms/r_bio101/ask", json={"question": "ما وظيفة الميتوكوندريا؟", "language": "ar"}, headers=SARA).json()
    assert body["grounded"] and body["citations"] == [{"file_name": "biology_ch3.pdf", "page": 12}] and body["citation_support"] == "supported"


def test_vision_citations_are_passages_that_were_retrieved(world, reader):
    client, _ = world
    reader(["الميتوكوندريا تنتج الطاقة", "ATP = طاقة"])
    body = post_photo(client, _photo(("a", "b")), lang="ar").json()
    real = {(c["file_name"], c["page"]) for c in m2.search("r_bio101", "x")[0]}
    assert all((c["file_name"], c["page"]) in real for c in body["citations"])


# ----------------------------------------------------------------------------- the agent: conditional routing, capped retries, fallbacks
@pytest.fixture
def web(monkeypatch):
    calls = []
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    return calls


def test_handwriting_goes_to_check_work_and_never_to_the_web(world, monkeypatch):
    monkeypatch.setattr(web_search, "multi_search", lambda *a, **k: pytest.fail("Tavily must not be called for handwriting"))
    st = agent.run_check("r", "2 + 3 = 5\n4 x 2 = 9", "", "u_sara", "en")
    assert st["trace"] == ["tag_question", "check_work"] and st["work"]["needs_help"] and st["route"] == "check_work"


def test_level1_never_calls_the_web(world, monkeypatch):
    monkeypatch.setattr(web_search, "multi_search", lambda *a, **k: pytest.fail("Level 1 must not spend the Tavily quota"))
    st = agent.run_qa("r_bio101", "ما وظيفة الميتوكوندريا؟", 1, "u_sara")
    assert st["route"] == "answer" and "web_search" not in st["trace"]


def test_a_transient_web_failure_is_retried_once_then_falls_back_to_m2(world, web, monkeypatch):
    def boom(*a, **k):
        web.append(1)
        raise RuntimeError("network blip")
    monkeypatch.setattr(web_search, "multi_search", boom)
    st = agent.run_qa("r_bio101", "ما وظيفة الميتوكوندريا؟", 2, "u_sara")
    assert len(web) == 1 + agent.MAX_RETRIES and st["trace"].count("web_search") == 2                  # tried, retried once, stopped
    assert st["fallback"] == "m2" and st["answer"]["grounded"] and st["answer"]["citations"] and not st.get("study_file")


def test_a_missing_key_or_quota_is_never_retried(world, web, monkeypatch):
    def nokey(*a, **k):
        web.append(1)
        raise web_search.SearchError("TAVILY_API_KEY is not set.")
    monkeypatch.setattr(web_search, "multi_search", nokey)
    st = agent.run_qa("r_bio101", "what is a list", 2, "u_sara")
    assert len(web) == 1 and st["fallback"] == "m2"


def test_a_study_file_failure_falls_back_instead_of_an_error_page(world, web, monkeypatch):
    monkeypatch.setattr(web_search, "multi_search", lambda *a, **k: {"sources": [{"n": 1, "title": "T", "url": "https://e.org/1", "content": "x " * 80}], "cache_hits": 0, "api_calls": 1, "errors": []})
    monkeypatch.setattr(agent.research, "build_study_file", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    st = agent.run_qa("r_bio101", "ما وظيفة الميتوكوندريا؟", 2, "u_sara")
    assert st["trace"][-3:] == ["write_study_file:failed", "fallback_to_m2", "answer"] or "fallback_to_m2" in st["trace"]
    assert st["fallback"] == "m2" and st["answer"]["grounded"]


def test_the_api_answers_a_failed_level2_from_m2_and_says_so(world, web, monkeypatch):
    client, _ = world
    monkeypatch.setattr(web_search, "multi_search", lambda *a, **k: (_ for _ in ()).throw(web_search.SearchError("Tavily quota or rate limit reached.")))
    body = client.post("/rooms/r_bio101/ask", json={"question": "ما وظيفة الميتوكوندريا؟", "level": 2}, headers=SARA).json()
    assert_contract("RoomAskResponse", body)
    assert body["study_file"] is None and body["grounded"] and body["citations"] and body["error"] and "quota" in body["error"].lower()


def test_m2_being_down_is_retried_once_not_forever(world, monkeypatch):
    monkeypatch.setenv("TUTOR_SAMPLE_FALLBACK", "0")
    calls = []
    monkeypatch.setattr(agent, "tutor_answer", lambda *a, **k: calls.append(1) or {"answer": "", "citations": [], "grounded": False, "steps": [], "chunks": [], "source": "none"})
    st = agent.run_qa("r", "anything", 1, "u_sara")
    assert len(calls) == 1 + agent.MAX_RETRIES and st["offer_level2"] is True


def test_an_honest_nothing_found_from_m2_is_not_retried(world, monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "tutor_answer", lambda *a, **k: calls.append(1) or {"answer": "", "citations": [], "grounded": False, "steps": [], "chunks": [], "source": "m2"})
    agent.run_qa("r", "anything", 1, "u_sara")
    assert len(calls) == 1


def test_a_checker_crash_is_an_error_not_a_verdict(world, monkeypatch):
    monkeypatch.setattr(agent, "check_work_detail", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    st = agent.run_check("r", "2 + 2 = 5", "", "u_sara", "en")
    assert st["work"] is None and "check failed" in st["error"]

"""The coloured quiz and report PDFs (need fpdf2 and a Unicode font; skipped when the machine has none)."""
import pytest

from rafeeq.quiz.contracts import QuizItem

pytest.importorskip("fpdf")
pypdf = pytest.importorskip("pypdf")

from rafeeq.quiz.sheet import styled_pdf  # noqa: E402
from rafeeq.tutor.export import _font_files  # noqa: E402

try:
    _font_files()
except RuntimeError:
    pytest.skip("no Unicode font on this machine", allow_module_level=True)


def mk(kind, stem, **over):
    base = dict(item_id=f"i-{abs(hash(stem)) % 10_000}", topic_id="t", stem=stem, explanation="Because it is so.",
                file_name="book.pdf", page=3, kind=kind)
    if kind == "mcq":
        base.update(options=["first option", "second option", "third option", "fourth option"], right_choice=1)
    elif kind == "true_false":
        base.update(options=["True", "False"], right_choice=0)
    else:
        base.update(answer="2x + 3", rubric=["one", "two", "three"] if kind in ("essay", "problem") else [])
    base.update(over)
    return QuizItem(**base)


ITEMS = [mk("mcq", "Which option is right here?"), mk("true_false", "A constant has a zero derivative."),
         mk("short", "What is it called?"), mk("math", "Differentiate x^2 + 3x."),
         mk("problem", "A ball falls for two seconds, find its speed. " * 6), mk("essay", "Explain why. " * 30),
         mk("mcq", "ما هو تعريف المشتقة؟", options=["أ", "ب", "ج", "د"], right_choice=0)]


def text_of(data: bytes) -> str:
    import io
    return " ".join(page.extract_text() or "" for page in pypdf.PdfReader(io.BytesIO(data)).pages)


def test_the_quiz_pdf_has_every_question_and_option():
    data = styled_pdf.quiz_pdf("quiz-1", ITEMS)
    assert data[:5] == b"%PDF-"
    text = text_of(data)
    assert "Which option is right here?" in text and "second option" in text and "Essay" in text


def test_a_long_quiz_runs_onto_more_pages_and_an_empty_one_still_works():
    many = [mk("essay", f"Explain topic number {i}.") for i in range(12)]
    import io
    assert len(pypdf.PdfReader(io.BytesIO(styled_pdf.quiz_pdf("quiz-2", many))).pages) >= 3
    assert styled_pdf.quiz_pdf("quiz-3", [])[:5] == b"%PDF-"


def test_the_report_shows_the_score_the_answers_and_the_feedback():
    quiz = {"quiz_id": "quiz-1", "items": [it.model_dump() for it in ITEMS]}
    answers = [
        {"item_id": ITEMS[0].item_id, "choice": 1, "text": "", "score": None},               # right
        {"item_id": ITEMS[1].item_id, "choice": 1, "text": "", "score": None},               # wrong
        {"item_id": ITEMS[2].item_id, "choice": None, "text": "derivative", "score": 1.0, "feedback": "Good."},
        {"item_id": ITEMS[3].item_id, "choice": None, "text": "2x", "score": 0.0, "feedback": "Missing the 3."},
        {"item_id": ITEMS[4].item_id, "choice": None, "text": "v = g t", "score": 0.67, "feedback": "Partly there."},
        {"item_id": ITEMS[5].item_id, "choice": None, "text": "", "score": None},            # not graded
        {"item_id": ITEMS[6].item_id, "choice": None, "text": "", "score": None},
    ]
    data = styled_pdf.report_pdf(quiz, {"answers": answers})
    assert data[:5] == b"%PDF-"
    text = text_of(data)
    assert "Score report" in text and "Missing the 3." in text and "Partly there." in text
    assert "2 / 7" in text.replace("  ", " ")                                                # right + the 1.0 short answer

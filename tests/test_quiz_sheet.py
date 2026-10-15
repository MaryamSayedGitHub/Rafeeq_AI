import random

import numpy as np
import pytest

from rafeeq.quiz.sheet import grader, layout, synthetic


def photo(answers, seed=0, **kw):
    return synthetic.fake_photo(synthetic.draw_sheet(answers), np.random.default_rng(seed), **kw)


def test_reads_every_bubble_across_many_fake_photos():
    py = random.Random(7)
    for seed in range(12):
        n = py.choice([5, 17, 30])
        truth = [py.randrange(4) for _ in range(n)]
        marks = grader.detect_marks(photo(truth, seed, tilt=py.choice([0.5, 1.0, 1.5])), n)
        assert [m.choice for m in marks] == truth


def test_blank_and_double_marks_are_flagged_not_guessed():
    truth = [0, None, (1, 2), 3]
    marks = grader.detect_marks(photo(truth), len(truth))
    assert marks[0].choice == 0 and marks[3].choice == 3
    assert marks[1].choice is None and marks[2].choice is None
    assert marks[1].confidence == 0 and marks[2].confidence == 0


def test_a_light_pencil_mark_is_still_read():
    truth = [2, 0, 3]
    sheet = synthetic.draw_sheet(truth, ink=120)  # light gray ink
    data = synthetic.fake_photo(sheet, np.random.default_rng(3), shadow=0.2)
    assert [m.choice for m in grader.detect_marks(data, 3)] == truth


def test_bad_inputs_give_a_clear_error():
    with pytest.raises(grader.SheetError):
        grader.detect_marks(b"not an image", 5)
    import cv2

    ok, white = cv2.imencode(".jpg", np.full((900, 700), 255, np.uint8))
    with pytest.raises(grader.SheetError):
        grader.detect_marks(white.tobytes(), 5)


def test_question_count_is_checked():
    data = photo([0, 1])
    with pytest.raises(ValueError):
        grader.detect_marks(data, layout.MAX_ITEMS + 1)
    with pytest.raises(ValueError):
        grader.detect_marks(data, 0)


def test_printed_pdf_can_be_filled_photographed_and_graded(tmp_path):
    """End to end: real PDF -> image -> pen marks -> fake phone photo -> grader."""
    fitz = pytest.importorskip("pymupdf")
    import cv2

    from rafeeq.quiz.contracts import QuizItem
    from rafeeq.quiz.sheet.make_pdf import build_sheet_pdf

    items = [QuizItem(item_id=f"i{k}", topic_id="t", stem="A fairly long question " * 6 + str(k),
                      options=["first", "second", "third", "fourth"], right_choice=k % 4,
                      explanation="because", file_name="b.pdf", page=1) for k in range(12)]
    pdf = build_sheet_pdf("quiz-test", items, tmp_path / "s.pdf")

    doc = fitz.open(str(pdf))
    assert doc.page_count >= 2  # answer sheet + question pages
    dpi = 150
    pix = doc[0].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    page = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    k = dpi / 72
    answers = [(i * 3 + 1) % 4 for i in range(12)]
    for i, a in enumerate(answers):
        cx, cy = layout.bubble_center(i, a)
        cv2.circle(page, (int(cx * k), int(cy * k)), int(layout.BUBBLE_R * k * 0.8), 40, -1, cv2.LINE_AA)

    data = synthetic.fake_photo(page, np.random.default_rng(5))
    assert [m.choice for m in grader.detect_marks(data, 12)] == answers
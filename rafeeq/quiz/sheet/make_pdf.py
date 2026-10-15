"""Make the printable PDF: page 1 = answer sheet (bubbles), next pages = the questions.

Colour is used only where the grader does not look. The 4 corner squares and the
bubble rings stay pure black on white, so the sheet is read the same in a photo.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

from rafeeq.quiz.sheet import layout as L

LETTERS = "ABCD"
CHARS_PER_LINE = 85  # safe estimate for 10pt text across the page
LINE_H = 15.5  # PyMuPDF spaces 10pt helv lines ~14.8pt apart; a box too short draws nothing

INDIGO = (0.29, 0.24, 0.69)
SOFT = (0.78, 0.76, 0.93)
DARK = (0.12, 0.15, 0.20)
GRAY = (0.35, 0.38, 0.42)
WHITE = (1, 1, 1)
BLACK = (0, 0, 0)


def sheets_dir() -> Path:
    return Path(os.getenv("M4_SHEETS_DIR", "data/sheets_out"))


def sheet_path(quiz_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", quiz_id):
        raise ValueError("bad quiz id")
    return sheets_dir() / f"{quiz_id}.pdf"


def _fitz():
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz
    return fitz


def _block_height(text: str, chars_per_line: int = CHARS_PER_LINE) -> float:
    lines = sum(max(1, math.ceil(len(line) / chars_per_line)) for line in text.split("\n"))
    return lines * LINE_H + 4


def _put(page, fitz, x0: float, x1: float, y: float, text: str, size: float,
         font: str, color, chars_per_line: int = CHARS_PER_LINE) -> float:
    """Write a block of text and return the height it really used.

    insert_textbox draws NOTHING when the box is too short (it returns a negative
    number), so the box is grown until the text fits."""
    h = _block_height(text, chars_per_line)
    while True:
        rc = page.insert_textbox(fitz.Rect(x0, y, x1, y + h), text, fontsize=size,
                                 fontname=font, color=color)
        if rc >= 0:
            return h - rc
        h += LINE_H
        if y + h > L.PAGE_H:
            raise ValueError("a question is too long to print")


def build_sheet_pdf(quiz_id: str, items, out_path: Path | str, n: int | None = None) -> Path:
    """`items` = list of QuizItem (questions get printed). With items=None only the sheet is made, for `n` questions."""
    fitz = _fitz()
    count = len(items) if items is not None else n
    if not count or not 1 <= count <= L.MAX_ITEMS:
        raise ValueError(f"need 1..{L.MAX_ITEMS} questions")

    doc = fitz.open()
    page = doc.new_page(width=L.PAGE_W, height=L.PAGE_H)

    # corner squares: pure black, the grader finds the page with them
    half = L.MARKER_SIZE / 2
    for cx, cy in L.MARKER_CENTERS.values():
        page.draw_rect(fitz.Rect(cx - half, cy - half, cx + half, cy + half), color=BLACK, fill=BLACK)

    # header (between the corner squares, away from the bubbles)
    page.insert_text((100, 78), "Rafeeq", fontsize=24, fontname="hebo", color=INDIGO)
    page.insert_text((187, 78), "answer sheet", fontsize=14, fontname="helv", color=GRAY)
    page.insert_text((100, 96), f"Quiz: {quiz_id}", fontsize=9, fontname="helv", color=GRAY)
    page.insert_text((100, 112), "Name: ______________________", fontsize=10, fontname="helv", color=DARK)
    page.insert_text((385, 112), f"Score: ____ / {count}", fontsize=10, fontname="hebo", color=INDIGO)
    page.insert_text((100, 127), "Fill ONE circle per question completely with a dark pen or pencil.",
                     fontsize=8, fontname="helv", color=GRAY)
    page.draw_line(fitz.Point(100, 134), fitz.Point(L.PAGE_W - 100, 134), color=INDIGO, width=1.2)

    ncols = 1 if count <= L.ROWS_PER_COL else 2  # no A B C D header over an empty column
    for col in range(ncols):
        for j in range(4):
            page.insert_text((L.COL_X[col] + j * L.CHOICE_PITCH - 3.5, L.FIRST_ROW_Y - 22),
                             LETTERS[j], fontsize=11, fontname="hebo", color=INDIGO)
    for i in range(count):
        x0, y0 = L.bubble_center(i, 0)
        page.insert_text((x0 - 40, y0 + 4), str(i + 1), fontsize=11, fontname="hebo", color=INDIGO)
        for j in range(4):
            cx, cy = L.bubble_center(i, j)
            page.draw_circle(fitz.Point(cx, cy), L.BUBBLE_R, color=BLACK, width=1.0)

    page.insert_text((235, 826), "Rafeeq  -  learning copilot", fontsize=8, fontname="helv", color=GRAY)

    if items is not None:
        def new_questions_page():
            pg = doc.new_page(width=L.PAGE_W, height=L.PAGE_H)
            pg.draw_rect(fitz.Rect(0, 0, L.PAGE_W, 54), color=None, fill=INDIGO)
            pg.insert_text((50, 34), "Rafeeq", fontsize=18, fontname="hebo", color=WHITE)
            pg.insert_text((120, 34), "questions", fontsize=12, fontname="helv", color=WHITE)
            pg.insert_text((L.PAGE_W - 50 - 6 * len(quiz_id), 34), quiz_id, fontsize=9,
                           fontname="helv", color=SOFT)
            return pg

        qpage = new_questions_page()
        y = 76.0
        bottom = L.PAGE_H - 50
        for i, item in enumerate(items):
            stem = f"{i + 1}.  {item.stem}"
            opts = "\n".join(f"{LETTERS[j]})  {opt}" for j, opt in enumerate(item.options))
            if y + _block_height(stem, 80) + _block_height(opts, 82) + 14 > bottom:
                qpage = new_questions_page()
                y = 76.0
            top = y
            y += _put(qpage, fitz, 50, L.PAGE_W - 50, y, stem, 10.5, "hebo", DARK, 80) + 2
            y += _put(qpage, fitz, 68, L.PAGE_W - 50, y, opts, 10, "helv", GRAY, 82)
            qpage.draw_line(fitz.Point(40, top + 2), fitz.Point(40, y), color=INDIGO, width=2.5)
            y += 14

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make a blank answer sheet PDF (no questions) for testing")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--quiz-id", default="demo")
    ap.add_argument("--out", default="data/sheet_demo.pdf")
    a = ap.parse_args()
    print("wrote", build_sheet_pdf(a.quiz_id, None, a.out, n=a.n))
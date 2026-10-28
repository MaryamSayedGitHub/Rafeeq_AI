"""Make a score report PDF from a quiz and its grading result.

Usage (PowerShell):
    uv run python -m rafeeq.quiz.sheet.make_report quiz.json result.json --out data/report.pdf

quiz.json   = the reply of  POST /rooms/{id}/quizzes     (has "quiz_id" and "items")
result.json = the reply of  POST /quizzes/{id}/grade-sheet (has "answers")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rafeeq.quiz.sheet import layout as L

LETTERS = "ABCD"
INDIGO = (0.31, 0.27, 0.90)
GREY = (0.30, 0.30, 0.35)
GREEN = (0.10, 0.55, 0.30)
RED = (0.78, 0.20, 0.20)
AMBER = (0.80, 0.50, 0.05)
MARGIN = 50.0
LINE_H = 14.0


def _fitz():
    try:
        import pymupdf as fitz
    except ImportError:  # older PyMuPDF
        import fitz
    return fitz


def _load(path: str) -> dict:
    # utf-8-sig also reads files saved by Notepad with a BOM
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


PASS_MARK = 0.7   # same as rafeeq.quiz.quiz.grading.PASS_MARK


def _is_typed(item: dict) -> bool:
    return item.get("kind", "mcq") not in ("mcq", "true_false")


def _status(item: dict, choice, score: float | None = None) -> tuple[str, tuple]:
    if _is_typed(item):   # typed answers can earn part of the marks
        if score is None:
            return "Needs review (not graded)", AMBER
        if score >= PASS_MARK:
            return f"Correct ({round(score * 100)}%)", GREEN
        if score > 0:
            return f"Partly right ({round(score * 100)}%)", AMBER
        return "Wrong", RED
    if choice is None:
        return "Needs review (blank or two marks)", AMBER
    if choice == item["right_choice"]:
        return "Correct", GREEN
    return "Wrong", RED


def _is_right(item: dict, choice, score: float | None) -> bool:
    if _is_typed(item):
        return score is not None and score >= PASS_MARK
    return choice == item["right_choice"]


def build_report(quiz: dict, result: dict, out_path: str | Path) -> Path:
    """The coloured report (styled_pdf.report_pdf: every question type, Arabic too). If it cannot be made
    (no Unicode font, a layout bug) the plain Latin report below is written instead."""
    try:
        from rafeeq.quiz.sheet.styled_pdf import report_pdf

        data = report_pdf(quiz, result)
    except Exception:
        import logging

        logging.getLogger(__name__).warning("styled score report failed, writing the plain one", exc_info=True)
        return _build_plain_report(quiz, result, out_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def _build_plain_report(quiz: dict, result: dict, out_path: str | Path) -> Path:
    fitz = _fitz()
    items = quiz["items"]
    picked = {a["item_id"]: a.get("choice") for a in result["answers"]}
    typed_text = {a["item_id"]: a.get("text", "") for a in result["answers"]}
    typed_score = {a["item_id"]: a.get("score") for a in result["answers"]}
    right = sum(1 for it in items if _is_right(it, picked.get(it["item_id"]), typed_score.get(it["item_id"])))
    review = sum(1 for it in items if (typed_score.get(it["item_id"]) is None if _is_typed(it) else picked.get(it["item_id"]) is None))

    reg, bold = fitz.Font("helv"), fitz.Font("hebo")
    width = L.PAGE_W - 2 * MARGIN

    def wrap(text: str, size: float, max_w: float, is_bold: bool = False) -> list[str]:
        font = bold if is_bold else reg
        lines: list[str] = []
        cur = ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if cur and font.text_length(trial, fontsize=size) > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
        return lines

    doc = fitz.open()

    def new_page(first: bool = False):
        pg = doc.new_page(width=L.PAGE_W, height=L.PAGE_H)
        pg.draw_rect(fitz.Rect(0, 0, L.PAGE_W, 56), color=INDIGO, fill=INDIGO)
        pg.insert_text((MARGIN, 36), "Rafeeq score report", fontsize=15, fontname="hebo", color=(1, 1, 1))
        pg.insert_text((L.PAGE_W - MARGIN - 130, 36), quiz["quiz_id"], fontsize=10, fontname="helv", color=(1, 1, 1))
        return pg

    page = new_page(True)
    page.insert_text((MARGIN, 100), f"Score: {right} / {len(items)}", fontsize=26, fontname="hebo", color=INDIGO)
    pct = round(100 * right / len(items)) if items else 0
    page.insert_text((MARGIN, 122), f"{pct}% correct", fontsize=11, fontname="helv", color=GREY)
    if review:
        page.insert_text((MARGIN, 140), f"{review} question(s) need review: blank or marked twice, counted as not correct.",
                         fontsize=9, fontname="helv", color=AMBER)
    page.draw_line((MARGIN, 152), (L.PAGE_W - MARGIN, 152), color=INDIGO, width=1.2)
    y = 176.0

    for i, item in enumerate(items):
        choice = picked.get(item["item_id"])
        label, color = _status(item, choice, typed_score.get(item["item_id"]))
        stem = wrap(f"{i + 1}. {item['stem']}", 10.5, width - 12, True)
        if _is_typed(item):
            mine = typed_text.get(item["item_id"]) or "no answer"
            mine_l = wrap(f"Your answer: {mine}", 9.5, width - 12)[:6]
            ok_l = wrap(f"Model answer: {item.get('answer', '')}", 9.5, width - 12)[:6]
        else:
            mine = "no answer" if choice is None else f"{LETTERS[choice]}) {item['options'][choice]}"
            mine_l = wrap(f"Your answer: {mine}", 9.5, width - 12)
            ok_l = wrap(f"Correct answer: {LETTERS[item['right_choice']]}) {item['options'][item['right_choice']]}", 9.5, width - 12)
        why_l = wrap(f"Why: {item['explanation']}", 9, width - 12)
        src = f"Source: {item['file_name']}, page {item['page']}"
        n_lines = 1 + len(stem) + len(mine_l) + len(ok_l) + len(why_l) + 1
        h = n_lines * LINE_H + 12
        if y + h > L.PAGE_H - 40:
            page = new_page()
            y = 80.0
        page.draw_rect(fitz.Rect(MARGIN, y - 2, MARGIN + 3, y + h - 10), color=color, fill=color)
        ly = y + 10
        page.insert_text((MARGIN + 12, ly), label, fontsize=9, fontname="hebo", color=color)
        ly += LINE_H
        for line in stem:
            page.insert_text((MARGIN + 12, ly), line, fontsize=10.5, fontname="hebo")
            ly += LINE_H
        for line in mine_l:
            page.insert_text((MARGIN + 12, ly), line, fontsize=9.5, fontname="helv", color=color)
            ly += LINE_H
        for line in ok_l:
            page.insert_text((MARGIN + 12, ly), line, fontsize=9.5, fontname="helv", color=GREEN)
            ly += LINE_H
        for line in why_l:
            page.insert_text((MARGIN + 12, ly), line, fontsize=9, fontname="helv", color=GREY)
            ly += LINE_H
        page.insert_text((MARGIN + 12, ly), src, fontsize=8, fontname="helv", color=GREY)
        y += h + 6

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    doc.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make a score report PDF from a quiz and its grading result")
    ap.add_argument("quiz_json")
    ap.add_argument("result_json")
    ap.add_argument("--out", default="data/report.pdf")
    a = ap.parse_args()
    print("wrote", build_report(_load(a.quiz_json), _load(a.result_json), a.out))
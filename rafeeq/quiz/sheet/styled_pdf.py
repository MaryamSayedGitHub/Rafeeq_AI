"""Coloured, good-looking PDFs for the quiz and its score report (every question type, Arabic and English).

Written with fpdf2 and the Unicode font of the tutor's export (Arial on Windows), so Arabic text is shaped and
right-aligned. The answer sheet with bubbles (make_pdf.build_sheet_pdf) is NOT made here: the grader reads
black rings on white, so that page keeps its plain look.

    quiz_pdf(quiz_id, items)        -> bytes   the questions, with options and lines to write on
    report_pdf(quiz, result)        -> bytes   the score, then every question with the answer, the key and the feedback
"""
from __future__ import annotations

import re

PAGE_W, PAGE_H = 210.0, 297.0
MARGIN = 14.0
CONTENT_W = PAGE_W - 2 * MARGIN
TOP = 30.0               # below the header band
BOTTOM = PAGE_H - 16.0   # above the footer

INDIGO = (74, 61, 176)
INDIGO_SOFT = (238, 235, 252)
INK = (31, 41, 55)
MUTED = (100, 110, 125)
LINE = (203, 213, 225)
CARD_BORDER = (226, 232, 240)
PAGE_TINT = (248, 247, 252)
GREEN = (22, 163, 74)
GREEN_SOFT = (232, 247, 237)
AMBER = (217, 119, 6)
AMBER_SOFT = (254, 243, 224)
RED = (220, 38, 38)
RED_SOFT = (253, 236, 236)
WHITE = (255, 255, 255)

# colour and names of each question type
KINDS = {
    "mcq": ((74, 61, 176), "Multiple choice", "اختيار من متعدد"),
    "true_false": ((13, 148, 136), "True / False", "صح أو خطأ"),
    "short": ((217, 119, 6), "Short answer", "إجابة قصيرة"),
    "math": ((22, 163, 74), "Maths", "رياضيات"),
    "problem": ((234, 88, 12), "Problem", "مسألة"),
    "essay": ((219, 39, 119), "Essay", "مقالي"),
}
WRITING_LINES = {"short": 1, "math": 4, "problem": 9, "essay": 13}
OPTION_LETTERS = "ABCDEFGH"

TEXT = {
    "en": {"quiz": "Quiz", "report": "Score report", "name": "Name", "date": "Date", "score": "Score",
           "correct": "Correct", "partly": "Partly right", "wrong": "Wrong", "review": "Needs review",
           "yours": "Your answer", "key": "Correct answer", "model": "Model answer", "feedback": "Feedback",
           "why": "Why", "source": "Source", "page": "Page", "questions": "questions", "no_answer": "(no answer)",
           "by_type": "By question type"},
    "ar": {"quiz": "اختبار", "report": "تقرير النتيجة", "name": "الاسم", "date": "التاريخ", "score": "الدرجة",
           "correct": "صحيحة", "partly": "صحيحة جزئيًا", "wrong": "خاطئة", "review": "تحتاج مراجعة",
           "yours": "إجابتك", "key": "الإجابة الصحيحة", "model": "الإجابة النموذجية", "feedback": "ملاحظات",
           "why": "التفسير", "source": "المصدر", "page": "صفحة", "questions": "أسئلة", "no_answer": "(بدون إجابة)",
           "by_type": "حسب نوع السؤال"},
}
PASS_MARK = 0.7   # same as rafeeq.quiz.quiz.grading.PASS_MARK


def _is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("؀" <= c <= "ۿ" for c in letters) > len(letters) / 2


def _clean(text: str) -> str:
    """Characters the PDF font may not have, and runs of spaces."""
    text = (text or "").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _kind_meta(kind: str, lang: str) -> tuple[tuple, str]:
    color, en, ar = KINDS.get(kind, KINDS["mcq"])
    return color, (ar if lang == "ar" else en)


def _make_pdf(title: str, right_text: str, lang: str):
    from fpdf import FPDF
    from rafeeq.tutor.export import _font_files

    fonts = _font_files()

    class Doc(FPDF):
        def header(self):                                    # a coloured band on every page
            self.set_fill_color(*INDIGO)
            self.rect(0, 0, PAGE_W, 20, style="F")
            self.set_fill_color(*(255, 196, 61))
            self.rect(0, 20, PAGE_W, 1.6, style="F")         # a thin gold line under the band
            self.set_text_shaping(True, direction="ltr")
            self.set_text_color(*WHITE)
            self.set_font("U", "B", 15)
            self.set_xy(MARGIN, 5)
            self.cell(80, 10, "Rafeeq  رفيق")
            self.set_font("U", "", 10)
            self.set_xy(PAGE_W - MARGIN - 100, 6)
            self.cell(100, 8, right_text, align="R")

        def footer(self):
            self.set_text_shaping(True, direction="ltr")
            self.set_y(-11)
            self.set_font("U", "", 8)
            self.set_text_color(*MUTED)
            self.cell(0, 6, f"Rafeeq  |  {title}  |  {self.page_no()} / {{nb}}", align="C")

    pdf = Doc(format="A4")
    pdf.set_auto_page_break(False)
    pdf.set_margins(MARGIN, TOP, MARGIN)
    for style, path in fonts.items():
        pdf.add_font("U", style, str(path))
    for style in ("B", "I", "BI"):
        if style not in fonts:
            pdf.add_font("U", style, str(fonts[""]))
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_y(TOP)
    return pdf


class _Painter:
    """Small drawing helpers on top of an FPDF document."""

    def __init__(self, pdf, lang: str):
        self.pdf = pdf
        self.lang = lang
        self.t = TEXT[lang]

    # ---- text
    def _font(self, size: float, bold: bool = False, color=INK):
        self.pdf.set_font("U", "B" if bold else "", size)
        self.pdf.set_text_color(*color)

    def _shape(self, text: str) -> bool:
        rtl = _is_arabic(text)
        self.pdf.set_text_shaping(True, direction="rtl" if rtl else "ltr")
        return rtl

    def lines(self, text: str, w: float, size: float, bold: bool = False) -> int:
        from fpdf.enums import MethodReturnValue

        self._font(size, bold)
        self._shape(text)
        out = self.pdf.multi_cell(w, 1, text or " ", dry_run=True, output=MethodReturnValue.LINES)
        return max(1, len(out))

    def height(self, text: str, w: float, size: float, line_h: float, bold: bool = False) -> float:
        return self.lines(text, w, size, bold) * line_h

    def write(self, x: float, y: float, w: float, text: str, size: float, line_h: float, bold: bool = False,
              color=INK) -> float:
        """Write wrapped text at (x, y); Arabic is shaped and right-aligned. Returns its height."""
        from fpdf.enums import XPos, YPos

        self._font(size, bold, color)
        rtl = self._shape(text)
        h = self.height(text, w, size, line_h, bold)
        self._font(size, bold, color)
        self._shape(text)
        self.pdf.set_xy(x, y)
        self.pdf.multi_cell(w, line_h, text or " ", align="R" if rtl else "L", new_x=XPos.LEFT, new_y=YPos.NEXT)
        return h

    # ---- shapes
    def box(self, x, y, w, h, fill, border=None, radius=2.5):
        self.pdf.set_fill_color(*fill)
        if border:
            self.pdf.set_draw_color(*border)
            self.pdf.set_line_width(0.25)
            style = "DF"
        else:
            style = "F"
        self.pdf.rect(x, y, w, h, style=style, round_corners=True, corner_radius=radius)

    def pill(self, x, y, text, color, size=8.5, rtl_right_edge=False) -> float:
        """A small coloured label. Returns its width. With rtl_right_edge, x is the right edge."""
        self._font(size, True, WHITE)
        self.pdf.set_text_shaping(True, direction="ltr" if not _is_arabic(text) else "rtl")
        w = self.pdf.get_string_width(text) + 7
        left = x - w if rtl_right_edge else x
        self.box(left, y, w, 6, color, radius=3)
        self.pdf.set_xy(left, y)
        self._font(size, True, WHITE)
        self.pdf.cell(w, 6, text, align="C")
        return w

    def number_badge(self, x, y, n: int, color):
        self.pdf.set_fill_color(*color)
        self.pdf.ellipse(x, y, 8, 8, style="F")
        self._font(10, True, WHITE)
        self.pdf.set_text_shaping(True, direction="ltr")
        self.pdf.set_xy(x, y + 0.6)
        self.pdf.cell(8, 6.8, str(n), align="C")

    def stripe(self, x, y, h, color):
        self.pdf.set_fill_color(*color)
        self.pdf.rect(x, y, 2.4, h, style="F")

    def ensure(self, y: float, need: float) -> float:
        """Start a new page when `need` does not fit under y; returns the y to draw at."""
        if y + need <= BOTTOM:
            return y
        self.pdf.add_page()
        return TOP


# ======================================================================================= the quiz
def _quiz_card(p: _Painter, y: float, n: int, item) -> float:
    pdf = p.pdf
    color, kind_name = _kind_meta(item.kind, p.lang)
    pad = 5.0
    inner_x = MARGIN + pad + 2.4
    inner_w = CONTENT_W - 2 * pad - 2.4
    opt_x, opt_w = inner_x + 10, inner_w - 10

    stem_h = p.height(item.stem, inner_w, 11.5, 5.6, bold=True)
    opts = list(item.options)
    opt_h = [max(7.0, p.height(o, opt_w, 10.5, 5.2) + 2.5) for o in opts]
    write_lines = WRITING_LINES.get(item.kind, 0)
    card_h = (pad + 10 + stem_h + 2 + (sum(opt_h) + 1 if opts else 0)
              + (write_lines * 8.2 + 3 if write_lines else 0) + pad)

    y = p.ensure(y, min(card_h, BOTTOM - TOP))
    p.box(MARGIN, y, CONTENT_W, card_h, WHITE, CARD_BORDER)
    p.stripe(MARGIN, y + 1, card_h - 2, color)

    # the top row: number on the left, the question type on the right
    p.number_badge(inner_x, y + pad, n, color)
    p.pill(MARGIN + CONTENT_W - pad, y + pad + 1, kind_name, color, rtl_right_edge=True)

    cy = y + pad + 10
    p.write(inner_x, cy, inner_w, item.stem, 11.5, 5.6, bold=True)
    cy += stem_h + 2

    for i, (opt, h) in enumerate(zip(opts, opt_h)):
        letter = OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else str(i + 1)
        pdf.set_draw_color(*color)
        pdf.set_line_width(0.5)
        pdf.ellipse(inner_x + 1, cy + 0.6, 6, 6, style="D")
        p._font(8.5, True, color)
        pdf.set_text_shaping(True, direction="ltr")
        pdf.set_xy(inner_x + 1, cy + 1.3)
        pdf.cell(6, 4.8, letter, align="C")
        p.write(opt_x, cy + 0.9, opt_w, opt, 10.5, 5.2)
        cy += h

    if write_lines:
        pdf.set_draw_color(*LINE)
        pdf.set_line_width(0.25)
        ly = cy + 6
        for _ in range(write_lines):
            pdf.line(inner_x, ly, inner_x + inner_w, ly)
            ly += 8.2
    return y + card_h + 5


def quiz_pdf(quiz_id: str, items) -> bytes:
    lang = "ar" if items and sum(_is_arabic(it.stem) for it in items) > len(items) / 2 else "en"
    pdf = _make_pdf(TEXT[lang]["quiz"], quiz_id, lang)
    p = _Painter(pdf, lang)
    t = p.t

    # the name / date strip and a summary of the question types
    y = TOP
    p.box(MARGIN, y, CONTENT_W, 22, INDIGO_SOFT, None, 3)
    p.write(MARGIN + 5, y + 3.5, 90, f"{t['name']}: ................................", 10.5, 6, color=INK)
    p.write(MARGIN + 105, y + 3.5, 70, f"{t['date']}: ..................", 10.5, 6, color=INK)
    counts: dict[str, int] = {}
    for it in items:
        counts[it.kind] = counts.get(it.kind, 0) + 1
    cx = MARGIN + 5
    p.write(cx, y + 12.8, 40, f"{len(items)} {t['questions']}", 9.5, 5, bold=True, color=INDIGO)
    cx += 34
    for kind, n in counts.items():
        color, name = _kind_meta(kind, lang)
        cx += p.pill(cx, y + 13.2, f"{name}  {n}", color, size=8) + 3
    y += 28

    for n, item in enumerate(items, 1):
        y = _quiz_card(p, y, n, item)
    return bytes(pdf.output())


# ======================================================================================= the report
def _status(item: dict, choice, score):
    """(label key, colour, soft colour, is_right)."""
    if item.get("kind", "mcq") in ("mcq", "true_false"):
        if choice is None:
            return "review", AMBER, AMBER_SOFT, False
        ok = choice == item["right_choice"]
        return ("correct", GREEN, GREEN_SOFT, True) if ok else ("wrong", RED, RED_SOFT, False)
    if score is None:
        return "review", AMBER, AMBER_SOFT, False
    if score >= PASS_MARK:
        return "correct", GREEN, GREEN_SOFT, True
    if score > 0:
        return "partly", AMBER, AMBER_SOFT, False
    return "wrong", RED, RED_SOFT, False


def _option_text(item: dict, index) -> str:
    if index is None or not 0 <= index < len(item.get("options", [])):
        return ""
    letter = OPTION_LETTERS[index] if index < len(OPTION_LETTERS) else str(index + 1)
    return f"{letter}) {item['options'][index]}"


def _report_card(p: _Painter, y: float, n: int, item: dict, ans: dict) -> float:
    pdf = p.pdf
    t = p.t
    typed = item.get("kind", "mcq") not in ("mcq", "true_false")
    choice, score = ans.get("choice"), ans.get("score")
    key, color, soft, _ = _status(item, choice, score)
    label = t[key] + (f"  {round(score * 100)}%" if typed and score is not None and key in ("correct", "partly") else "")
    pad = 5.0
    inner_x = MARGIN + pad + 2.4
    inner_w = CONTENT_W - 2 * pad - 2.4

    mine = (_clean(ans.get("text", ""))[:900] if typed else _option_text(item, choice)) or t["no_answer"]
    right = _clean(item.get("answer", ""))[:900] if typed else _option_text(item, item.get("right_choice", 0))
    rows = [(t["yours"], mine, soft, color)]
    if key != "correct":
        rows.append((t["model"] if typed else t["key"], right, GREEN_SOFT, GREEN))
    elif typed and right:
        rows.append((t["model"], right, GREEN_SOFT, GREEN))
    if typed and ans.get("feedback"):
        rows.append((t["feedback"], _clean(ans["feedback"]), AMBER_SOFT, AMBER))
    if item.get("explanation"):
        rows.append((t["why"], _clean(item["explanation"]), PAGE_TINT, MUTED))

    stem_h = p.height(item["stem"], inner_w - 10, 11, 5.4, bold=True)
    row_h = [4.6 + p.height(text, inner_w - 10, 9.5, 4.7) + 4 for _, text, _, _ in rows]
    src = f"{t['source']}: {item.get('file_name', '')}, {t['page']} {item.get('page', '')}"
    card_h = pad + 8 + 3 + stem_h + 3 + sum(h + 2.5 for h in row_h) + 5 + pad - 1

    y = p.ensure(y, min(card_h, BOTTOM - TOP))
    p.box(MARGIN, y, CONTENT_W, card_h, WHITE, CARD_BORDER)
    p.stripe(MARGIN, y + 1, card_h - 2, color)
    p.number_badge(inner_x, y + pad, n, color)
    kcolor, kname = _kind_meta(item.get("kind", "mcq"), p.lang)
    w1 = p.pill(MARGIN + CONTENT_W - pad, y + pad + 1, label, color, rtl_right_edge=True)
    p.pill(MARGIN + CONTENT_W - pad - w1 - 3, y + pad + 1, kname, kcolor, rtl_right_edge=True)

    cy = y + pad + 8 + 3
    cy += p.write(inner_x, cy, inner_w, item["stem"], 11, 5.4, bold=True) + 3
    for (title, text, fill, accent), h in zip(rows, row_h):
        p.box(inner_x, cy, inner_w, h, fill, None, 2)
        p.pdf.set_fill_color(*accent)
        p.pdf.rect(inner_x, cy + 1, 1.6, h - 2, style="F")
        p.write(inner_x + 4, cy + 1.6, inner_w - 8, title.upper() if p.lang == "en" else title, 7.5, 3.6, bold=True,
                color=accent)
        p.write(inner_x + 4, cy + 5.6, inner_w - 10, text, 9.5, 4.7)
        cy += h + 2.5
    p.write(inner_x, cy + 0.5, inner_w, src, 7.5, 3.6, color=MUTED)
    return y + card_h + 5


def _bar_color(share: float):
    return GREEN if share >= PASS_MARK else AMBER if share >= 0.4 else RED


def report_pdf(quiz: dict, result: dict) -> bytes:
    items = quiz["items"]
    lang = "ar" if items and sum(_is_arabic(it["stem"]) for it in items) > len(items) / 2 else "en"
    pdf = _make_pdf(TEXT[lang]["report"], quiz["quiz_id"], lang)
    p = _Painter(pdf, lang)
    t = p.t
    answers = {a["item_id"]: a for a in result["answers"]}

    marks, by_kind, review = [], {}, 0
    for it in items:
        ans = answers.get(it["item_id"], {})
        key, _, _, ok = _status(it, ans.get("choice"), ans.get("score"))
        typed = it.get("kind", "mcq") not in ("mcq", "true_false")
        mark = ans.get("score") if typed else (1.0 if ok else 0.0)
        marks.append(mark or 0.0)
        review += key == "review"
        k = by_kind.setdefault(it.get("kind", "mcq"), [0, 0.0])
        k[0] += 1
        k[1] += mark or 0.0
    total = len(items)
    right = sum(1 for it in items if _status(it, answers.get(it["item_id"], {}).get("choice"),
                                              answers.get(it["item_id"], {}).get("score"))[3])
    share = sum(marks) / total if total else 0.0

    # ---- the score banner
    y = TOP
    p.box(MARGIN, y, CONTENT_W, 40, INDIGO, None, 4)
    pdf.set_text_shaping(True, direction="ltr")
    p._font(34, True, WHITE)
    pdf.set_xy(MARGIN + 8, y + 5)
    pdf.cell(70, 16, f"{right} / {total}")
    p._font(11, False, (221, 214, 254))
    pdf.set_xy(MARGIN + 8, y + 21)
    pdf.cell(70, 6, f"{t['score']}: {round(share * 100)}%")
    # the progress bar
    bx, bw = MARGIN + 84, CONTENT_W - 84 - 8
    p.box(bx, y + 11, bw, 7, (111, 99, 205), None, 3.5)
    if share > 0:
        p.box(bx, y + 11, max(7.0, bw * share), 7, _bar_color(share), None, 3.5)
    # one chip per question type: correct share
    cx = bx
    for kind, (count, got) in by_kind.items():
        color, name = _kind_meta(kind, lang)
        cx += p.pill(cx, y + 24, f"{name}  {round(got)}/{count}", color, size=7.5) + 2.5
        if cx > MARGIN + CONTENT_W - 30:
            break
    if review:
        p._font(8.5, True, (255, 214, 140))
        pdf.set_text_shaping(True, direction="rtl" if lang == "ar" else "ltr")
        pdf.set_xy(bx, y + 33)
        pdf.cell(bw, 5, f"{review} {t['review']}", align="R" if lang == "ar" else "L")
    y += 47

    for n, it in enumerate(items, 1):
        y = _report_card(p, y, n, it, answers.get(it["item_id"], {}))
    return bytes(pdf.output())

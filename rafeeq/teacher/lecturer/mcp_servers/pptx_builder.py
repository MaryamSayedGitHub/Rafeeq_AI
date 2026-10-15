"""Builds the session .pptx with python-pptx. Plain functions, no MCP here, so it is easy to test.

Every slide is drawn from shapes on a blank 16:9 page (no dependency on a template file), so
the design is the same on every computer. pptx_server.py exposes build_presentation() as an
MCP tool.

Slide order:  cover -> content slides -> code examples -> quiz questions -> last content slide
Each content slide can carry a small flow diagram (`diagram_steps`) drawn next to the bullets.
The lecturer's name, the course and the slide number are in the footer of every slide.
"""

import re
import threading
from contextlib import contextmanager
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ------------------------------------------------------------------ design
W, H = 13.333, 7.5          # inches, 16:9
MARGIN = 0.7
INK = RGBColor(0x1B, 0x25, 0x40)
INK_SOFT = RGBColor(0x4A, 0x55, 0x72)
PAPER = RGBColor(0xF3, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD5, 0xDA, 0xE5)
BLUE = RGBColor(0x3A, 0x5B, 0xD9)
BLUE_SOFT = RGBColor(0xE4, 0xEA, 0xFB)
YELLOW = RGBColor(0xFF, 0xE0, 0x66)
GREEN = RGBColor(0x1F, 0x7A, 0x4D)
GREEN_SOFT = RGBColor(0xDD, 0xF3, 0xE7)
CODE_BG = RGBColor(0x12, 0x18, 0x2B)
CODE_TEXT = RGBColor(0xE6, 0xEA, 0xF5)

FONT = "Segoe UI"       # on every Windows PC; other systems substitute a similar sans font
FONT_CODE = "Consolas"

MAX_QUIZ_SLIDES = 3     # each question takes 2 slides (question, then answer)
MAX_DIAGRAM_STEPS = 5
MAX_CODE_LINES = 22

_ARABIC = re.compile(r"[؀-ۿ]")


# ------------------------------------------------------------------ themes
# The palette is module-level so the helpers above stay simple. build_presentation() swaps it
# for the deck's theme under a lock and puts the default back afterwards.
_THEME_LOCK = threading.Lock()
_THEME_GLOBALS = {"ink": "INK", "ink_soft": "INK_SOFT", "paper": "PAPER", "line": "LINE", "blue": "BLUE",
                  "blue_soft": "BLUE_SOFT", "yellow": "YELLOW", "green": "GREEN", "green_soft": "GREEN_SOFT",
                  "code_bg": "CODE_BG", "code_text": "CODE_TEXT"}


@contextmanager
def _themed(theme: dict | None):
    if not theme:
        yield
        return
    from rafeeq.teacher.themes import normalize

    theme = normalize(theme)
    with _THEME_LOCK:
        names = list(_THEME_GLOBALS.values()) + ["FONT"]
        saved = {name: globals()[name] for name in names}
        try:
            for key, name in _THEME_GLOBALS.items():
                globals()[name] = RGBColor.from_string(theme["colors"][key][1:])
            globals()["FONT"] = theme["font"]
            yield
        finally:
            globals().update(saved)


def _picture(slide, path, x: float, y: float, w: float, max_h: float) -> bool:
    """Put an image in the box, keeping its proportions. A missing or damaged file is skipped."""
    try:
        pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    except Exception:
        return False
    limit = Inches(max_h)
    if pic.height > limit:
        scale = limit / pic.height
        pic.width, pic.height = int(pic.width * scale), limit
        pic.left = Inches(x) + int((Inches(w) - pic.width) / 2)
    return True


# ----------------------------------------------------------------- helpers
def _rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    box = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line
        box.line.width = Pt(1)
    box.shadow.inherit = False
    return box


def _style(paragraph, text, size, color, bold=False, font=None, align=None):
    """Write `text` into a paragraph. Arabic text is right-aligned and marked right-to-left."""
    text = str(text)
    font = font or FONT
    is_arabic = bool(_ARABIC.search(text)) and font != FONT_CODE
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = font
    run.font.color.rgb = color
    paragraph.alignment = align if align is not None else (PP_ALIGN.RIGHT if is_arabic else PP_ALIGN.LEFT)
    if is_arabic:
        paragraph._p.get_or_add_pPr().set("rtl", "1")


def _text(slide, x, y, w, h, text, size, color, bold=False, font=None, align=None,
          anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    _style(frame.paragraphs[0], text, size, color, bold, font, align)
    return box


def _label(shape, text, size, color, bold=False, align=PP_ALIGN.CENTER):
    """Text inside a drawn shape (box, card)."""
    frame = shape.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.15)
    _style(frame.paragraphs[0], text, size, color, bold, align=align)


def _new_slide(prs, background=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # 6 = blank
    _rect(slide, 0, 0, W, H, background or PAPER)
    return slide


def _footer(slide, meta: dict, number: int, total: int, dark: bool = False) -> None:
    color = RGBColor(0xB8, 0xC1, 0xD9) if dark else INK_SOFT
    if not dark:
        _rect(slide, MARGIN, H - 0.72, W - 2 * MARGIN, 0.015, LINE)
    who = " · ".join(part for part in (meta.get("lecturer"), meta.get("course_name")) if part)
    _text(slide, MARGIN, H - 0.6, 9, 0.35, who, 12, color, align=PP_ALIGN.LEFT)
    _text(slide, W - MARGIN - 2, H - 0.6, 2, 0.35, f"{number} / {total}", 12, color, align=PP_ALIGN.RIGHT)


def _heading(slide, title: str, kicker: str = "") -> None:
    """Accent bar + optional small label + slide title."""
    _rect(slide, MARGIN, 0.62, 0.12, 0.95, BLUE)
    if kicker:
        _text(slide, MARGIN + 0.32, 0.55, W - 2 * MARGIN - 0.4, 0.3, kicker.upper(), 12, BLUE, bold=True)
    size = 30 if len(title) <= 48 else 24
    _text(slide, MARGIN + 0.32, 0.85 if kicker else 0.62, W - 2 * MARGIN - 0.4, 0.95, title, size, INK,
          bold=True, anchor=MSO_ANCHOR.MIDDLE if not kicker else MSO_ANCHOR.TOP)


def _notes(slide, text) -> None:
    text = str(text or "").strip()
    if text:
        slide.notes_slide.notes_text_frame.text = text


# ------------------------------------------------------------- slide types
def _cover(prs, meta: dict, number: int, total: int) -> None:
    slide = _new_slide(prs, INK)
    _rect(slide, 0, 0, 0.35, H, BLUE)
    _rect(slide, MARGIN + 0.3, 2.35, 1.1, 0.09, YELLOW)
    if meta.get("course_name"):
        _text(slide, MARGIN + 0.3, 1.75, 11, 0.45, meta["course_name"].upper(), 16,
              RGBColor(0xB8, 0xC1, 0xD9), bold=True)
    title = meta["title"]
    _text(slide, MARGIN + 0.3, 2.7, 11.3, 2.4, title, 48 if len(title) <= 40 else 36, WHITE, bold=True)
    if meta.get("lecturer"):
        _text(slide, MARGIN + 0.3, 5.35, 11, 0.5, meta["lecturer"], 22, WHITE)
    _footer(slide, {}, number, total, dark=True)


def _bullets(slide, bullets: list[str], x: float, y: float, w: float) -> None:
    size = 24 if len(bullets) <= 4 else 20
    step = min(0.95, 4.3 / max(len(bullets), 1))
    for i, bullet in enumerate(bullets):
        top = y + i * step
        arabic = bool(_ARABIC.search(str(bullet)))
        marker_x = x + w - 0.16 if arabic else x
        text_x = x if arabic else x + 0.4
        _rect(slide, marker_x, top + 0.14, 0.16, 0.16, BLUE)
        _text(slide, text_x, top, w - 0.4, step, bullet, size, INK)


def _diagram(slide, steps: list[str], x: float, y: float, w: float) -> None:
    """A vertical flow: box, arrow, box... Drawn from shapes, so it stays editable in PowerPoint."""
    steps = [str(s) for s in steps[:MAX_DIAGRAM_STEPS]]
    gap = 0.3
    box_h = min(0.85, (4.5 - gap * (len(steps) - 1)) / len(steps))
    for i, step in enumerate(steps):
        top = y + i * (box_h + gap)
        last = i == len(steps) - 1
        box = _rect(slide, x, top, w, box_h, BLUE if last else WHITE, None if last else BLUE,
                    MSO_SHAPE.ROUNDED_RECTANGLE)
        _label(box, step, 16 if len(step) <= 28 else 13, WHITE if last else INK, bold=True)
        if not last:
            _rect(slide, x + w / 2 - 0.11, top + box_h + 0.04, 0.22, gap - 0.08, BLUE,
                  shape=MSO_SHAPE.DOWN_ARROW)


def _content(prs, item: dict, meta: dict, number: int, total: int) -> None:
    slide = _new_slide(prs)
    _heading(slide, str(item.get("title", "")))
    bullets = [str(b) for b in item.get("bullets", [])]
    steps = [s for s in item.get("diagram_steps") or [] if str(s).strip()]
    body_y, full_w = 2.05, W - 2 * MARGIN
    image = item.get("image_path")
    if image and Path(str(image)).is_file() and _picture(  # a picture takes the place of the small diagram
            slide, image, MARGIN + full_w * 0.62, body_y - 0.05, full_w * 0.38, 4.5):
        _bullets(slide, bullets, MARGIN, body_y, full_w * 0.56)
    elif len(steps) >= 2:
        _bullets(slide, bullets, MARGIN, body_y, full_w * 0.56)
        _diagram(slide, steps, MARGIN + full_w * 0.62, body_y - 0.05, full_w * 0.38)
    else:
        _bullets(slide, bullets, MARGIN, body_y, full_w)
    _footer(slide, meta, number, total)
    _notes(slide, item.get("notes"))


def _code(prs, example: dict, meta: dict, number: int, total: int) -> None:
    slide = _new_slide(prs)
    _heading(slide, str(example.get("title", "")), "Code example")

    lines = str(example.get("code", "")).rstrip().splitlines() or [""]
    if len(lines) > MAX_CODE_LINES:
        lines = lines[:MAX_CODE_LINES - 1] + ["# ... (full code is in the examples file)"]
    output = str(example.get("output", "")).strip()
    full_w = W - 2 * MARGIN
    code_w = full_w * 0.64 if output else full_w
    panel_y, panel_h = 2.0, 4.55
    size = 16 if len(lines) <= 14 else 13 if len(lines) <= 18 else 11

    _rect(slide, MARGIN, panel_y, code_w, panel_h, CODE_BG, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.03
    box = slide.shapes.add_textbox(Inches(MARGIN + 0.25), Inches(panel_y + 0.2),
                                   Inches(code_w - 0.5), Inches(panel_h - 0.4))
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        _style(paragraph, line, size, CODE_TEXT, font=FONT_CODE, align=PP_ALIGN.LEFT)

    if output:
        out_x = MARGIN + code_w + 0.3
        out_w = full_w - code_w - 0.3
        _rect(slide, out_x, panel_y, out_w, panel_h, WHITE, LINE, MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.03
        _text(slide, out_x + 0.25, panel_y + 0.2, out_w - 0.5, 0.3, "OUTPUT", 12, GREEN, bold=True)
        out_lines = output.splitlines()[:12]
        out = slide.shapes.add_textbox(Inches(out_x + 0.25), Inches(panel_y + 0.6),
                                       Inches(out_w - 0.5), Inches(panel_h - 0.8))
        out.text_frame.word_wrap = True
        for i, line in enumerate(out_lines):
            paragraph = out.text_frame.paragraphs[0] if i == 0 else out.text_frame.add_paragraph()
            _style(paragraph, line, 13, INK, font=FONT_CODE, align=PP_ALIGN.LEFT)

    _footer(slide, meta, number, total)
    _notes(slide, example.get("explanation"))


def _question(prs, q: dict, index: int, reveal: bool, meta: dict, number: int, total: int) -> None:
    """Two slides per question: first the question, then the same slide with the answer marked."""
    slide = _new_slide(prs)
    _heading(slide, str(q["question"]), f"Answer {index}" if reveal else f"Check your understanding {index}")
    full_w = W - 2 * MARGIN
    card_w, card_h, gap = (full_w - 0.3) / 2, 1.15, 0.3
    for i, option in enumerate(q["options"][:4]):
        x = MARGIN + (i % 2) * (card_w + gap)
        y = 2.25 + (i // 2) * (card_h + gap)
        correct = reveal and i == q["answer_index"]
        card = _rect(slide, x, y, card_w, card_h, GREEN_SOFT if correct else WHITE,
                     GREEN if correct else LINE, MSO_SHAPE.ROUNDED_RECTANGLE)
        card.adjustments[0] = 0.12
        _label(card, f"{'ABCD'[i]}.  {option}", 18 if len(str(option)) <= 60 else 14,
               GREEN if correct else INK, bold=correct, align=PP_ALIGN.LEFT)
    if reveal and q.get("explanation"):
        _text(slide, MARGIN, 5.35, full_w, 0.9, str(q["explanation"]), 16, INK_SOFT)
    _footer(slide, meta, number, total)
    if not reveal:
        _notes(slide, "Give students a minute to answer before showing the next slide.")


# -------------------------------------------------------------------- main
def _build_presentation(output_path: str | Path, deck: dict) -> Path:
    """Create the file and return its path.

    deck = {
      "title": str, "course_name": str, "lecturer": str,
      "slides": [{"title": str, "bullets": [str], "notes": str, "diagram_steps": [str]}],
      "code_examples": [{"title", "code", "explanation", "output"}],        # optional
      "quiz": [{"question", "options", "answer_index", "explanation"}],     # optional
    }
    """
    output_path = Path(output_path)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    meta = {
        "title": str(deck.get("title", "")),
        "course_name": str(deck.get("course_name") or ""),
        "lecturer": str(deck.get("lecturer") or ""),
    }
    slides = list(deck.get("slides") or [])
    code = list(deck.get("code_examples") or [])
    quiz = list(deck.get("quiz") or [])[:MAX_QUIZ_SLIDES]

    # The summary (last content slide) closes the deck, after the code and the questions.
    body, closing = (slides[:-1], slides[-1:]) if len(slides) > 1 and (code or quiz) else (slides, [])

    plan = [("cover", None)]
    plan += [("content", s) for s in body]
    plan += [("code", c) for c in code]
    for i, q in enumerate(quiz, 1):
        plan += [("question", (q, i, False)), ("question", (q, i, True))]
    plan += [("content", s) for s in closing]

    total = len(plan)
    for number, (kind, data) in enumerate(plan, 1):
        if kind == "cover":
            _cover(prs, meta, number, total)
        elif kind == "content":
            _content(prs, data, meta, number, total)
        elif kind == "code":
            _code(prs, data, meta, number, total)
        else:
            q, index, reveal = data
            _question(prs, q, index, reveal, meta, number, total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output_path))
    return output_path


def build_presentation(output_path: str | Path, deck: dict) -> Path:
    """Create the file and return its path. `deck` may also carry "theme" (see rafeeq/teacher/themes.py)
    and, on a content slide, "image_path" (a picture file shown next to the bullets)."""
    with _themed(deck.get("theme")):
        return _build_presentation(output_path, deck)

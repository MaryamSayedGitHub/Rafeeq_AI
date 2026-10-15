"""Study file -> Word (.docx) or PDF, built from the saved Markdown (headings, bullets, numbered steps, **bold**, links).

Arabic is supported: RTL paragraphs in Word, text shaping + bidi in the PDF (needs a Unicode font: Arial/Tahoma on Windows,
DejaVu on Linux; set PDF_FONT_DIR to use another folder with arial.ttf / arialbd.ttf / ariali.ttf / arialbi.ttf).
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path

_INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|_[^_\s][^_]*?_|\[[^\]]+\]\([^)]+\))")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("؀" <= c <= "ۿ" for c in letters) > len(letters) / 2


def _blocks(md: str) -> list[tuple[str, str]]:
    """-> [(kind, text)] with kind in h1 h2 h3 bullet num p."""
    out: list[tuple[str, str]] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if m := re.match(r"(#{1,3})\s+(.*)", line):
            out.append((f"h{len(m.group(1))}", m.group(2)))
        elif m := re.match(r"\s*[-*]\s+(.*)", line):
            out.append(("bullet", m.group(1)))
        elif m := re.match(r"\s*(\d+)\.\s+(.*)", line):
            out.append(("num", f"{m.group(1)}. {m.group(2)}"))
        else:
            out.append(("p", line.strip()))
    return out


def _runs(text: str) -> list[tuple[str, bool, bool]]:
    """-> [(text, bold, italic)]; links become 'title (url)'."""
    runs: list[tuple[str, bool, bool]] = []
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            runs.append((part[2:-2], True, False))
        elif (part[0] in "*_") and part[-1] == part[0] and len(part) > 2:
            runs.append((part[1:-1], False, True))
        elif m := _LINK.fullmatch(part):
            runs.append((f"{m.group(1)} ({m.group(2)})", False, False))
        else:
            runs.append((part, False, False))
    return runs


# ----------------------------------------------------------------------------- Word
def to_docx(md: str) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)
    for kind, text in _blocks(md):
        rtl = _is_arabic(text)
        if kind in ("h1", "h2", "h3"):
            p = doc.add_heading(level=int(kind[1]))
        elif kind == "bullet":
            p = doc.add_paragraph(style="List Bullet")
        else:
            p = doc.add_paragraph()
        for t, bold, italic in _runs(text):
            r = p.add_run(t)
            r.bold, r.italic = bold or None, italic or None
            if rtl:
                r._r.get_or_add_rPr().append(OxmlElement("w:rtl"))
        if rtl:
            p._p.get_or_add_pPr().append(OxmlElement("w:bidi"))
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ----------------------------------------------------------------------------- PDF
def _font_files() -> dict[str, Path]:
    names = {"": "arial.ttf", "B": "arialbd.ttf", "I": "ariali.ttf", "BI": "arialbi.ttf"}
    dirs = [Path(os.environ["PDF_FONT_DIR"])] if os.getenv("PDF_FONT_DIR") else []
    dirs += [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/msttcorefonts"),
             Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental")]
    for d in dirs:
        if (d / names[""]).exists():
            return {style: (d / f) for style, f in names.items() if (d / f).exists()}
    dejavu = Path("/usr/share/fonts/truetype/dejavu")
    if (dejavu / "DejaVuSans.ttf").exists():
        return {"": dejavu / "DejaVuSans.ttf", "B": dejavu / "DejaVuSans-Bold.ttf",
                "I": dejavu / "DejaVuSans-Oblique.ttf", "BI": dejavu / "DejaVuSans-BoldOblique.ttf"}
    raise RuntimeError("No Unicode font found for the PDF. Set PDF_FONT_DIR to a folder with arial.ttf (and arialbd.ttf).")


def _pdf_markup(text: str) -> str:
    """fpdf2 inline markup: **bold**, __italic__. Links are written as 'title (url)'."""
    out = ""
    for t, bold, italic in _runs(text):
        t = t.replace("--", "–").replace("__", "_").replace("**", "")
        out += f"**{t}**" if bold else f"__{t}__" if italic else t
    return out


def to_pdf(md: str) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import TextDirection

    fonts = _font_files()
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.set_margins(18, 18, 18)
    for style, path in fonts.items():
        pdf.add_font("U", style, str(path))
    for style in ("B", "I", "BI"):                              # missing styles fall back to regular
        if style not in fonts:
            pdf.add_font("U", style, str(fonts[""]))
    pdf.add_page()
    pdf.set_text_shaping(True)
    sizes = {"h1": 20, "h2": 15, "h3": 12.5}
    for kind, text in _blocks(md):
        rtl = _is_arabic(text)
        pdf.set_text_shaping(True, direction="rtl" if rtl else "ltr")
        align = "R" if rtl else "L"
        pdf.set_x(pdf.l_margin)
        if kind in sizes:
            pdf.ln(3 if kind != "h1" else 0)
            pdf.set_font("U", "B", sizes[kind])
            pdf.multi_cell(0, sizes[kind] * 0.5, _pdf_markup(text).replace("**", ""), align=align, markdown=True)
            pdf.ln(1.5)
            continue
        pdf.set_font("U", "", 11)
        if kind == "bullet":
            text = "• " + text if not rtl else text + " •"
        pdf.multi_cell(0, 6, _pdf_markup(text), align=align, markdown=True)
        pdf.ln(1)
    return bytes(pdf.output())


FORMATS = {
    "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", to_docx),
    "pdf": ("application/pdf", to_pdf),
}

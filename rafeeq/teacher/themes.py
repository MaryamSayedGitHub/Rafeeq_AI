"""Slide themes: built-in presets, custom themes saved in the library, and extraction from a .pptx.

A theme is a plain dict:  {"name": str, "font": str, "colors": {key: "#RRGGBB", ...}}
with the colour keys the deck builder uses (COLOR_KEYS). Built-in presets are code; a custom theme is
a library asset (kind "theme") whose `settings` hold the same dict, so it can be reused by any deck.

extract_from_pptx() reads the colour scheme and fonts out of an existing PowerPoint file
(ppt/theme/theme1.xml) so a teacher can keep the look of a deck they already have.
"""

import io
import re
import zipfile
from xml.etree import ElementTree

COLOR_KEYS = ("ink", "ink_soft", "paper", "line", "blue", "blue_soft", "yellow", "green", "green_soft",
              "code_bg", "code_text")
DEFAULT_FONT = "Segoe UI"
MAX_THEME_BYTES = 30 * 1024 * 1024
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


class ThemeError(ValueError):
    """The file is not a usable PowerPoint theme, or the theme dict is invalid."""


PRESETS = {
    "rafeeq": {"name": "Rafeeq", "font": DEFAULT_FONT, "colors": {
        "ink": "#1B2540", "ink_soft": "#4A5572", "paper": "#F3F5F9", "line": "#D5DAE5", "blue": "#3A5BD9",
        "blue_soft": "#E4EAFB", "yellow": "#FFE066", "green": "#1F7A4D", "green_soft": "#DDF3E7",
        "code_bg": "#12182B", "code_text": "#E6EAF5"}},
    "teal": {"name": "Teal", "font": DEFAULT_FONT, "colors": {
        "ink": "#173F3B", "ink_soft": "#4E6B68", "paper": "#F0F8F5", "line": "#CFE2DC", "blue": "#087F73",
        "blue_soft": "#DDF1EC", "yellow": "#F2C14E", "green": "#23704F", "green_soft": "#E2F3EA",
        "code_bg": "#0E2A27", "code_text": "#E4F3EF"}},
    "sunrise": {"name": "Sunrise", "font": DEFAULT_FONT, "colors": {
        "ink": "#3A2230", "ink_soft": "#6E5460", "paper": "#FFF7F0", "line": "#EBD9CC", "blue": "#D9582B",
        "blue_soft": "#FBE6DC", "yellow": "#FFD166", "green": "#2E7D5B", "green_soft": "#E1F3EA",
        "code_bg": "#2A1824", "code_text": "#FBEDE4"}},
    "graphite": {"name": "Graphite", "font": DEFAULT_FONT, "colors": {
        "ink": "#202428", "ink_soft": "#5B636B", "paper": "#F4F4F2", "line": "#D8D9D6", "blue": "#3F4C5C",
        "blue_soft": "#E3E6EA", "yellow": "#E8B931", "green": "#3F7D58", "green_soft": "#E2EEE7",
        "code_bg": "#16191C", "code_text": "#E8EAEC"}},
}


def preset_list() -> list[dict]:
    return [{"theme_id": f"preset:{key}", **theme} for key, theme in PRESETS.items()]


# ------------------------------------------------------------------ colour helpers
def _rgb(hex_color: str) -> tuple[int, int, int]:
    return int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)


def _hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(c))) for c in rgb))


def mix(a: str, b: str, amount: float) -> str:
    """`amount` of b mixed into a (0 = a, 1 = b)."""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(x + (y - x) * amount for x, y in zip(ra, rb)))


def luminance(hex_color: str) -> float:
    r, g, b = (c / 255 for c in _rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ------------------------------------------------------------------ validation
def normalize(theme: dict) -> dict:
    """A safe theme: every colour a valid #RRGGBB (missing ones come from the default), a short font name."""
    if not isinstance(theme, dict):
        raise ThemeError("A theme must be an object")
    base = PRESETS["rafeeq"]["colors"]
    given = theme.get("colors") or {}
    colors = {}
    for key in COLOR_KEYS:
        value = str(given.get(key, base[key])).strip()
        if not _HEX.match(value):
            raise ThemeError(f"Colour '{key}' must look like #RRGGBB")
        colors[key] = value.upper()
    font = re.sub(r"[^\w .\-]", "", str(theme.get("font") or DEFAULT_FONT), flags=re.UNICODE).strip()[:60]
    name = re.sub(r"\s+", " ", str(theme.get("name") or "Theme")).strip()[:60] or "Theme"
    return {"name": name, "font": font or DEFAULT_FONT, "colors": colors}


# ------------------------------------------------------------------ extraction
def _scheme_colors(root) -> dict[str, str]:
    scheme = root.find(f".//{_A}clrScheme")
    out = {}
    if scheme is None:
        return out
    for child in scheme:
        name = child.tag.replace(_A, "")
        inner = child[0] if len(child) else None
        if inner is None:
            continue
        value = inner.get("val") if inner.tag == f"{_A}srgbClr" else inner.get("lastClr")
        if value and re.fullmatch(r"[0-9a-fA-F]{6}", value):
            out[name] = "#" + value.upper()
    return out


def extract_from_pptx(data: bytes, fallback_name: str = "") -> dict:
    """Read colours and the body font from the first theme inside a .pptx / .potx. Raises ThemeError."""
    if len(data) > MAX_THEME_BYTES:
        raise ThemeError("The file is larger than 30 MB")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = sorted(n for n in z.namelist() if re.fullmatch(r"ppt/theme/theme\d+\.xml", n))
            if not names:
                raise ThemeError("No theme found in this file")
            xml = z.read(names[0])
    except zipfile.BadZipFile:
        raise ThemeError("This is not a PowerPoint file") from None
    try:
        root = ElementTree.fromstring(xml)  # the theme XML of a pptx: no external entities are resolved
    except ElementTree.ParseError:
        raise ThemeError("The theme in this file is damaged") from None

    c = _scheme_colors(root)
    if "accent1" not in c:
        raise ThemeError("The theme has no colour scheme")
    scheme = root.find(f".//{_A}clrScheme")
    name = (scheme.get("name") if scheme is not None else "") or root.get("name") or fallback_name or "Imported theme"

    white = "#FFFFFF"
    ink = c.get("dk2") if luminance(c.get("dk2", white)) < 0.35 else c.get("dk1", "#1B2540")
    paper = c.get("lt2") if luminance(c.get("lt2", "#000000")) > 0.85 else c.get("lt1", "#F3F5F9")
    blue = c["accent1"]
    yellow = c.get("accent4") or c.get("accent2") or "#FFE066"
    green = c.get("accent3") or c.get("accent6") or "#1F7A4D"

    font = DEFAULT_FONT
    for tag in ("minorFont", "majorFont"):
        node = root.find(f".//{_A}{tag}/{_A}latin")
        if node is not None and node.get("typeface"):
            font = node.get("typeface")
            break

    return normalize({"name": name, "font": font, "colors": {
        "ink": ink, "ink_soft": mix(ink, white, 0.35), "paper": paper, "line": mix(paper, ink, 0.12),
        "blue": blue, "blue_soft": mix(blue, white, 0.85), "yellow": yellow, "green": green,
        "green_soft": mix(green, white, 0.85), "code_bg": mix(ink, "#000000", 0.35), "code_text": "#E6EAF5"}})

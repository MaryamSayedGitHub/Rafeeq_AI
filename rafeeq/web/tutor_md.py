"""The study file's Markdown (headings, bullets, numbered steps, **bold**, links) as safe HTML, for the `md` template filter."""
from __future__ import annotations

import html
import re

from markupsafe import Markup

from rafeeq.tutor import export

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_NUM = re.compile(r"^\d+\.\s*")


def _inline(text: str) -> str:
    text = html.escape(text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    return _LINK.sub(r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', text)


def render_md(md: str) -> Markup:
    out, open_list = [], None
    for kind, text in export._blocks(md):
        want = {"bullet": "ul", "num": "ol"}.get(kind)
        if open_list and open_list != want:
            out.append("</%s>" % open_list)
            open_list = None
        if want and not open_list:
            out.append("<%s>" % want)
            open_list = want
        if kind in ("h1", "h2", "h3"):
            out.append("<%s>%s</%s>" % (kind, _inline(text), kind))
        elif want:
            out.append("<li>%s</li>" % _inline(_NUM.sub("", text) if kind == "num" else text))
        else:
            out.append("<p>%s</p>" % _inline(text))
    if open_list:
        out.append("</%s>" % open_list)
    return Markup("".join(out))

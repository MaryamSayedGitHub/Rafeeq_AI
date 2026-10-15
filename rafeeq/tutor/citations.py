"""Citation verification: a citation is only valid when it is one of the passages M2 really returned for this question.

The tutor never writes a citation itself (answer.py copies file_name / page from the chunks), but nothing used to PROVE that. verify() checks:
  1. every citation is well formed (non-empty file_name, page >= 1);
  2. it matches (file_name, page) of a retrieved chunk, so a name or page the model invented is dropped, never shown;
  3. the answer is supported by the cited text: when answer and passage are in the same script, they must share real content words
     (numbers count too). Across scripts (Arabic answer from an English passage) the overlap cannot be measured: "unchecked", not "supported".
An answer that is unsupported by its own citations is flagged and answer.py then says "not found" instead of showing it.
"""
from __future__ import annotations

import re

MIN_SHARED = 1                       # content words an answer must share with its cited passages (zero = nothing links them)
_WORD = re.compile(r"[A-Za-zء-يٱ-ۓ]{3,}|\d+")
_STOP = {"the", "and", "for", "that", "this", "with", "are", "was", "from", "which", "have", "has", "into", "its", "their", "can", "not",
         "في", "من", "على", "التي", "الذي", "هذا", "هذه", "إلى", "عن", "أن", "كان", "هو", "هي"}


def _script(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "none"
    return "ar" if sum("؀" <= c <= "ۿ" for c in letters) > len(letters) / 2 else "en"


def _stem(w: str) -> str:
    """Crude on purpose: drop the Arabic article / common prefixes and English plural / -ing / -ed, so 'الطاقة' matches 'بالطاقة', 'cells' matches 'cell'."""
    w = re.sub(r"^(?:وال|بال|فال|كال|لل|ال)(?=.{3})", "", w)
    return re.sub(r"(?:ing|ed|es|s)$", "", w) if re.fullmatch(r"[a-z]{5,}", w) else w


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def verify(citations: list[dict], retrieved: list[dict], answer: str = "") -> dict:
    """-> {"valid": [citation...], "invalid": [citation...], "support": "supported" | "unsupported" | "unchecked" | "none", "ok": bool}
    retrieved = the chunks M2 returned ({file_name, page, text}). ok = at least one valid citation and the answer is not unsupported."""
    index = {(c.get("file_name"), c.get("page")): c.get("text", "") for c in retrieved}
    valid, invalid = [], []
    for c in citations:
        name, page = c.get("file_name"), c.get("page")
        well_formed = isinstance(name, str) and name.strip() and isinstance(page, int) and not isinstance(page, bool) and page >= 1
        (valid if well_formed and (name, page) in index else invalid).append({"file_name": name, "page": page})
    if not valid:
        return {"valid": [], "invalid": invalid, "support": "none", "ok": False}
    if not answer.strip():
        return {"valid": valid, "invalid": invalid, "support": "unchecked", "ok": not invalid}
    cited = " ".join(index[(c["file_name"], c["page"])] for c in valid)
    sa, sc = _script(answer), _script(cited)
    if "none" in (sa, sc) or sa != sc:
        support = "unchecked"
    else:
        shared = _words(answer) & _words(cited)
        support = "supported" if len(shared) >= min(MIN_SHARED, len(_words(answer))) else "unsupported"
    return {"valid": valid, "invalid": invalid, "support": support, "ok": support != "unsupported"}

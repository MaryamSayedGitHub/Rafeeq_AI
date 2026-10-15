"""Is the text we were given reliable enough to judge? (handwriting read by a model can be wrong)

assess() looks at every line the reader gave us and says which ones cannot be trusted and why. The checker then does NOT judge those lines
(no model call, no hint made up from a misread): the step is "check" and the hint asks the student what the line says.
If too much of the page is unreliable, the whole check is a request for clarification (retake the photo or type the line).
Pure rules, no model: the same input always gives the same answer.
"""
from __future__ import annotations

import re

CONF_MIN = 0.6               # overall reader confidence below this: ask for clarification (service.read_photo gives 0.5-0.95)
BAD_SHARE = 0.5              # MORE than this share of the lines unreliable: the page as a whole is unreliable (one bad line of two is not)

_MARK = re.compile(r"\[\s*\?+\s*\]|\?{3,}|�|\[\s*(?:illegible|unclear|unreadable|غير\s*واضح)[^\]]*\]|<\s*illegible\s*>", re.I)
_STRUCK = re.compile(r"~~.+?~~|<\s*s\s*>|\[\s*(?:crossed[\s-]*out|struck|scratched|مشطوب)[^\]]*\]|\(\s*(?:crossed[\s-]*out|مشطوب)[^)]*\)", re.I)
_AMBIG = re.compile(r"[A-Za-z0-9)٠-٩]\s*\?\s*[A-Za-z0-9(٠-٩]|=\s*\?\s*$|±|\+\s*[-−]\s*\d|[-−]\s*\+\s*\d|[+\-−]{2,}\s*\d")
_AR = re.compile(r"[؀-ۿ]")
_LAT = re.compile(r"[A-Za-z]")
_ALNUM_OK = re.compile(r"[\w\s=+\-−*/^×÷()\[\]{}<>.,:;!?'\"%√²³_|\\#&@$~`؀-ۿ]", re.U)

REASONS = ("unreadable", "crossed_out", "ambiguous_sign", "noisy", "mixed_script", "too_short")

ASK = {                                # ONE sentence each (the same one-hint rule as grading.one_hint)
    "unreadable": {"ar": "لم أستطع قراءة هذا السطر بوضوح، هل يمكنك كتابته لي؟", "en": "I could not read this line clearly, could you type it for me?"},
    "crossed_out": {"ar": "هذا السطر يبدو مشطوبًا، هل تريد أن أراجعه أم تتجاهله؟", "en": "This line looks crossed out, should I check it or ignore it?"},
    "ambiguous_sign": {"ar": "لست متأكدًا من الإشارة أو العملية في هذا السطر، ما هي بالضبط؟", "en": "I am not sure about the sign or operation in this line, what is it exactly?"},
    "noisy": {"ar": "قراءة هذا السطر غير موثوقة، هل يمكنك كتابته لي؟", "en": "My reading of this line is not reliable, could you type it for me?"},
    "mixed_script": {"ar": "يبدو أن حروفًا عربية وإنجليزية اختلطت في هذا السطر، ماذا كتبتَ فيه؟", "en": "Arabic and English letters seem mixed in this line, what did you write?"},
    "too_short": {"ar": "هذا السطر قصير جدًا لأحكم عليه، هل هو كامل؟", "en": "This line is too short for me to judge, is it complete?"},
}
PAGE_ASK = {
    "ar": "لا أستطيع الحكم على هذا الحل لأن قراءة الخط غير موثوقة. أعد التصوير بإضاءة أفضل أو اكتب الخطوات هنا.",
    "en": "I cannot judge this work because the handwriting was not read reliably. Retake the photo with better light, or type the steps here.",
}


def _lang(lang: str | None) -> str:
    return lang if lang in ("ar", "en") else "ar"


def line_reasons(line: str) -> list[str]:
    """Why this single line cannot be trusted (empty list = it can)."""
    s = line.strip()
    out = []
    if _MARK.search(s):
        out.append("unreadable")
    if _STRUCK.search(s):
        out.append("crossed_out")
    if _AMBIG.search(s):
        out.append("ambiguous_sign")
    if s:
        bad = sum(1 for c in s if not _ALNUM_OK.match(c))
        if len(s) >= 4 and bad / len(s) > 0.3:
            out.append("noisy")
        if any(_AR.search(tok) and _LAT.search(tok) for tok in re.findall(r"\S+", s) if len(tok) > 2 and not re.fullmatch(r"[\W\d]*[A-Za-z]\d*[\W\d]*", tok)):
            out.append("mixed_script")
        if len(s) < 2 and not s.isalnum():
            out.append("too_short")
    return out


def assess(lines: list[str], confidence: float | None = None, lang: str | None = None) -> dict:
    """-> {"lines": [{"i", "reasons", "ask"}] only for unreliable lines, "page_unreliable": bool, "page_ask": str, "reliable": bool}"""
    lang = _lang(lang)
    texts = [l for l in lines if l.strip()]
    flagged = []
    for i, l in enumerate(texts, 1):
        r = line_reasons(l)
        if r:
            flagged.append({"i": i, "reasons": r, "ask": ASK[r[0]][lang]})
    low_conf = confidence is not None and confidence < CONF_MIN
    page_bad = bool(texts) and (low_conf or len(flagged) / len(texts) > BAD_SHARE)
    return {"lines": flagged, "page_unreliable": page_bad, "page_ask": PAGE_ASK[lang] if page_bad else "", "reliable": not flagged and not page_bad}

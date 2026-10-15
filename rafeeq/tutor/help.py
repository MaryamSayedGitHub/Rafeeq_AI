"""Help for a math problem or code (typed or read from a photo): 3 hints, one at a time, then the full solution with steps.

make_help() asks the model once for all of it, in the language of the SITE (Arabic or English). Hints go from general to specific and never
contain the final answer; the solution is only shown when the student asks for it. The result is cached on disk under a key, so
the page can reveal hint 1, 2, 3 and the solution with separate clicks without paying for the model again.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import llm
from .grading import fmt_ctx
from .study_store import data_dir

LANG_NAME = {"ar": "Arabic", "en": "English"}

# used when the model is not reachable: still three nudges, in the site language
GENERIC_HINTS = {
    "ar": ["اسأل نفسك: ما القاعدة أو القانون أو الفكرة التي تحتاجها هذه المسألة؟",
           "طبّق هذه القاعدة على أرقام مسألتك، خطوة صغيرة في كل مرة، وراجع إشارة كل رقم.",
           "جرّب الآن الخطوة التالية بنفسك، ثم قارن ناتجها بالخطوة السابقة للتأكد."],
    "en": ["Ask yourself: which rule, law or idea does this problem need?",
           "Apply that rule to the numbers of your problem, one small step at a time, and watch the sign of every number.",
           "Now try the next step yourself, then compare it with the previous step to make sure it follows."],
}


def _prompt(text: str, kind: str, context: list[dict], lang: str) -> str:
    return f"""You are a patient tutor. This is the student's {kind} (the problem, and/or their attempt):
{text}

Course material (use it when it has the law or formula):
{fmt_ctx(context, 500)}

Return JSON only:
{{"hints": ["...", "...", "..."], "solution": ["step 1 ...", "step 2 ..."], "final_answer": "..."}}
Rules:
- Write EVERYTHING in {LANG_NAME[lang]} (code and math symbols stay as they are).
- Exactly 3 hints, each more specific than the last. Hint 1 names the law, formula or concept (quote the course material if it is there).
  Hint 2 says how to apply it to THIS problem. Hint 3 gives the next concrete move but NOT the final answer.
- solution: the complete correct worked solution, one short step per item, every step shown (no skipped steps).
  For code: explain the bug, then give the corrected code as the last item.
- final_answer: the final result alone, short."""


def make_help(text: str, kind: str, lang: str, context: list[dict] | None = None) -> dict:
    """-> {"hints": [3 str], "solution": [str], "final_answer": str, "llm_ok": bool}"""
    context = context or []
    res = llm.ask_json(_prompt(text, kind, context, lang), max_tokens=3000)
    ok = isinstance(res, dict)
    hints = [str(h).strip() for h in (res.get("hints") or []) if str(h).strip()][:3] if ok else []
    solution = [str(s).strip() for s in (res.get("solution") or []) if str(s).strip()] if ok else []
    final = str(res.get("final_answer", "")).strip() if ok else ""
    if len(final) >= 4:                                            # keep the hints from leaking the answer
        hints = [h.replace(final, "…") for h in hints]
    for generic in GENERIC_HINTS[lang]:                            # always three hints, whatever the model gave
        if len(hints) >= 3:
            break
        hints.append(generic)
    return {"hints": hints[:3], "solution": solution, "final_answer": final, "llm_ok": ok and bool(solution)}


# ----------------------------------------------------------------------------- cache: reveal one piece per click
def _path(key: str):
    d = data_dir() / "help"
    d.mkdir(exist_ok=True)
    return d / f"{key}.json"


def prepare(text: str, kind: str, lang: str, context: list[dict] | None = None) -> str:
    """Remember the problem and return its key. Nothing is generated yet (the model runs on the first hint)."""
    key = hashlib.sha1(f"{kind}|{lang}|{text}".encode("utf-8")).hexdigest()[:16]
    if not _path(key).exists():
        _path(key).write_text(json.dumps({"text": text, "kind": kind, "lang": lang, "context": context or [], "help": None}, ensure_ascii=False), encoding="utf-8")
    return key


def get(key: str) -> dict | None:
    """The help for this key (generated on the first call), or None for an unknown key."""
    if not re.fullmatch(r"[0-9a-f]{16}", key or "") or not _path(key).exists():
        return None
    data = json.loads(_path(key).read_text(encoding="utf-8"))
    if data["help"] is None:
        data["help"] = make_help(data["text"], data["kind"], data["lang"], data["context"])
        _path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["help"]

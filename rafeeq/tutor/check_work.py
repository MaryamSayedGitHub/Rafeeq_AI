"""POST /tutor/check-work: check the text read from a photo of the student's own work (contracts: CheckWorkRequest -> HandwritingCheck).

The route gets the text only (SnapResult.text). If /snap saved this photo's line boxes (same file_id, same number of lines) the REAL box of the
wrong line is returned; otherwise the box is placed from the line's position: line i of n sits at y = (i-1)/n, height 1/n, across the page.
Only the FIRST wrong step is boxed, and the hint is a nudge, never the corrected line or the final answer.
The answer is in the site language (`lang`, default: the language of the text).
"""
from __future__ import annotations

from rafeeq.snap.service import load_lines

from . import uncertainty
from .answer import guess_language
from .grading import classify, grade
from .search import relevant, search

ALL_GOOD = {"ar": "عمل ممتاز! كل الخطوات تبدو صحيحة.", "en": "Great work! Every step looks right."}
NO_TEXT = {"ar": "لم أستطع قراءة أي خطوات من الصورة. جرّب صورة أوضح.", "en": "I could not read any steps from the photo. Try a clearer one."}
CHECK_THIS = {"ar": "راجع الخطوات التي لست متأكدًا منها بحذر.", "en": "Look carefully at the steps I am unsure about."}
INCOMPLETE = {"ar": "خطواتك حتى الآن سليمة لكن الحل لم يكتمل بعد. ما الخطوة التالية للوصول إلى الإجابة النهائية؟",
              "en": "Your steps so far are fine, but the solution is not finished. What is the next step towards the final answer?"}


def _box(i: int, n: int) -> dict:
    return {"x": 0.05, "y": round((i - 1) / n, 3), "width": 0.9, "height": round(1 / n, 3)}


def check_work_detail(room_id: str, file_id: str, text: str, lang: str | None = None, confidence: float | None = None) -> dict:
    """The contract's HandwritingCheck plus: "needs_help" (a wrong step, an unsure step or unfinished work), "complete", "kind",
    "items" (every step: status ok | check | error, one hint, its box) and "passages" (the course pages used: file_name + page).
    Uncertainty: a line the reader may have misread (unreadable, crossed out, ambiguous sign, noise...) is never judged: it is "check" with a question
    for the student, and when the whole page is unreliable (low reader confidence, too many bad lines) nothing is judged at all: "needs_clarification"."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lang = lang or guess_language(text)
    if not lines:
        return {"boxes": [], "steps": [], "hint": NO_TEXT[lang], "needs_help": False, "complete": True, "kind": "question", "items": [], "passages": [],
                "reliable": True, "needs_clarification": False, "clarification": ""}
    kind = classify(lines)
    unc = uncertainty.assess(lines, confidence, lang)
    flagged = {f["i"]: f["ask"] for f in unc["lines"]}
    if unc["page_unreliable"]:                                      # nothing is judged: no course search, no model call, no invented hint
        steps = [{"i": i, "text": t, "status": "check", "hint": flagged.get(i) or uncertainty.ASK["noisy"][lang]} for i, t in enumerate(lines, 1)]
        context, complete = [], True
        hint = unc["page_ask"]
    else:
        context = relevant(search(room_id, " ".join(lines)[:300], top_k=3), text)     # course material helps the model name the right law / formula
        steps, _concepts, _llm_ok, complete = grade(lines, "code" if kind == "code" else "math", context, lang)
        for s in steps:
            if s["i"] in flagged:                                   # a line that may be misread is never marked wrong (or right): ask instead
                s["status"], s["hint"] = "check", flagged[s["i"]]
        hint = ""
    first_error = next((s for s in steps if s["status"] == "error"), None)
    unsure = next((s for s in steps if s["status"] == "check"), None)
    if hint:
        pass
    elif first_error:
        hint = first_error["hint"] or CHECK_THIS[lang]
    elif flagged:                                                   # a line that may be misread comes before a vague "not sure"
        hint = next(iter(flagged.values()))
    elif unsure:
        hint = unsure["hint"] or CHECK_THIS[lang]
    elif not complete:
        hint = INCOMPLETE[lang]
    else:
        hint = ALL_GOOD[lang]
    saved = load_lines(file_id)
    real = saved["boxes"] if saved and len(saved["boxes"]) == len(steps) else None
    box_of = lambda i: real[i - 1] if real else _box(i, len(steps))
    boxes = [box_of(first_error["i"])] if first_error else []
    # every step with its own colour (ok / check / error), its own box and exactly one hint (none for an ok step): for the vision UI and API
    items = [{"i": s["i"], "text": s["text"], "status": s["status"], "hint": s["hint"], "box": box_of(s["i"])} for s in steps]
    return {"boxes": boxes, "steps": [{"text": s["text"], "correct": s["status"] == "ok"} for s in steps], "hint": hint,
            "needs_help": bool(first_error or unsure or not complete), "complete": complete, "kind": kind, "items": items,
            "passages": [{"file_name": c["file_name"], "page": c["page"]} for c in context],
            "reliable": unc["reliable"], "needs_clarification": bool(flagged) or unc["page_unreliable"],
            "clarification": unc["page_ask"] or (next(iter(flagged.values())) if flagged else "")}


def check_work(room_id: str, file_id: str, text: str, lang: str | None = None) -> dict:
    d = check_work_detail(room_id, file_id, text, lang)
    return {k: d[k] for k in ("boxes", "steps", "hint")}

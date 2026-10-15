"""Step-by-step checking of a student's own work (used by POST /tutor/check-work).

Every line is marked ok / check / error, with ONE short hint for the wrong ones (a nudge, never the corrected line or the answer).
Pure-number equalities are verified by code (rule_check), so the check also works without an LLM.
"""
from __future__ import annotations

import ast
import operator as op
import re

from . import llm

STATUSES = ("ok", "check", "error")


# --------------------------------------------------------------------------- what is on the text?
_CODE_STRONG = re.compile(
    r"\bdef\s+\w+\s*\(|\bprint\s*\(|\breturn\b|\bfor\s+\w+\s+in\b|\bwhile\b.*:|\bif\b.*:\s*$|\belif\b|\bimport\s+\w+|"
    r"\bclass\s+\w+|console\.log|=>|\bfunction\s+\w*\s*\(|\b(?:const|let|var)\s+\w+\s*=|\bint\s+\w+\s*[=;(]|#include",
    re.M)
_CODE_SYM = re.compile(r"==|!=|\+=|-=|->|;\s*$", re.M)
_MATH_OPS = re.compile(r"\d\s*[-+*/^×÷]\s*[\d(a-z]|[a-z]\s*[-+*/^]\s*\d|\d[a-z]\b|[a-z]\^?[²2]\b|√|∫|∑|\b(?:sin|cos|tan)\b", re.I)


def classify(lines: list[str]) -> str:
    text = "\n".join(l for l in lines if l.strip())
    if not text:
        return "question"
    n = max(1, len([l for l in lines if l.strip()]))
    if _CODE_STRONG.search(text) or len(_CODE_SYM.findall(text)) >= 2:
        return "code"
    words = len(re.findall(r"[A-Za-z؀-ۿ]{3,}", text))
    has_eq = bool(re.search(r"[=<>≤≥]", text))
    digits = len(re.findall(r"\d", text))
    ops = len(_MATH_OPS.findall(text))
    if words <= 2 * n and ((has_eq and digits >= 1) or ops >= 2):
        return "math"
    return "question"


# --------------------------------------------------------------------------- helpers
def fmt_ctx(context: list[dict], limit: int = 320) -> str:
    return "\n".join(f"- {c.get('file_name', '')} p.{c.get('page', '')}: {c['text'][:limit]}" for c in context) or "(none)"


_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _safe_eval(expr: str) -> float:
    """Evaluate +, -, *, /, ** on plain numbers only (no names, no calls)."""
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
           ast.USub: op.neg, ast.UAdd: op.pos}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 8:
                raise ValueError("exponent too large")
            return ops[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
            return ops[type(n.op)](ev(n.operand))
        raise ValueError("unsupported")
    return ev(ast.parse(expr, mode="eval"))


RULE_HINTS = {
    "error": {"ar": "راجع الحساب في هذه الخطوة.", "en": "Re-check the arithmetic in this step."},
    "check": {"ar": "تأكد أن هذه الخطوة تنتج من التي قبلها.", "en": "Make sure this step follows from the previous one."},
}


def rule_check(step: str, lang: str = "ar") -> dict:
    """Offline fallback (no LLM): verify pure-number equalities such as '2 + 3 * 4 = 14'."""
    s = step.translate(_DIGITS).replace("×", "*").replace("÷", "/").replace("^", "**").replace("−", "-").replace(" ", "")
    if "=" in s and re.fullmatch(r"[0-9+\-*/().=]+", s):
        try:
            vals = [_safe_eval(p) for p in s.split("=") if p]
            if len(vals) >= 2:
                if all(abs(v - vals[0]) < 1e-6 for v in vals):
                    return {"status": "ok", "hint": ""}
                return {"status": "error", "hint": RULE_HINTS["error"][lang]}
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            pass
    return {"status": "check", "hint": RULE_HINTS["check"][lang]}


_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")
# a "line of working" inside a hint (2x = 8, x = 4, 14 - 6 = 8): that is the solution, not a nudge
_WORKING = re.compile(r"[\w()²^.]+\s*[-+*/×÷^]?\s*[\w()²^.]*\s*=\s*[-+]?[\w(]*\d[\w)²^.]*|\d\s*[-+*/×÷^]\s*\d")
NUDGE = {
    "error": {"ar": "راجع هذه الخطوة: ما القاعدة التي استخدمتها فيها، وهل طبّقتها صحيحًا؟", "en": "Re-check this step: which rule did you use, and did you apply it correctly?"},
    "check": {"ar": "تأكد أن هذه الخطوة تنتج من التي قبلها.", "en": "Make sure this step follows from the previous one."},
}


def one_hint(hint: str, status: str, lang: str = "ar", step: str = "") -> str:
    """Exactly one short hint for a step that is not ok (none for an ok step). It must not hold the corrected line or the final result."""
    if status == "ok":
        return ""
    hint = re.sub(r"\s+", " ", str(hint or "")).strip()
    hint = _SENTENCE_END.split(hint)[0] if hint else ""             # one sentence only
    words = hint.split()
    if len(words) > 28:
        hint = " ".join(words[:28]) + "…"
    if step.strip() and len(step.strip()) > 3 and step.strip() in hint:
        hint = hint.replace(step.strip(), "…")
    if not hint or _WORKING.search(hint):                           # empty, or it writes out a line of the solution
        hint = NUDGE[status if status in NUDGE else "check"][lang if lang in ("ar", "en") else "ar"]
    return hint


def _clean(res: dict, step: str, lang: str = "ar") -> dict:
    status = res.get("status", "check")
    status = status if status in STATUSES else "check"
    return {"status": status, "hint": one_hint(res.get("hint", ""), status, lang, step)}


# --------------------------------------------------------------------------- 1) check the work
def _analyze_prompt(texts: list[str], kind: str, context: list[dict], lang: str | None = None) -> str:
    hint_lang = {"ar": "Arabic", "en": "English"}.get(lang or "", "the same language as the line")
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    return f"""You are a patient tutor. A student wrote this {kind}, one numbered line per step:
{numbered}

Course material (may be empty):
{fmt_ctx(context)}

Check every line. Return JSON only:
{{"steps": [{{"i": 1, "status": "ok", "hint": ""}}], "complete": true, "concepts": "short search query naming the law, formula or concept needed"}}
Rules:
- status ok = correct and follows from the previous lines (hint must be ""). check = not sure, or the handwriting may have been misread. error = wrong.
- For check/error give ONE short hint (max 20 words) in {hint_lang}. Point to the idea to re-check. NEVER write the corrected line, the correct operation, or the final answer: ask a question or point at the idea (for example "what happens to the sign of a number when it moves to the other side?").
- complete = true when the lines reach a final answer (math) or a finished working program (code); false when the work stops half way.
- If the text is only a problem statement with no worked steps, mark every line "ok".
- For code, an error is anything that would crash or give a wrong result (syntax, indentation, off-by-one, = instead of ==, missing return)."""


def grade(lines: list[str], kind: str, context: list[dict], lang: str | None = None):
    """Returns (steps, concepts, llm_ok, complete). steps = [{"i", "text", "status", "hint"}] one per non-empty line.
    complete = the work reaches a final answer (True when the model could not say)."""
    texts = [l.strip() for l in lines if l.strip()]
    res = llm.ask_json(_analyze_prompt(texts, kind, context, lang)) if texts else None
    llm_ok = isinstance(res, dict) and isinstance(res.get("steps"), list)
    by_i: dict[int, dict] = {}
    if llm_ok:
        for s in res["steps"]:
            try:
                by_i[int(s.get("i"))] = s
            except (TypeError, ValueError, AttributeError):
                pass
    steps = []
    for i, t in enumerate(texts, 1):
        rc = rule_check(t, lang if lang in ("ar", "en") else "ar")
        base = rc if rc["status"] in ("ok", "error") else (by_i.get(i) or rc)   # a verified sum beats the model; no model answer -> offline check
        steps.append({"i": i, "text": t, **_clean(base, t, lang if lang in ("ar", "en") else "ar")})
    concepts = str(res.get("concepts", "")).strip() if llm_ok else ""
    complete = bool(res.get("complete", True)) if llm_ok else True
    return steps, concepts, llm_ok, complete



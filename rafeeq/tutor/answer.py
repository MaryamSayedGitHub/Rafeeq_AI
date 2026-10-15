"""POST /tutor/answer: a grounded answer from the room's material only (contracts: TutorRequest -> TutorAnswer).

M2 gives at most 3 passages; the model answers from those passages and says which ones it used. Citations are the
passages' own file_name / page, never written by the model. Without an LLM the best passage itself is the answer.
"""
from __future__ import annotations

import re

from . import citations as cite
from . import llm, memory
from .search import TOP_K, search_with_source

NOT_FOUND = {"ar": "لم أجد الإجابة في مواد هذه المادة، لذلك لن أخمّن.", "en": "I could not find this in the course material, so I will not guess."}


def guess_language(text: str) -> str:
    """By WORDS, not letters: "ما هو الـ photosynthesis؟" is an Arabic question about an English term. A tie goes to Arabic."""
    ar = en = 0
    for w in re.findall(r"[^\W\d_]+", text):
        if any("؀" <= c <= "ۿ" for c in w):
            ar += 1
        else:
            en += 1
    return "ar" if ar and ar >= en else "en"


_WANTS_EN = re.compile(r"(?:\b(?:in|into|to)\s+english\b|\benglish\s+(?:please|only)\b|بال[إا]نج[ل]?[يى]ز[يىة]+|بال[إا]نجلش|ب[إا]نجليزي)", re.I)
_WANTS_AR = re.compile(r"(?:\b(?:in|into|to)\s+arabic\b|\barabic\s+(?:please|only)\b|بالعربي[ةه]?\b|بالعربى)", re.I)


def answer_language(question: str) -> str:
    """The language the ANSWER is written in: the one the student asks for ("... بالانجليزي", "in Arabic"), else the language of the question."""
    if _WANTS_EN.search(question):
        return "en"
    if _WANTS_AR.search(question):
        return "ar"
    return guess_language(question)


_DETAIL = re.compile(r"(?:\bin[\s-]+(?:detail|depth)\b|\bdetailed\b|\bin\s+full\b|\bthorough(?:ly)?\b|\belaborate\b|\bcomprehensive\b|\bat\s+length\b|"
                     r"بالتفصيل|بتفصيل|تفصيل[يى]|تفصيلا|تفصيلاً|مفصّ?ل[ةه]?|بالتفاصيل|باستفاضة|باستفاضه|بإسهاب|باسهاب|شرح\s+وافي|وافي[ةه]?|بالتفصيل\s+الممل|الممل)", re.I)
DETAIL_TOP_K = 8


def wants_detail(question: str) -> bool:
    """"explain it in detail" / "بالتفصيل الممل": the student wants the long answer, with more of the material (or of the web)."""
    return bool(_DETAIL.search(question or ""))


def keeps_english_terms(question: str, lang: str) -> bool:
    """An English question answered in Arabic keeps the technical terms in English (the student studies them in English)."""
    return lang == "ar" and guess_language(question) == "en"


def _prompt(question: str, chunks: list[dict], lang: str) -> str:
    detail = wants_detail(question)
    ex = "\n".join(f"[{i}] ({c['file_name']} p.{c['page']}) {c['text'][:1500 if detail else 700]}" for i, c in enumerate(chunks, 1))
    length = ("the student asked for it IN DETAIL: write a full, well organised explanation (about 350-500 words): what it is, how it works step by step, "
              "the important points and one example, using every relevant excerpt. Put the main parts in the \"steps\" list too (up to 6). No outside knowledge."
              if detail else "simple short words (max 120 words). No outside knowledge.")
    name = "Arabic" if lang == "ar" else "English"
    terms = " Keep every technical term in English (Latin letters), the rest in Arabic: write 'mitochondria', not its Arabic spelling." if keeps_english_terms(question, lang) else ""
    who = memory.note()
    return f"""You are a friendly tutor. Answer the student's question using ONLY the numbered course excerpts below.
{('About the student: ' + who) if who else ''}
Question: {question}

Excerpts:
{ex}

Return JSON only: {{"grounded": true, "answer": "...", "steps": [], "used": [1]}}
Rules:
- Write in {name}, {length}{terms}
- "used" = the excerpt numbers your answer relies on.
- "steps": leave it [] for a normal answer. Fill it (2-6 short strings) ONLY when the question asks how to do something / for steps / to explain a process.
- If the excerpts do not contain the answer, return {{"grounded": false, "answer": "", "steps": [], "used": []}}."""


def related(text: str, chunks: list[dict]) -> bool:
    """Do the passages explain the SAME subject as this text (a photo's)? M2 always returns its nearest passages, even when they are about
    something else, so a photo of "the Peace of Westphalia" must not be answered from a page on mitochondria."""
    from .search import relevant
    ex = "\n".join(f"[{i}] {c['text'][:400]}" for i, c in enumerate(chunks, 1))
    res = llm.ask_json(f"""Text written in a photo:
{text[:600]}

Course excerpts:
{ex}

Do the excerpts explain the SAME subject as the photo text? A different topic in a similar field, or only a shared common word, is NOT the same.
Return JSON only: {{"related": true}} or {{"related": false}}""", max_tokens=60)
    if isinstance(res, dict) and isinstance(res.get("related"), bool):
        return res["related"]
    if cite._script(text) != cite._script(chunks[0]["text"]):      # no model and two languages: it cannot be measured, so do not block
        return True
    return bool(relevant(chunks, text))


def _citations(chunks: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in chunks:
        key = (c["file_name"], c["page"])
        if key not in seen:
            seen.add(key)
            out.append({"file_name": key[0], "page": key[1]})
    return out


def tutor_answer(room_id: str, question: str, language: str | None = None, chunks: list[dict] | None = None, topic_id: str | None = None,
                 check_related: bool = False) -> dict:
    """-> {"answer", "citations", "grounded", "steps", "chunks", "source"}; `chunks` (the passages used) and `source` ("m2" | "sample" | "none") are for the caller, not part of the contract."""
    lang = language or answer_language(question)
    source = "given"
    if chunks is None:
        chunks, source = search_with_source(room_id, question, DETAIL_TOP_K if wants_detail(question) else TOP_K, topic_id=topic_id, detail=wants_detail(question))
    not_found = {"answer": NOT_FOUND[lang], "citations": [], "grounded": False, "steps": [], "chunks": chunks, "source": source}
    if not chunks or (check_related and not related(question, chunks)):
        return not_found
    res = llm.ask_json(_prompt(question, chunks, lang), max_tokens=2000) if wants_detail(question) else llm.ask_json(_prompt(question, chunks, lang))
    if res is None:                                                # no LLM: show the best passage itself, with its source
        best = chunks[0]
        return {"answer": best["text"], "citations": _citations([best]), "grounded": True, "steps": [], "chunks": [best], "source": source,
                "citation_check": cite.verify(_citations([best]), chunks)}
    used = [i for i in res.get("used", []) if isinstance(i, int) and 1 <= i <= len(chunks)]
    answer = str(res.get("answer", "")).strip()
    if not (res.get("grounded") and answer and used):
        return not_found
    steps = [str(s).strip() for s in (res.get("steps") or []) if str(s).strip()][:6]
    used_chunks = [chunks[i - 1] for i in dict.fromkeys(used)]
    check = cite.verify(_citations(used_chunks), chunks, answer)           # every citation must be a passage M2 really returned
    if not check["ok"]:                                                    # unsupported by its own sources: flag it, do not show it
        return {**not_found, "citation_check": check, "unsupported_answer": answer}
    return {"answer": answer, "citations": check["valid"], "grounded": True, "steps": steps, "chunks": used_chunks, "source": source,
            "citation_check": check}

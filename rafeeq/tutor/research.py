"""Level 2: plan 2-3 search queries, then write ONE study file (overview, key ideas, worked example, sources).

Sources come only from the real search results: the model refers to them as [1]..[5], and the Sources section is built by
code, so a link can never be invented.
"""
from __future__ import annotations

import re
from datetime import datetime

from . import llm, memory
from . import study_store as store


def _is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum("؀" <= c <= "ۿ" for c in letters) > len(letters) / 2


DEFAULT_PAGES, MAX_PAGES, WORDS_PER_PAGE = 1, 15, 450
_PAGES_RE = re.compile(r"(\d+|[٠-٩]+)\s*(?:pages?|صفح[ةات])", re.IGNORECASE)


def requested_pages(question: str) -> int | None:
    """'a 10 page study file' / 'في ٣ صفحات' -> 10 / 3 (clamped to 1..MAX_PAGES); None if the question names no number."""
    m = _PAGES_RE.search(question)
    if not m:
        return None
    n = int(m.group(1).translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
    return max(1, min(n, MAX_PAGES))


def pages_for(question: str, chosen: int | None = None) -> int:
    """The number in the question wins, then the length picked in the form, then the default (1 page)."""
    return requested_pages(question) or chosen or DEFAULT_PAGES


def sources_for(pages: int) -> int:
    return max(5, min(10, pages + 3))                          # longer files need more material


def plan_queries(question: str, topic: str | None = None) -> list[str]:
    field = memory.search_field()
    focus = f"\nThe student studies {field}: put the field in at least two queries so the results fit what they study." if field else ""
    res = llm.ask_json(
        f"""A student asked: {question}{focus}
Write 2 or 3 short web search queries (max 8 words each) that will find reliable explanations for a student.
Use different angles: (1) clear explanation, (2) worked example, optionally (3) common mistakes.
If the question is Arabic, make query 1 Arabic and the others English.
Return JSON only: {{"queries": ["...", "..."]}}""")
    queries: list[str] = []
    if isinstance(res, dict):
        for q in res.get("queries", []):
            q = re.sub(r"\s+", " ", str(q)).strip()
            if q and q.lower() not in [x.lower() for x in queries]:
                queries.append(q)
    if not queries:                                            # no AI model: still do something sensible
        queries = [f"{question.strip()} {field}".strip(), f"{question.strip()} worked example"]
    return queries[:3]


def _numbered(sources: list[dict]) -> str:
    return "\n".join(f"[{s['n']}] {s['title']}\n{s['content'][:900]}" for s in sources)


def _n_sections(pages: int) -> int:
    return max(2, pages)                                       # one section per page: each call stays short enough for a small model


def _section_prompt(question: str, heading: str, outline: list[str], sources: list[dict], words: int, lang: str) -> str:
    src = _numbered(sources)
    return f"""You are writing one section of a study file for a student. Overall question: {question}
Whole outline: {' | '.join(outline)}
Write ONLY the section: "{heading}".

Sources (numbered):
{src}

Return JSON only: {{"body": "..."}}
Rules:
- Write in {lang}, in simple words, about {words} words (at least {int(words * 0.8)}). Several paragraphs separated by a blank line;
  you may use short "- " bullet lists and **bold** for key terms. Add a small example when it helps.
- Use the sources for facts and cite them like [1]; you may also explain with your general knowledge, but do not invent links.
- Do not repeat the heading and do not cover the other sections."""


def _prompt(question: str, sources: list[dict], pages: int = 1, lang: str = "English") -> str:
    src = _numbered(sources)
    long = pages > 1
    n = _n_sections(pages)
    sections = f', "outline": ["heading 1", ..., "heading {n}"]' if long else ""
    size = (f"- outline: exactly {n} short section headings that together cover the topic in depth, in a sensible teaching order "
            f"(basics first, then details, pitfalls, practice). The sections themselves are written later.\n" if long else "")
    return f"""You write a study file for a student, about {pages} page{'s' if long else ''} long. Question: {question}

Sources (numbered):
{src}

Return JSON only:
{{"title": "...", "overview": "...", "key_ideas": ["...", "..."], "worked_example": {{"problem": "...", "steps": ["...", "..."], "answer": "..."}}{sections}}}
Rules:
{size}
- Write in {lang}, in simple words. overview: 3-4 sentences. key_ideas: 4-6 short bullets.
- Facts must come from the sources; cite them like [1] or [2]. Do not invent facts or links.
- worked_example: one small example that fits the question, with clear numbered steps.
- title: max 8 words."""


def to_markdown(question: str, topic: str | None, data: dict, sources: list[dict], degraded: bool = False) -> str:
    ex = data.get("worked_example") or {}
    lines = [f"# {data['title']}", "", f"*Question:* {question}  ", f"*Date:* {datetime.now():%Y-%m-%d}", "",
             "## Overview", data["overview"], "", "## Key ideas"]
    lines += [f"- {k}" for k in data["key_ideas"]]
    for sec in data.get("sections") or []:
        lines += ["", f"## {sec['heading']}", sec["body"]]
    lines += ["", "## Worked example"]
    if ex.get("problem"):
        lines += [f"**Problem:** {ex['problem']}", ""]
        lines += [f"{i}. {s}" for i, s in enumerate(ex.get("steps", []), 1)]
        if ex.get("answer"):
            lines += ["", f"**Answer:** {ex['answer']}"]
    else:
        lines += ["_The worked example needs the AI model (Ollama or Gemini). Ask again when it is running._"]
    lines += ["", "## Sources"] + [f"{s['n']}. [{s['title']}]({s['url']})" for s in sources]
    if degraded:
        lines += ["", "_Draft built from search snippets because the AI model was not reachable._"]
    return "\n".join(lines) + "\n"


def _write_sections(question: str, outline: list[str], sources: list[dict], pages: int, lang: str = "English") -> list[dict]:
    # small models write ~25% less than asked, so ask for more; the overview, key ideas and example already give ~200 words
    words = max(250, int((pages * WORDS_PER_PAGE - 200) * 1.25) // len(outline))
    out = []
    for h in outline:
        r = llm.ask_json(_section_prompt(question, h, outline, sources, words, lang), max_tokens=min(8000, words * 4 + 500))
        body = str(r.get("body") or "").strip() if isinstance(r, dict) else ""
        if body:
            out.append({"heading": h, "body": body})
    return out


def build_study_file(question: str, room_id: str | None, sources: list[dict], pages: int = 1, lang: str | None = None, user_id: str | None = None) -> dict:
    """-> {"id", "path", "title", "markdown", "degraded"}. pages > 1: first the plan, then one model call per section."""
    name = {"ar": "Arabic", "en": "English"}.get(lang or "") or ("Arabic" if _is_arabic(question) else "English")     # the site's language
    res = llm.ask_json(_prompt(question, sources, pages, name), max_tokens=2500)
    degraded = not (isinstance(res, dict) and res.get("overview") and res.get("key_ideas"))
    if degraded:
        res = {"title": question.strip()[:60], "overview": " ".join(s["content"][:220] for s in sources[:2]),
               "key_ideas": [f"{s['content'][:160]} [{s['n']}]" for s in sources], "worked_example": {}}
    else:
        res["title"] = str(res.get("title") or question)[:80]
        res["key_ideas"] = [str(k) for k in res["key_ideas"]][:8]
        outline = [str(h).strip() for h in res.get("outline") or [] if str(h).strip()][:_n_sections(pages)]
        if pages > 1 and outline:
            res["sections"] = _write_sections(question, outline, sources, pages, name)
    md = to_markdown(question, None, res, sources, degraded)
    saved = store.save_study_file(room_id, question, res["title"], md,
                                  [{k: s[k] for k in ("n", "title", "url")} for s in sources], user_id)
    return {**saved, "markdown": md, "degraded": degraded, "words": len(md.split())}

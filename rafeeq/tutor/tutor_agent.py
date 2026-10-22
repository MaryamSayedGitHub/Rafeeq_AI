"""The tutor agent that sits under M1's Copilot (the orchestrator): everything a student does with M3, behind ONE entry point.

  Copilot (M1, orchestrator) ──> act(room, action, ...) ──> route ─┬─ ask ········ Level 1: M2 answer + citations + 3 MCQ + "go deeper"      (agent.run_qa)
                                                                   ├─ explain ···· a photo's text: M2 first, then the web right away          (agent.run_qa x2)
                                                                   ├─ research ··· Level 2: Tavily -> study file (saved)                     (agent.run_qa)
                                                                   ├─ quiz ······· 3 MCQ, only when asked: M4's if the answer came from M2, else from the study file  (quiz.py)
                                                                   ├─ summary ···· 5-7 bullets of a study file                                (extras.py)
                                                                   ├─ cards ······ flashcards of a study file                                 (extras.py)
                                                                   ├─ files / file  the room's saved study files                              (study_store.py)
                                                                   ├─ rewrite ···· "simpler" / "example" / "compare" under an answer          (extras.py)
                                                                   ├─ help ······· a math / code problem: 3 hints, one per click, then the solution (help.py)
                                                                   ├─ hint / solution                                                         (help.py)
                                                                   └─ check ······ handwriting steps: ok | check | error, one hint each       (agent.run_check)

Every tool already exists in this package; this module only decides which one a request needs ("auto": from the text) and gives every
tool the same answer shape: {"action", "ok", "error", "result"}. A tool that fails returns ok=False with the reason, never an exception.
"""
from __future__ import annotations

import logging
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import extras, lookup, memory, quiz, safety, scope, study_store
from . import help as tutor_help
from .agent import run_check, run_qa
from .answer import _DETAIL, guess_language, keeps_english_terms
from .grading import classify
from .research import WORDS_PER_PAGE
from .search import search_with_source

log = logging.getLogger(__name__)
ACTIONS = ("auto", "ask", "explain", "research", "quiz", "summary", "cards", "files", "file", "rewrite", "help", "hint", "solution", "check", "teacher")
FILE_ACTIONS = ("quiz", "summary", "cards", "file")


class TState(TypedDict, total=False):
    room_id: str
    user_id: str | None
    action: str
    text: str
    language: str
    params: dict
    route: str
    result: dict
    error: str | None


# ----------------------------------------------------------------------------- the answer shape of Ask (shared with POST /rooms/{id}/ask)
def ask_payload(st: dict, level: int) -> dict:
    """The agent's state after run_qa -> the AskResponse dict."""
    from .quiz import passages_from_markdown
    a = st.get("answer") or {}
    sf = st.get("study_file")
    from_m2 = level == 1 or st.get("fallback") == "m2"          # a Level 2 that fell back to M2 answers like Level 1
    study = None
    if sf:
        study = {"id": sf["id"], "title": sf["title"], "markdown": sf["markdown"], "words": sf["words"], "degraded": bool(sf.get("degraded")),
                 "sources": [{"n": s["n"], "title": s["title"], "url": s["url"]} for s in st["sources"]], "queries": st["queries"]}
    overview = next((p["text"] for p in passages_from_markdown(sf["markdown"]) if p["title"].lower() == "overview"), sf["title"]) if sf else ""
    mcqs = [{"question": q["question"], "options": q["options"], "answer": q["answer"], "explanation": q.get("explanation", "")}
            for q in (st.get("quiz") or {}).get("questions", [])]
    return {"question_id": st["question_id"], "topic_id": st["topic_id"], "level": level,
            "answer": a.get("answer", "") if from_m2 else overview,
            "citations": a.get("citations", []) if from_m2 else [],
            "grounded": bool(a.get("grounded")) if from_m2 else bool(sf),
            "steps": a.get("steps", []) if from_m2 else [],
            "citation_support": (a.get("citation_check") or {}).get("support", "none") if from_m2 else "none",
            "answer_flagged": bool(a.get("unsupported_answer")) if from_m2 else False,
            "offer_level2": bool(st.get("offer_level2")), "material_source": a.get("source", "web") if from_m2 else "web",
            "language": st.get("language") or "ar", "web_sources": [],
            "study_file": study, "quiz": mcqs, "error": st.get("error"), "event_delivered": bool(st.get("event_delivered"))}


def _web_term(s: "TState", out: dict, photo: bool) -> bool:
    """M2 had nothing: answer from the web right here, short, with its sources, and NO study file (Level 2 is the deeper file).
    A question is searched as it is; a photo only when it holds a short term (anything longer becomes a study file)."""
    if not photo and lookup.is_course_specific(s["text"]):          # only the teacher knows: no web answer, the Copilot files it in the teacher's inbox (grounded=False)
        out.update(answer="", grounded=False, offer_level2=False, web_sources=[], material_source="none", needs_teacher=True)
        return False
    if not scope.web_allowed(s["room_id"]):                          # the teacher chose "material only": the question goes to the teacher
        out.update(answer="", grounded=False, offer_level2=False, web_sources=[], material_source="none", needs_teacher=True, web_off=True)
        return False
    if not photo and not scope.in_subject(s["room_id"], s["text"]):  # another subject altogether: no search, no teacher, back to the course
        out.update(answer=scope.off_topic_reply(s["room_id"], out["language"]), grounded=False, offer_level2=False, web_sources=[],
                   citations=[], steps=[], material_source="none", off_topic=True)
        return False
    term = lookup.term_of(s["text"], photo)
    term = re.sub(r"\s+", " ", _DETAIL.sub(" ", term)).strip() if term else term                  # "in detail" asks for a longer answer, it is not part of what is searched
    query = term or ("" if photo else re.sub(r"\s+", " ", _DETAIL.sub(" ", s["text"])).strip()[:150])
    if not query:
        return False
    try:
        found = lookup.web_lookup(query, s["text"], out["language"], keeps_english_terms(s["text"], out["language"]))
    except safety.Blocked as b:
        out["error"] = b.message(s.get("language") or out["language"])
        return False
    if not found:
        return False
    lead = ("لم أجد هذا في المنهج، فبحثتُ في النت:" if out["language"] == "ar" else "I did not find this in the course material, so I searched the web:")
    out.update(answer=lead + "\n\n" + found["answer"], grounded=True, citations=[], steps=[], web_sources=found["web_sources"], material_source="web",
               offer_level2=True, citation_support="none", answer_flagged=False, error=None)
    study_store.update_question(out["question_id"], grounded=True, engine="web", level=2)
    try:                                                            # the teacher sees what the files did not cover, and can confirm or replace the answer
        from rafeeq.copilot import clients
        clients.optional("log_web_answer", s["room_id"], s["text"], s.get("user_id"), {"answer": out["answer"], "web_sources": out["web_sources"]})
    except Exception as e:                                          # noqa: BLE001  never breaks the answer
        log.warning("could not log the web answer for the teacher: %s", type(e).__name__)
    return True


def _file_summary(f: dict) -> dict:
    words = len(f["markdown"].split()) if f.get("markdown") else 0
    return {"id": f["id"], "title": f["title"], "question": f.get("question") or "", "ts": f["ts"], "words": words,
            "pages": max(1, round(words / WORDS_PER_PAGE)) if words else None}


# ----------------------------------------------------------------------------- nodes
def n_route(s: TState) -> TState:
    action = s.get("action", "auto")
    if action == "auto":
        text = s.get("text", "")
        lines = [l for l in text.splitlines() if l.strip()]
        action = "help" if classify(lines) in ("math", "code") else "ask"          # a problem gets hints; anything else is answered from the material
    elif action == "explain":                                                      # a photo: a problem gets hints, anything else M2 then the web
        action = "help" if classify([l for l in s.get("text", "").splitlines() if l.strip()]) in ("math", "code") else "explain"
    return {"route": action}


def n_ask(s: TState) -> TState:
    p = s.get("params", {})
    level = 2 if s["route"] == "research" else 1
    if level == 2 and not scope.web_allowed(s["room_id"]):          # "material only" room: no web research, the material answers (Level 1)
        level = 1
        p = {**p, "lookup": False}
        web_off = True
    else:
        web_off = False
    st = run_qa(s["room_id"], s["text"], level, s.get("user_id"), s.get("language") if level == 2 else None, p.get("pages") if level == 2 else None,
                p.get("topic_id"), p.get("parent_question_id"), notify=p.get("notify", True), with_quiz=bool(p.get("with_quiz")),
                check_related=level == 1)       # M2 always returns its nearest passages: they must really be about the question
    out = ask_payload(st, level)
    if level == 2 and out["study_file"]:
        out["study_file"]["pages"] = max(1, round(out["study_file"]["words"] / WORDS_PER_PAGE))
        out["study_file"]["stats"] = st.get("search_stats") or {}
    if level == 1 and lookup.mentions_course_matter(s["text"]):      # an exam date, a grade...: the course page may match the words, but only the teacher knows
        out.update(answer="", citations=[], steps=[], grounded=False, offer_level2=False, web_sources=[], material_source="none", needs_teacher=True)
    elif level == 1 and not out["grounded"] and p.get("lookup", True):
        _web_term(s, out, photo=False)
    if not scope.web_allowed(s["room_id"]):
        out["offer_level2"] = False                                  # no "go deeper on the web" button in a material-only room
    if web_off:
        out["error"] = scope.WEB_OFF.get(out.get("language") or "ar", scope.WEB_OFF["ar"])
    return {"result": out, "error": None if (out["grounded"] or level == 1) else (out["error"] or "no study file")}


def n_explain(s: TState) -> TState:
    """What is in the photo? Ask M2 first (Level 1). The answer is kept in `result`; when M2 has nothing, n_explain_web goes on to the web."""
    p = s.get("params", {})
    st = run_qa(s["room_id"], s["text"], 1, s.get("user_id"), None, None, p.get("topic_id"), None, notify=p.get("notify", True), check_related=True)
    out = ask_payload(st, 1)
    if not out["grounded"] and _web_term(s, out, photo=True):       # a word or a short term in the photo: the web answers it, no file
        out["answered_by"] = "web"
        return {"result": out}
    out["answered_by"] = "m2" if out["grounded"] else "none"
    return {"result": out}


def n_explain_web(s: TState) -> TState:
    """M2 had nothing: search the web NOW (no button) and answer from what it found, with the sources, and keep it as a study file."""
    p = s.get("params", {})
    first = s["result"]
    if not scope.web_allowed(s["room_id"]):                          # "material only" room: say so instead of searching
        msg = scope.WEB_OFF.get(s.get("language") or "ar", scope.WEB_OFF["ar"])
        return {"result": {**first, "error": msg}, "error": msg}
    st = run_qa(s["room_id"], s["text"], 2, s.get("user_id"), s.get("language"), p.get("pages") or 1, None, first["question_id"], notify=False)   # the file: site language
    out = ask_payload(st, 2)
    if out["study_file"]:
        out["study_file"]["pages"] = max(1, round(out["study_file"]["words"] / WORDS_PER_PAGE))
        out["study_file"]["stats"] = st.get("search_stats") or {}
        out["answered_by"] = "web"
        return {"result": out}
    return {"result": {**first, "error": out["error"] or "the web had nothing either"}, "error": out["error"] or "the web had nothing either"}


def _file(s: TState) -> dict | None:
    fid = s.get("params", {}).get("study_file_id")
    f = study_store.get_study_file(int(fid)) if fid else None
    return f if study_store.owns(f, s["room_id"], s.get("user_id")) else None                 # only the student's own files


def n_file_tool(s: TState) -> TState:
    lang = s.get("language") or "ar"
    if s["route"] == "files":
        return {"result": {"files": [_file_summary(f) for f in study_store.list_study_files(s["room_id"], s.get("user_id"))]}}
    if s["route"] == "quiz" and not s.get("params", {}).get("study_file_id"):
        return _quiz_from_material(s)
    f = _file(s)
    if not f:
        return {"result": {}, "error": "study file not found"}
    if s["route"] == "file":
        return {"result": {**_file_summary(f), "markdown": f["markdown"]}}
    if s["route"] == "summary":
        points = extras.summary(f, lang)
        return {"result": {"points": points or []}, "error": None if points else "the model is not reachable"}
    if s["route"] == "cards":
        cards = extras.flashcards(f, lang)
        return {"result": {"cards": cards or []}, "error": None if cards else "the model is not reachable"}
    qz = quiz.quiz_for_study_file(f, language=lang)                 # a study file's tools: the site language
    qs = [{"question": q["question"], "options": q["options"], "answer": q["answer"], "explanation": q.get("explanation", "")} for q in qz["questions"]]
    return {"result": {"questions": qs, "source": qz["source"]}, "error": None if qs else "no quiz could be made"}


def _quiz_from_material(s: TState) -> TState:
    """The answer came from M2: the 3 questions come from M4 for that topic (quiz.get_mcqs asks M4 first, then the bank, then the model on the passages)."""
    p = s.get("params", {})
    question = p.get("question") or s.get("text", "")
    topic = p.get("topic_id") or None
    chunks, _ = search_with_source(s["room_id"], question, topic_id=topic if topic and topic != "other" else None)
    qz = quiz.get_mcqs(question, topic, chunks, room_id=s["room_id"], language=s.get("language"))      # the answer's language: the button sends it
    qs = [{"question": q["question"], "options": q["options"], "answer": q["answer"], "explanation": q.get("explanation", "")} for q in qz["questions"]]
    return {"result": {"questions": qs, "source": qz["source"]}, "error": None if qs else "no quiz could be made"}


def n_rewrite(s: TState) -> TState:
    p = s.get("params", {})
    text = extras.rewrite(p.get("mode", ""), p.get("question", ""), s.get("text", ""), s.get("language") or "ar", p.get("other", ""))
    return {"result": {"text": text or ""}, "error": None if text else "the model is not reachable"}


def n_help(s: TState) -> TState:
    lang = s.get("language") or guess_language(s.get("text", ""))
    lines = [l for l in s["text"].splitlines() if l.strip()]
    kind = "code" if classify(lines) == "code" else "math"
    key = tutor_help.prepare(s["text"], kind, lang)
    return {"result": {"key": key, "kind": kind, "language": lang}}


def n_hint(s: TState) -> TState:
    p = s.get("params", {})
    data = tutor_help.get(p.get("key", ""))
    if data is None:
        return {"result": {}, "error": "unknown help key"}
    if s["route"] == "solution":
        return {"result": {"steps": data["solution"], "final_answer": data["final_answer"], "llm_ok": data["llm_ok"]}}
    n = max(1, min(int(p.get("n", 1)), 3))
    return {"result": {"n": n, "hint": data["hints"][n - 1], "next": "hint" if n < 3 else "solution"}}


def n_teacher(s: TState) -> TState:
    """"Ask my teacher": file the question in the teacher's inbox (M5). The teacher's reply comes back to the student's page, and is reused when anyone asks the same."""
    from rafeeq.copilot import clients
    ticket = clients.optional("escalate", s["room_id"], s["text"], s.get("user_id"))
    if not ticket:
        return {"result": {"escalated": False}, "error": "the teacher's inbox is not available"}
    return {"result": {"escalated": True, "question_id": ticket["question_id"]}}


def n_check(s: TState) -> TState:
    p = s.get("params", {})
    st = run_check(s["room_id"], s["text"], p.get("file_id", ""), s.get("user_id"), s.get("language"), p.get("confidence"))
    work = st.get("work")
    if not work:
        return {"result": {}, "error": st.get("error") or "check failed"}
    return {"result": {**work, "event_delivered": bool(st.get("event_delivered"))}}


def build_graph():
    g = StateGraph(TState)
    for name, fn in [("route", n_route), ("ask", n_ask), ("explain", n_explain), ("explain_web", n_explain_web), ("file_tool", n_file_tool), ("rewrite", n_rewrite), ("help", n_help),
                     ("hint", n_hint), ("check", n_check), ("teacher", n_teacher)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["route"], {
        "ask": "ask", "explain": "explain", "research": "ask", "files": "file_tool", "file": "file_tool", "quiz": "file_tool", "summary": "file_tool", "cards": "file_tool",
        "rewrite": "rewrite", "help": "help", "hint": "hint", "solution": "hint", "check": "check", "teacher": "teacher"})
    g.add_conditional_edges("explain", lambda s: "done" if s["result"]["grounded"] else "web", {"done": END, "web": "explain_web"})
    for name in ("ask", "explain_web", "file_tool", "rewrite", "help", "hint", "check", "teacher"):
        g.add_edge(name, END)
    return g.compile()


_GRAPH = None


def act(room_id: str, action: str = "auto", text: str = "", user_id: str | None = None, language: str | None = None, **params) -> dict:
    """-> {"action": the tool that ran, "ok": bool, "error": str | None, "result": dict}. Never raises for a tool failure."""
    global _GRAPH
    if action not in ACTIONS:
        return {"action": action, "ok": False, "error": f"unknown action {action!r}", "result": {}}
    _GRAPH = _GRAPH or build_graph()
    if action in ("auto", "ask", "explain", "research") and text:
        text = memory.standalone(text)                              # "and its function?" -> "what is the function of the mitochondria?"
    lang = language or (guess_language(text) if text else "ar")
    try:
        st = _GRAPH.invoke({"room_id": room_id, "user_id": user_id, "action": action, "text": text.strip(), "language": lang,
                            "params": {k: v for k, v in params.items() if v is not None}})
    except Exception as e:                                         # noqa: BLE001  one broken tool must not take the chat down
        log.warning("tutor agent %s failed: %s", action, type(e).__name__)
        return {"action": action, "ok": False, "error": f"{type(e).__name__}", "result": {}}
    return {"action": st["route"], "ok": not st.get("error"), "error": st.get("error"), "result": st.get("result") or {}}

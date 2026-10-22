"""The tutor agent (LangGraph): one student question, from the question to the M4 event.

  tag_question ─┬─ handwriting text ─> check_work ─────────────────────────────────────────> notify_m4 ─> END
                ├─ level 1 ─> answer ─(M2 down: retry once)─> quiz ─> decide_next ───────────┐
                └─ level 2 ─> plan_queries ─> web_search ─ok─> write_study_file ─────────────┤
                                                   │  └─ transient error: retry once         │
                                                   └─ failed for good / no file ─> fallback_to_m2 ─> answer ─> quiz ─> decide_next
                                                                                              v
                                                                               log_question ─> notify_m4 ─> END
Routing is conditional and every retry is capped (MAX_RETRIES): Tavily is only called for Level 2 (the student's choice), a Tavily or
study-file failure falls back to what M2 knows instead of an empty answer, and a quota / key problem is never retried.

Tools (all already in this package, nothing is duplicated):
  answer          rafeeq/tutor/answer.py     grounded answer from M2's passages, citations verified against them (citations.py)
  M2 retrieve     rafeeq/tutor/m2.py         POST /search (and topics)
  record_event    rafeeq/tutor/events.py     POST /events to M4
  web search      rafeeq/tutor/web_search.py Tavily + cache + monthly credit guard
  study writer    rafeeq/tutor/research.py   overview, key ideas, worked example, sources -> saved, reopenable
  check work      rafeeq/tutor/check_work.py handwriting text -> every step ok | check | error, one hint each (uncertain lines are asked, not judged)
decide_next: when Level 1 is not grounded the agent offers Level 2 by itself (the student presses one button, so the free
search quota is never spent without them).
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import quiz, research, safety, study_store, web_search
from .answer import answer_language, guess_language, tutor_answer
from .check_work import check_work_detail
from .events import record_handwriting, send_question_event
from .topics import OTHER, tag_topic

log = logging.getLogger(__name__)
MAX_RETRIES = 1                  # one more try for a transient failure, never more
PERMANENT = ("not set", "rejected", "monthly search limit", "quota", "rate limit", "no usable sources", "not appropriate", "غير مناسب")   # retrying these only wastes the free quota


class State(TypedDict, total=False):
    room_id: str
    user_id: str | None
    question: str
    level: int
    language: str
    pages: int                    # level 2: study file length in pages
    topic_filter: str | None      # the student chose a topic: search only in it
    parent_question_id: int       # level 2 pressed after an ungrounded level 1: update that row, no duplicate
    topic_id: str
    answer: dict
    quiz: dict
    offer_level2: bool
    queries: list
    sources: list
    search_stats: dict
    study_file: dict
    error: str | None
    question_id: int
    event_delivered: bool
    work_text: str                # handwriting: the text read from the photo (routes to check_work)
    file_id: str
    confidence: float
    work: dict | None             # check_work_detail result
    m2_tries: int
    web_tries: int
    fallback: str                 # "m2": Level 2 could not run, the answer comes from M2
    route: str                    # "check_work" | "answer" | "web"
    trace: list                   # every node the agent went through (tests, debugging)
    check_related: bool           # a photo's text: the passages must be about the same subject, not just the nearest ones
    with_quiz: bool               # True: also make the 3 MCQ (default: only when the student presses the quiz button)
    notify: bool                  # False: log the question but do not send the M4 event (the caller sends it)


def _t(s: State, what: str) -> list:
    return [*s.get("trace", []), what]


def route_of(s: State) -> str:
    """The first decision: handwriting text -> check_work, Level 2 -> web research, otherwise M2 answer."""
    if s.get("work_text"):
        return "check_work"
    return "web" if s.get("level", 1) == 2 else "answer"


def n_tag(s: State) -> State:
    text = s.get("work_text") or s["question"]
    return {"topic_id": s.get("topic_filter") or tag_topic(s["room_id"], text), "route": route_of(s), "trace": _t(s, "tag_question")}


def n_answer(s: State) -> State:
    res = tutor_answer(s["room_id"], s["question"], s.get("language"), topic_id=s.get("topic_filter"), check_related=bool(s.get("check_related")))
    topic = s["topic_id"]
    if topic == OTHER and res["chunks"] and res["chunks"][0].get("topic_id"):    # the passages that answered know the topic better
        topic = tag_topic(s["room_id"], s["question"], res["chunks"])
    return {"answer": res, "topic_id": topic, "m2_tries": s.get("m2_tries", 0) + 1, "trace": _t(s, "answer")}


def after_answer(s: State) -> str:
    """M2 unreachable (no passages, source "none") -> try once more; an honest "nothing matches" from M2 is an answer, not a failure."""
    a = s["answer"]
    return "retry" if a.get("source") == "none" and not a["chunks"] and s["m2_tries"] <= MAX_RETRIES else "go"


def n_check_work(s: State) -> State:
    try:
        work = check_work_detail(s["room_id"], s.get("file_id") or "", s["work_text"], s.get("language"), s.get("confidence"))
    except Exception as e:                                         # noqa: BLE001  never an unhandled error for the student
        log.warning("check_work failed: %s", type(e).__name__)
        return {"error": f"check failed ({type(e).__name__})", "work": None, "trace": _t(s, "check_work:failed")}
    return {"work": work, "trace": _t(s, "check_work")}


def n_quiz(s: State) -> State:
    a = s["answer"]
    if not s.get("with_quiz") or not a["grounded"]:               # questions are made when the student asks for them, never by themselves
        return {"quiz": {"questions": [], "source": "none"}}
    return {"quiz": quiz.get_mcqs(s["question"], s["topic_id"], a["chunks"], room_id=s["room_id"])}


def n_decide(s: State) -> State:
    return {"offer_level2": not s["answer"]["grounded"]}


def n_plan(s: State) -> State:
    return {"queries": research.plan_queries(s["question"]), "trace": _t(s, "plan_queries")}


def n_search(s: State) -> State:
    pages = research.pages_for(s["question"], s.get("pages"))
    top = research.sources_for(pages)
    base = {"web_tries": s.get("web_tries", 0) + 1, "trace": _t(s, "web_search")}
    try:
        res = web_search.multi_search(s["queries"], top=top, per_query=max(5, top))
    except Exception as e:                                         # noqa: BLE001  SearchError or a network error: both end in the retry / fallback below
        msg = e.message(s.get("language") or "en") if isinstance(e, safety.Blocked) else (str(e) or type(e).__name__)        # an illegal / 18+ search is refused in the student's language
        if not isinstance(e, safety.Blocked):
            log.warning("web search failed: %s", msg[:200])                # the page shows the student a plain sentence; the reason is here
        return {**base, "sources": [], "search_stats": {"cache_hits": 0, "api_calls": 0, "errors": [msg]}, "error": msg}
    stats = {k: res[k] for k in ("cache_hits", "api_calls", "errors")}
    if not res["sources"]:
        return {**base, "sources": [], "search_stats": stats, "error": (res["errors"] or ["no usable sources"])[0]}
    return {**base, "sources": res["sources"], "search_stats": stats, "error": None}


def after_search(s: State) -> str:
    """ok -> write the file | transient failure not yet retried -> search again | failed for good (key, quota, nothing found) -> M2 fallback."""
    err = s.get("error")
    if not err:
        return "ok"
    permanent = any(w in err.lower() for w in PERMANENT)
    return "retry" if not permanent and s.get("web_tries", 0) <= MAX_RETRIES else "fallback"


def n_write(s: State) -> State:
    pages = research.pages_for(s["question"], s.get("pages"))
    try:
        return {"study_file": research.build_study_file(s["question"], s["room_id"], s["sources"], pages, s.get("language"), s.get("user_id")), "trace": _t(s, "write_study_file")}
    except Exception as e:                                         # noqa: BLE001  disk / store problem: do not lose the answer
        log.warning("study file failed: %s", type(e).__name__)
        return {"error": f"study file failed ({type(e).__name__})", "trace": _t(s, "write_study_file:failed")}


def n_fallback(s: State) -> State:
    """Level 2 could not run: the student still gets what M2 knows (grounded, cited), and the reason stays in "error"."""
    return {"fallback": "m2", "trace": _t(s, "fallback_to_m2")}


def n_log(s: State) -> State:
    level = s.get("level", 1)
    sf = s.get("study_file") or {}
    from_m2 = level == 1 or s.get("fallback") == "m2"
    grounded = bool(s.get("answer", {}).get("grounded")) if from_m2 else bool(sf)
    engine = s.get("answer", {}).get("source", "m2") if from_m2 else "web"
    if s.get("parent_question_id"):
        study_store.update_question(s["parent_question_id"], level=2, grounded=grounded, engine=engine, study_file_id=sf.get("id"), topic_id=s["topic_id"])
        return {"question_id": s["parent_question_id"]}
    qid = study_store.log_question(s["room_id"], s["question"], "question", grounded, s["topic_id"], level, s.get("user_id"), engine, sf.get("id"))
    return {"question_id": qid}


def n_notify(s: State) -> State:
    if s.get("notify") is False:
        return {"event_delivered": False}
    if s.get("work_text"):                                         # handwriting: "understood" only when every step is ok and nothing was unreliable
        w = s.get("work") or {}
        understood = bool(w.get("items")) and not w.get("needs_help")
        wrong = next((i for i in w.get("items", []) if i["status"] != "ok"), None)
        text = (s["work_text"] if understood or not wrong else wrong["text"])[:300]
        items = [i for i in w.get("items", []) if i.get("status") in ("ok", "check", "error")]
        record_handwriting(s.get("user_id"), s["room_id"], s["topic_id"], sum(i["status"] == "ok" for i in items), len(items))    # the result moves the topic's mastery
        return {"event_delivered": bool(send_question_event(s.get("user_id"), s["room_id"], s["topic_id"], text, kind="understood" if understood else "question"))}
    return {"event_delivered": bool(send_question_event(s.get("user_id"), s["room_id"], s["topic_id"], s["question"]))}


def build_graph():
    g = StateGraph(State)
    for name, fn in [("tag_question", n_tag), ("answer", n_answer), ("quiz", n_quiz), ("decide_next", n_decide), ("plan_queries", n_plan),
                     ("web_search", n_search), ("write_study_file", n_write), ("fallback_to_m2", n_fallback), ("check_work", n_check_work),
                     ("log_question", n_log), ("notify_m4", n_notify)]:
        g.add_node(name, fn)
    g.add_edge(START, "tag_question")
    g.add_conditional_edges("tag_question", lambda s: s["route"], {"answer": "answer", "web": "plan_queries", "check_work": "check_work"})
    g.add_conditional_edges("answer", after_answer, {"retry": "answer", "go": "quiz"})
    g.add_edge("quiz", "decide_next")
    g.add_edge("decide_next", "log_question")
    g.add_edge("plan_queries", "web_search")
    g.add_conditional_edges("web_search", after_search, {"ok": "write_study_file", "retry": "web_search", "fallback": "fallback_to_m2"})
    g.add_conditional_edges("write_study_file", lambda s: "fallback" if s.get("error") else "ok", {"ok": "log_question", "fallback": "fallback_to_m2"})
    g.add_edge("fallback_to_m2", "answer")
    g.add_edge("check_work", "notify_m4")
    g.add_edge("log_question", "notify_m4")
    g.add_edge("notify_m4", END)
    return g.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    _GRAPH = _GRAPH or build_graph()
    return _GRAPH


def run_qa(room_id: str, question: str, level: int = 1, user_id: str | None = None, language: str | None = None, pages: int | None = None,
           topic_filter: str | None = None, parent_question_id: int | None = None, notify: bool = True,
           with_quiz: bool = False, check_related: bool = False) -> State:
    """Run the agent for one question. Level 1 = grounded answer (+ quiz, + the Level 2 offer); level 2 = web research + a saved study file
    (or, when the web or the file fails, the M2 answer with state["fallback"] == "m2")."""
    state: State = {"room_id": room_id, "question": question.strip(), "level": level, "user_id": user_id,
                    "language": language or answer_language(question)}
    if pages:
        state["pages"] = pages
    if topic_filter:
        state["topic_filter"] = topic_filter
    if parent_question_id:
        state["parent_question_id"] = parent_question_id
    if not notify:
        state["notify"] = False
    if with_quiz:
        state["with_quiz"] = True
    if check_related:
        state["check_related"] = True
    return _graph().invoke(state)


def run_check(room_id: str, text: str, file_id: str = "", user_id: str | None = None, language: str | None = None, confidence: float | None = None) -> State:
    """Handwriting through the same agent: text read from a photo -> check_work -> M4 event. state["work"] is the check_work_detail result."""
    return _graph().invoke({"room_id": room_id, "question": text[:300], "work_text": text, "file_id": file_id, "user_id": user_id,
                            "language": language or guess_language(text), "confidence": confidence, "level": 1})

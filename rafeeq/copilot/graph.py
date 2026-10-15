"""The Copilot as a small graph: classify the message, then run exactly one capability.

Uses LangGraph when it is installed. If it is not, the same node functions run through a
plain dispatcher, so the app still boots (useful in CI or on a laptop without the extra).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

from rafeeq.copilot import clients
from rafeeq.copilot.router import ROUTES, classify

UNAVAILABLE_REPLY = "هذه الميزة غير متاحة بعد."


class State(TypedDict, total=False):
    room_id: str
    text: str
    file_id: str | None
    mode: str
    user_id: str | None
    history: list
    route: str
    reply: str
    payload: dict


def _mark(payload: dict, source: str) -> dict:
    if source == "sample":
        payload = {**payload, "sample": True}
    return payload


def classify_node(s: State) -> dict:
    return {"route": classify(s["text"], bool(s.get("file_id")), s.get("history", []), s.get("mode", "auto"))}


_OVERVIEW_WORDS = ("الدرس", "المحاضرة", "المحاضره", "الفصل", "الملف", "الشابتر", "لخص", "لخّص", "ملخص",
                   "lesson", "lecture", "chapter", "summar", "overview", "the file")
_NOT_COVERED_EN = "This topic is not covered in the provided course material."


def is_overview_request(text: str) -> bool:
    """"Explain the last lesson", "لخّص المحاضرة": about a whole file, not one fact in it."""
    t = (text or "").lower()
    return any(w in t for w in _OVERVIEW_WORDS)


def ask_node(s: State) -> dict:
    verified = clients.optional("find_verified", s["room_id"], s["text"])
    if verified:
        return {"reply": verified["answer"], "payload": {**verified, "verified_by_teacher": True}}

    answer, source = clients.call("tutor_answer", s["room_id"], s["text"])
    if answer is None:
        return {"reply": UNAVAILABLE_REPLY, "payload": {"unavailable": True}}

    if not answer.get("grounded", True) and source == "live" and is_overview_request(s["text"]):
        # A question about the lesson as a whole has no single passage to retrieve: explain the file instead.
        overview = clients.optional("overview", s["room_id"], s["text"])
        if overview:
            answer = overview

    if source == "live" and s.get("user_id"):
        topic_id = answer.get("topic_id") or clients.optional("topic_of", s["room_id"], answer.get("citations", []))
        if topic_id:
            clients.optional("record_event", {
                "user_id": s["user_id"], "room_id": s["room_id"], "topic_id": topic_id,
                "kind": "question", "text": s["text"], "time": datetime.now(timezone.utc).isoformat(),
            })

    if not answer.get("grounded", True):
        ticket = clients.optional("escalate", s["room_id"], s["text"], s.get("user_id"))
        payload = {**answer, "escalated": bool(ticket)}
        if ticket:
            payload["question_id"] = ticket["question_id"]
        text = answer.get("answer", "")
        if text.strip() == _NOT_COVERED_EN:
            text = ""  # the page already says it in the student's language
        return {"reply": text, "payload": _mark(payload, source)}

    return {"reply": answer["answer"], "payload": _mark(answer, source)}


def quiz_node(s: State) -> dict:
    """The chat suggests WHICH topic to practise and links to the practice page, where the quiz is
    generated. With M4 connected the topic comes from the student's profile (next_step)."""
    if clients.is_connected("next_step") and not clients.is_connected("make_quiz"):
        step, source = clients.call("next_step", s["room_id"], s.get("user_id"))
        if step is None:
            return {"reply": UNAVAILABLE_REPLY, "payload": {"unavailable": True}}
        return {"reply": "جهّزت لك اختبارًا قصيرًا.", "payload": _mark({"topic_id": step.get("topic_id")}, source)}
    quiz, source = clients.call("make_quiz", s["room_id"], None)
    if quiz is None:
        return {"reply": UNAVAILABLE_REPLY, "payload": {"unavailable": True}}
    return {"reply": "جهّزت لك اختبارًا قصيرًا.",
            "payload": _mark({"quiz_id": quiz["quiz_id"], "topic_id": quiz.get("topic_id")}, source)}


def check_work_node(s: State) -> dict:
    if not s.get("file_id"):
        return {"reply": "أرفق صورة حلّك وسأراجعه.", "payload": {"needs_photo": True}}
    result, source = clients.call("check_work", s["room_id"], s["file_id"], s["text"])
    if result is None:
        return {"reply": UNAVAILABLE_REPLY, "payload": {"unavailable": True}}
    return {"reply": result["hint"], "payload": _mark(result, source)}


def plan_node(s: State) -> dict:
    step, source = clients.call("next_step", s["room_id"], s.get("user_id"))
    if step is None:
        return {"reply": UNAVAILABLE_REPLY, "payload": {"unavailable": True}}
    return {"reply": step["suggestion"], "payload": _mark({"topic_id": step.get("topic_id")}, source)}


def clarify_node(s: State) -> dict:
    return {"reply": "وضّح لي أكثر: هل تريد شرح جزء معيّن، أم اختبارًا، أم مراجعة حلّك؟", "payload": {}}


NODES = {"ask": ask_node, "quiz": quiz_node, "check_work": check_work_node,
         "plan": plan_node, "clarify": clarify_node}


def _build():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None
    g = StateGraph(State)
    g.add_node("classify", classify_node)  # must not equal a state key, so not "route"
    for name, fn in NODES.items():
        g.add_node(name, fn)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", lambda s: s["route"], {r: r for r in ROUTES})
    for name in NODES:
        g.add_edge(name, END)
    return g.compile()


_graph = _build()


def run(state: State) -> State:
    if _graph is not None:
        return _graph.invoke(state)
    out: State = {**state, **classify_node(state)}
    out.update(NODES[out["route"]](out))
    return out

"""Lesson studio (M5, section A) and its Presentations tab.

Each case is a real lecturer brief (Arabic and English, with and without code). The REAL LangGraph lecturer graph runs
it to the approval pause (orchestrator -> research -> slides / code / quiz -> draft), exactly as the web page does,
then the draft is turned into a deck and exported to .pptx twice, as the Presentations tab does.

Deterministic checks: every requested part is there, quiz shape, the code examples the code agent marked as
verified by running them, the slide language, the slide bullet rule, the .pptx opens, the second export is version 2.
LLM-as-judge (Gemini, a different model family from the generator): quiz answered blind and compared with the key,
and a 1-5 grade of the slides (accuracy, coverage of the objectives, fit to the level)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from evals import judge, observe
from evals.core import Metric, Spec, pct
from evals.fakes import FakeStore

NAME = "lesson_studio"
DESCRIPTION = "Lecturer graph drafts from real briefs: completeness, quiz key, verified code, language, judge grade; .pptx export + versions"
LIVE = True

_BRIEFS = [
    ("py-list-comprehensions", dict(topic="Python list comprehensions", student_level="beginner", needs=["slides", "code", "quiz"],
                                    num_quiz_questions=3, learning_objectives=["write a list comprehension", "add a condition to it"])),
    ("binary-search", dict(topic="Binary search", student_level="intermediate", needs=["slides", "code", "quiz"],
                           num_quiz_questions=3, learning_objectives=["explain why it is O(log n)", "implement it iteratively"])),
    ("ar-loops", dict(topic="الحلقات التكرارية في بايثون (for و while)", language="Arabic", student_level="beginner",
                      needs=["slides", "code", "quiz"], num_quiz_questions=3,
                      learning_objectives=["التفريق بين for و while", "استخدام range"])),
    ("ar-photosynthesis", dict(topic="التمثيل الضوئي", language="Arabic", student_level="beginner", needs=["slides", "quiz"],
                               num_quiz_questions=3, learning_objectives=["معادلة التمثيل الضوئي", "دور الكلوروفيل"])),
    ("newton-laws", dict(topic="Newton's three laws of motion", student_level="intermediate", needs=["slides", "quiz"],
                         num_quiz_questions=3, learning_objectives=["state each law", "apply F = m a"])),
    ("sql-joins", dict(topic="SQL JOINs with Python's sqlite3", student_level="intermediate", needs=["slides", "code", "quiz"],
                       num_quiz_questions=3, learning_objectives=["INNER vs LEFT JOIN", "run a join from Python"])),
]


def cases() -> list[dict]:
    return [{"id": cid, "input": {"lecturer_id": "eval", "lecturer_name": "Rafeeq eval", "course_name": "Evaluation",
                                  "duration_minutes": 45, **brief}, "expected": None}
            for cid, brief in _BRIEFS]


def missing() -> str:
    return "" if os.getenv("GROQ_API_KEY") or os.getenv("LLM_PROVIDER", "").lower() not in ("", "groq") else "GROQ_API_KEY is not set"


_CONTROLLER = None


def _controller():
    global _CONTROLLER
    if _CONTROLLER is None:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        from rafeeq.teacher.lecturer.controller import GraphController
        from rafeeq.teacher.lecturer.graph import build_graph

        _CONTROLLER = GraphController(build_graph(InMemorySaver(), InMemoryStore()))   # nothing touches real data
    return _CONTROLLER


def _export(draft: dict, brief: dict, session_id: str) -> dict:
    """Presentations tab: draft -> deck -> .pptx, then a second export (must be version 2 of the first)."""
    from pptx import Presentation

    from rafeeq.teacher import assets, decks
    from rafeeq.teacher.lecturer import settings

    if not draft.get("slides"):
        return {}
    keep = settings.ASSET_DIR
    with tempfile.TemporaryDirectory() as tmp:
        settings.ASSET_DIR = Path(tmp).resolve()       # Windows temp can be a short 8.3 path (MARYAM~1)
        try:
            store = FakeStore()
            deck = decks.create_from_draft(store, "eval", session_id, draft, brief)
            first = decks.export_deck(store, "eval", deck["deck_id"])
            second = decks.export_deck(store, "eval", deck["deck_id"])
            path = assets.asset_file(first)
            pages = len(Presentation(str(path)).slides) if path else 0
            return {"opens": pages > 0, "pptx_slides": pages, "draft_slides": len(draft["slides"]),
                    "second_version": second.get("version"), "second_parent_is_first": second.get("parent_asset_id") == first["asset_id"]}
        finally:
            settings.ASSET_DIR = keep


def target(inp: dict) -> dict:
    from rafeeq.teacher.lecturer.schemas import SessionRequest

    brief = SessionRequest(**inp).model_dump()
    view = _controller().start(brief)
    draft = view.get("draft") or {}
    try:
        pptx = _export(draft, brief, view["session_id"])
    except Exception as e:                                                       # noqa: BLE001 - a failed export is a score
        pptx = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"status": view.get("status"), "agents": view.get("selected_agents", []),
            "slides": draft.get("slides", []), "quiz": draft.get("quiz", []),
            "code": [{k: c.get(k) for k in ("title", "code", "verified", "fix_attempts")} for c in draft.get("code_examples", [])],
            "pptx": pptx}


# ------------------------------------------------------------------ scorers (0..1, None = not applicable)
def complete(i, o, e):
    ok = o["status"] == "awaiting_approval"
    ok &= ("slides" not in i["needs"]) or bool(o["slides"])
    ok &= ("code" not in i["needs"]) or bool(o["code"])
    ok &= ("quiz" not in i["needs"]) or bool(o["quiz"])
    return float(ok)


def quiz_count_exact(i, o, e):
    return None if "quiz" not in i["needs"] else float(len(o["quiz"]) == i["num_quiz_questions"])


def _well_formed(q: dict) -> bool:
    opts = q.get("options") or []
    return len(opts) == 4 and len({str(x).strip() for x in opts}) == 4 and 0 <= int(q.get("answer_index", -1)) < 4


def quiz_well_formed(i, o, e):
    return None if not o["quiz"] else sum(_well_formed(q) for q in o["quiz"]) / len(o["quiz"])


def quiz_key_agreement(i, o, e):
    if not o["quiz"] or not judge.available():
        return None
    blind = judge.solve_quiz(o["quiz"])
    pairs = [(a, q.get("answer_index")) for a, q in zip(blind, o["quiz"]) if a is not None]
    return sum(a == k for a, k in pairs) / len(pairs) if pairs else None


def code_verified(i, o, e):
    marks = [c["verified"] for c in o["code"] if c.get("verified") is not None]
    return None if "code" not in i["needs"] or not marks else sum(bool(m) for m in marks) / len(marks)


def _slide_text(s: dict) -> str:
    return " ".join([s.get("title", ""), *s.get("bullets", [])])


def language_match(i, o, e):
    if not o["slides"]:
        return None
    return sum(observe.in_language(_slide_text(s), i.get("language", "English")) for s in o["slides"]) / len(o["slides"])


def bullet_rule(i, o, e):
    """The slide schema asks for 3-5 bullets of at most 12 words."""
    if not o["slides"]:
        return None
    good = [3 <= len(s.get("bullets", [])) <= 5 and all(len(b.split()) <= 12 for b in s.get("bullets", [])) for s in o["slides"]]
    return sum(good) / len(good)


def judge_grade(i, o, e):
    if not o["slides"] or not judge.available():
        return None
    criteria = (f"- correct and specific about '{i['topic']}' for {i['student_level']} students\n"
                f"- covers the learning objectives: {'; '.join(i.get('learning_objectives', []))}\n"
                "- slides follow a teaching order and do not repeat each other")
    return judge.grade(criteria, json.dumps([{k: s.get(k) for k in ("title", "bullets", "notes")} for s in o["slides"]],
                                            ensure_ascii=False))


def pptx_ok(i, o, e):
    p = o.get("pptx") or {}
    return None if not o["slides"] else float(bool(p.get("opens")) and p.get("pptx_slides", 0) >= p.get("draft_slides", 1))


def export_versioning(i, o, e):
    p = o.get("pptx") or {}
    return None if not o["slides"] else float(p.get("second_version") == 2 and bool(p.get("second_parent_is_first")))


SCORERS = {f.__name__: f for f in (complete, quiz_count_exact, quiz_well_formed, quiz_key_agreement, code_verified,
                                   language_match, bullet_rule, judge_grade, pptx_ok, export_versioning)}


SPEC = [
    pct("drafts_complete", "complete", 90.0, "reached the approval step with every part the lecturer asked for"),
    pct("quiz_count_exact", "quiz_count_exact", 90.0, "exactly the number of questions asked for"),
    pct("quiz_well_formed", "quiz_well_formed", 95.0, "4 different options and a valid answer index"),
    pct("quiz_key_agreement", "quiz_key_agreement", 80.0, "the judge answers the quiz without the key; share of questions where it picks the keyed answer"),
    pct("code_verified", "code_verified", 80.0, "code examples the code agent ran successfully in the sandbox (after its fix attempts)"),
    pct("language_match", "language_match", 90.0, "slides written in the brief's language"),
    pct("slide_bullet_rule", "bullet_rule", 80.0, "slides with 3-5 bullets of at most 12 words"),
    Spec("judge_grade", "judge_grade", "", 0.7, True, "Gemini judge, 1-5 grade mapped to 0-1 (accuracy, objectives, order); a factual error caps it at 2"),
    pct("pptx_export_ok", "pptx_ok", 100.0, "Presentations: the exported .pptx opens and has every slide"),
    pct("export_versioning", "export_versioning", 100.0, "Presentations: exporting again makes version 2 linked to version 1 in My library"),
    Spec("median_seconds", "@median_seconds", "s", None, False, "brief -> draft ready for approval (LangSmith run latency)"),
    Spec("crashed", "@crashed", "%", 10.0, False, "briefs whose run ended with an error"),
]

def run(ctx: dict) -> list[Metric]:
    import sys

    return observe.run_case_suite(sys.modules[__name__], ctx)

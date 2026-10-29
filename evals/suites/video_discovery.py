"""Video discovery (M5, section A): a lecturer asks a question, Rafeeq returns YouTube MOMENTS (video + start time).

Each case is a developer-written lecturer question (Arabic and English). The real search runs: query expansion ->
YouTube candidates -> transcripts (or a Gemini outline) -> 45 s segments -> Gemini embeddings in pgvector -> the best
segment per video. The same question is then asked again to measure the cached path (segments already indexed).

Relevance is graded by the LLM judge on the returned segment text: does watching it answer the question?"""
from __future__ import annotations

import os
import time

from evals import judge, observe
from evals.core import Metric

NAME = "video_discovery"
DESCRIPTION = "YouTube moments for lecturer questions: found, top-1 relevance and precision@3 (judge), valid timestamps, speed"
LIVE = True

QUESTIONS = [
    ("en-recursion", "What is recursion in programming?"),
    ("en-binary-search", "How does binary search work?"),
    ("en-tcp-handshake", "How does the TCP three-way handshake work?"),
    ("en-newton-second", "Explain Newton's second law with an example"),
    ("en-linked-list", "What is a linked list and how is it different from an array?"),
    ("ar-photosynthesis", "ما هو التمثيل الضوئي؟"),
    ("ar-ohm", "شرح قانون أوم"),
    ("ar-python-loops", "شرح الحلقات التكرارية في بايثون"),
]


def cases() -> list[dict]:
    return [{"id": cid, "input": {"question": q}, "expected": None} for cid, q in QUESTIONS]


def missing() -> str:
    from rafeeq.teacher.lecturer.settings import DATABASE_URL

    if not DATABASE_URL:
        return "DATABASE_URL (Postgres + pgvector) is not set"
    return "" if os.getenv("GOOGLE_API_KEY") else "GOOGLE_API_KEY (embeddings) is not set"


def target(inp: dict) -> dict:
    from rafeeq.teacher.lecturer import youtube_discovery as yd

    t0 = time.perf_counter()
    first = yd.search(inp["question"], top_k=5)
    t1 = time.perf_counter()
    again = yd.search(inp["question"], top_k=5)
    t2 = time.perf_counter()
    keep = ("video_id", "title", "channel", "start_sec", "end_sec", "score", "url", "text")
    return {"results": [{k: r.get(k) for k in keep} for r in first], "first_seconds": round(t1 - t0, 2),
            "repeat_seconds": round(t2 - t1, 2), "repeat_same_top": bool(first and again and first[0]["url"] == again[0]["url"])}


def _judged(i, o) -> list:
    if "_judged" not in o:
        o["_judged"] = judge.relevant(i["question"], [r["text"] or r["title"] for r in o["results"][:3]]) if judge.available() else []
    return o["_judged"]


def found(i, o, e):
    return float(bool(o["results"]))


def top1_relevant(i, o, e):
    j = _judged(i, o)
    return None if not o["results"] or not j or j[0] is None else float(j[0])


def precision_at_3(i, o, e):
    j = [v for v in _judged(i, o) if v is not None]
    return None if not j else sum(j) / len(j)


def timestamps_valid(i, o, e):
    if not o["results"]:
        return None
    return sum(1 for r in o["results"] if r["start_sec"] is not None and 0 <= r["start_sec"] < (r["end_sec"] or 0)) / len(o["results"])


def first_seconds(i, o, e):
    return o["first_seconds"]


def repeat_seconds(i, o, e):
    return o["repeat_seconds"]


def repeat_stable(i, o, e):
    return None if not o["results"] else float(o["repeat_same_top"])


SCORERS = {f.__name__: f for f in (found, top1_relevant, precision_at_3, timestamps_valid, first_seconds, repeat_seconds, repeat_stable)}


def summarize(rows: list[dict]) -> list[Metric]:
    first = [r["scores"]["first_seconds"] for r in rows if "first_seconds" in r["scores"]]
    repeat = [r["scores"]["repeat_seconds"] for r in rows if "repeat_seconds" in r["scores"]]
    med = lambda v: round(sorted(v)[len(v) // 2], 2) if v else None                  # noqa: E731
    return [
        Metric("questions_answered", observe.pct(rows, "found"), "%", gate=85.0,
               note=f"{len(rows)} lecturer questions (en + ar) that got at least one moment above the similarity floor"),
        Metric("top1_relevant", observe.pct(rows, "top1_relevant"), "%", gate=75.0,
               note="judge: watching the first moment answers the question"),
        Metric("precision_at_3", observe.pct(rows, "precision_at_3"), "%", gate=60.0, note="judge: relevant moments among the top 3"),
        Metric("timestamps_valid", observe.pct(rows, "timestamps_valid"), "%", gate=100.0, note="every moment has 0 ≤ start < end"),
        Metric("repeat_stable", observe.pct(rows, "repeat_stable"), "%", gate=90.0, note="asking again gives the same first moment"),
        Metric("median_first_search", med(first), "s", higher_is_better=False, note="new question: search + transcripts + embeddings"),
        Metric("median_repeat_search", med(repeat), "s", higher_is_better=False, note="same question again: segments already in pgvector"),
        Metric("crashed", observe.error_rate(rows), "%", higher_is_better=False, gate=10.0, note="questions that raised an error"),
    ]


def run(ctx: dict) -> list[Metric]:
    import sys

    return observe.run_case_suite(sys.modules[__name__], ctx)

"""Parent report (teacher): can a parent trust what it says, and can the shared link be abused?

Reports are built for a simulated class with a fixed seed. The numbers in the Arabic summary are checked against the raw
records, independently of the code under test. The summary is written from templates (no LLM), so any number or topic
in it that is not in the data would be an invented claim."""
from __future__ import annotations

import os
import random
import re
import time
from datetime import datetime, timedelta, timezone

from evals.core import Metric
from evals.fakes import FakeStore
from rafeeq.teacher import class_report as cr
from rafeeq.teacher import parent_report as pr

NAME = "parent_report"
DESCRIPTION = "Faithfulness of the weekly report to the recorded data, and safety of the signed share link"

ROOM = "r_eval"
NAMES = {"t1": "الوراثة", "t2": "الخلية", "t3": "الهضم", "t4": "التنفس"}


def _ints(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def run(ctx: dict) -> list[Metric]:
    rnd = random.Random(11)
    store = FakeStore()
    now = datetime.now(timezone.utc)
    saved = (pr._room_and_student, cr.room_topic_names, os.environ.get("RAFEEQ_WEB_SECRET"), pr.SHARE_DAYS)
    pr._room_and_student = lambda room_id, user_id: ("الأحياء", "ليلى")
    cr.room_topic_names = lambda room_id: NAMES
    os.environ["RAFEEQ_WEB_SECRET"] = "eval-secret"
    try:
        n = 40
        for i in range(n):                                       # student 0..4 are silent this week
            u = f"u{i}"
            for _ in range(rnd.randint(0, 5) if i >= 5 else 0):
                t = rnd.choice(list(NAMES))
                right = rnd.randint(0, 8)
                cr.record_quiz_result(ROOM, u, t, [{"question": f"q{k}", "chosen": "a" if k < right else "b", "correct": "a"} for k in range(8)],
                                      time=(now - timedelta(days=rnd.choice([1, 2, 3, 4, 5, 6, 9, 12]))).isoformat(), store=store)
            for _ in range(rnd.randint(0, 6) if i >= 5 else 0):
                cr.record_question(ROOM, u, "سؤال تجريبي", time=(now - timedelta(days=rnd.choice([0, 1, 3, 5, 8]))).isoformat(),
                                   store=store, topic_id=rnd.choice(list(NAMES)))

        cutoff = (now - timedelta(days=pr.PERIOD_DAYS)).isoformat()
        arithmetic_ok = faithful_ok = topics_ok = silent_ok = silent_n = 0
        for i in range(n):
            u = f"u{i}"
            r = pr.build(store, ROOM, u)
            week = [it.value for it in cr._all(store, cr._ns(ROOM, "quiz")) if it.value["user_id"] == u and it.value["time"] >= cutoff]
            asked = [it.value for it in cr._all(store, cr._ns(ROOM, "questions")) if it.value["user_id"] == u and it.value["time"] >= cutoff]
            answered, right = sum(v["total"] for v in week), sum(v["score"] for v in week)
            expect_avg = round(right / answered, 2) if answered else None
            arithmetic_ok += (r["quizzes_taken"] == len(week) and r["questions_answered"] == answered and r["questions_asked"] == len(asked)
                              and r["average_score"] == expect_avg)
            text = " ".join(r["summary"])
            allowed = {len(week), answered, right, len(asked), round(100 * right / answered) if answered else 0}
            faithful_ok += set(_ints(text)) <= allowed
            named = re.findall(r"«([^»]+)»", text)
            topics_ok += all(t in NAMES.values() for t in named)
            if not week and not asked:
                silent_n += 1
                silent_ok += len(r["summary"]) == 1 and not named and "لم يسجّل" in text

        # the signed link
        token = pr.make_share_token(ROOM, "u7")
        checks = {
            "valid link opens the right student": pr.read_share_token(token) == (ROOM, "u7"),
            "a changed character is rejected": pr.read_share_token(token[:-2] + ("AA" if token[-2:] != "AA" else "BB")) is None,
            "another student's id cannot be swapped in": pr.read_share_token(make_other(token)) is None,
            "garbage is rejected": pr.read_share_token("not-a-token") is None and pr.read_share_token("") is None,
        }
        pr.SHARE_DAYS = -1                                        # every link is now "older than allowed"
        checks["an expired link is rejected"] = pr.read_share_token(token) is None
        pr.SHARE_DAYS = saved[3]
        os.environ["RAFEEQ_WEB_SECRET"] = "another-secret"
        checks["a link signed with another secret is rejected"] = pr.read_share_token(token) is None
    finally:
        pr._room_and_student, cr.room_topic_names = saved[0], saved[1]
        pr.SHARE_DAYS = saved[3]
        if saved[2] is None:
            os.environ.pop("RAFEEQ_WEB_SECRET", None)
        else:
            os.environ["RAFEEQ_WEB_SECRET"] = saved[2]
    failed = [k for k, ok in checks.items() if not ok]
    return [
        Metric("report_arithmetic", arithmetic_ok / n * 100, "%", gate=100.0, note=f"counts and average score recomputed from the raw records for {n} students"),
        Metric("summary_numbers_faithful", faithful_ok / n * 100, "%", gate=100.0, note="every number in the Arabic summary is one of the real figures (nothing invented)"),
        Metric("summary_topics_real", topics_ok / n * 100, "%", gate=100.0, note="every «topic» named in the summary is a topic of the room"),
        Metric("silent_week_honest", silent_ok / silent_n * 100 if silent_n else None, "%", gate=100.0, note=f"{silent_n} students with no activity get the single 'no activity' line, not praise"),
        Metric("share_link_checks", (len(checks) - len(failed)) / len(checks) * 100, "%", gate=100.0,
               note="signed link: " + ("all pass" if not failed else "FAILED: " + "; ".join(failed))),
    ]


def make_other(token: str) -> str:
    """A token for a different student, signed with the right secret and salt but for another user, cannot be built by
    editing the old one: take the payload part of another valid token and keep the old signature."""
    other = pr.make_share_token(ROOM, "u8")
    return other.split(".")[0] + "." + ".".join(token.split(".")[1:])

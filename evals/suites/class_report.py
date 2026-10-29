"""Class report (teacher): does the mastery picture match what the students really know?

A class is SIMULATED with a fixed seed: every student has a true ability per topic, and takes quizzes whose scores are
drawn from it (so there is noise, as in real life). The report is then compared with the truth we planted."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from evals.core import Metric
from evals.fakes import FakeStore
from rafeeq.teacher import class_report as cr

NAME = "class_report"
DESCRIPTION = "At-risk students, topic ranking, how fast improvement shows, and the no-LLM topic grouping of questions"

ROOM = "r_eval"
TOPICS = ["الوراثة", "الخلية", "التمثيل الضوئي", "الهضم", "التنفس", "الدورة الدموية"]
LEVELS = [0.25, 0.4, 0.55, 0.7, 0.85]

# developer-written questions with the topic each belongs to (the word-overlap fallback is what runs without an LLM)
QUESTIONS = [
    ("ما الفرق بين الجين والكروموسوم في الوراثة؟", "الوراثة"), ("كيف تنتقل الصفات بالوراثة من الآباء؟", "الوراثة"),
    ("ما هو الأليل السائد في الوراثة؟", "الوراثة"), ("ما وظيفة الجدار في الخلية النباتية؟", "الخلية"),
    ("أين تقع النواة داخل الخلية؟", "الخلية"), ("ما الفرق بين الخلية الحيوانية والنباتية؟", "الخلية"),
    ("كيف يحدث التمثيل الضوئي في الورقة؟", "التمثيل الضوئي"), ("ما نواتج التمثيل الضوئي؟", "التمثيل الضوئي"),
    ("ما دور الكلوروفيل في التمثيل الضوئي؟", "التمثيل الضوئي"), ("أين يبدأ الهضم في الجسم؟", "الهضم"),
    ("ما دور الإنزيمات في الهضم؟", "الهضم"), ("كيف يمتص الجسم الغذاء بعد الهضم؟", "الهضم"),
    ("ما هي مراحل التنفس الخلوي؟", "التنفس"), ("أين يحدث التنفس الخلوي؟", "التنفس"),
    ("ما الفرق بين التنفس الهوائي واللاهوائي؟", "التنفس"), ("كيف تعمل الدورة الدموية في الجسم؟", "الدورة الدموية"),
    ("ما وظيفة القلب في الدورة الدموية؟", "الدورة الدموية"), ("ما الفرق بين الشرايين والأوردة في الدورة الدموية؟", "الدورة الدموية"),
    # phrased without the topic's own words: the word-overlap fallback is expected to miss these
    ("لماذا نشعر بالتعب عند الجري بسرعة؟", "التنفس"), ("كيف تصنع النبتة طعامها من الشمس؟", "التمثيل الضوئي"),
    ("ما سبب تشابه الأبناء مع آبائهم؟", "الوراثة"), ("كيف يصل الدم إلى كل أجزاء الجسم؟", "الدورة الدموية"),
]


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


def run(ctx: dict) -> list[Metric]:
    rnd = random.Random(7)
    store = FakeStore()
    orig_students, orig_topics, orig_llm = cr._students, cr.room_topic_names, cr._llm_topics
    students = {f"u{i}": f"طالب {i}" for i in range(40)}
    cr._students = lambda room_id: students
    cr.room_topic_names = lambda room_id: {}
    cr._llm_topics = lambda texts, candidates: None                 # the fallback is what is measured
    try:
        truth = {(u, t): rnd.choice(LEVELS) for u in students for t in TOPICS}
        now = datetime.now(timezone.utc)
        for (u, t), p in truth.items():
            for k in range(4):                                       # four quizzes of ten questions, oldest first
                right = sum(rnd.random() < p for _ in range(10))
                answers = [{"question": f"q{i}", "chosen": "a" if i < right else "b", "correct": "a"} for i in range(10)]
                cr.record_quiz_result(ROOM, u, t, answers, time=(now - timedelta(days=8 - 2 * k)).isoformat(), store=store)
        report = cr.build_report(store, ROOM)
        flagged = {(s["user_id"], t) for s in report["students_at_risk"] for t in s["topic_ids"]}      # (student, topic) pairs
        should = {(u, t) for (u, t), p in truth.items() if p < cr.WATCH}
        tp = len(flagged & should)
        recall = tp / len(should) * 100 if should else 100.0
        precision = tp / len(flagged) * 100 if flagged else 100.0

        true_topic = [sum(truth[(u, t)] for u in students) / len(students) for t in TOPICS]
        got = {t["topic_id"]: t["mastery"] for t in report["topics"]}
        rho = _spearman([got[t] for t in TOPICS], true_topic)

        # how many quizzes after a student really improves until the report calls them on track?
        s2 = FakeStore()
        for k in range(3):
            cr.record_quiz_result(ROOM, "u0", "الخلية", [{"question": "q", "chosen": "b", "correct": "a"}] * 7 +
                                  [{"question": "q", "chosen": "a", "correct": "a"}] * 3, time=(now - timedelta(days=10 - k)).isoformat(), store=s2)
        lag = 0
        while cr.student_levels(s2, ROOM)["u0"][1] != "on_track" and lag < 10:
            lag += 1
            cr.record_quiz_result(ROOM, "u0", "الخلية", [{"question": "q", "chosen": "a", "correct": "a"}] * 8 +
                                  [{"question": "q", "chosen": "b", "correct": "a"}] * 2, time=(now - timedelta(hours=10 - lag)).isoformat(), store=s2)

        hits = sum(cr._keyword_topic(q, TOPICS) == t for q, t in QUESTIONS)
        general = sum(cr._keyword_topic(q, TOPICS) == cr.GENERAL for q, t in QUESTIONS)
        wrong = len(QUESTIONS) - hits - general
    finally:
        cr._students, cr.room_topic_names, cr._llm_topics = orig_students, orig_topics, orig_llm
    return [
        Metric("at_risk_recall", recall, "%", gate=85.0, note=f"of {len(should)} real (student, topic) struggles, the share the report flags (simulated class of 40 students x 6 topics, 4 noisy quizzes each)"),
        Metric("at_risk_precision", precision, "%", gate=85.0, note=f"of {len(flagged)} flagged (student, topic) pairs, the share that really struggle"),
        Metric("topic_ranking_correlation", rho, "", gate=0.9, note="Spearman: does the weakest-first topic order match the real difficulty? (1 = identical)"),
        Metric("quizzes_until_improvement_shows", float(lag), "", higher_is_better=False, gate=3.0,
               note="a student who went from 30% to 80%: quizzes until the report says on_track (the average reacts slowly on purpose)"),
        Metric("topic_fallback_accuracy", hits / len(QUESTIONS) * 100, "%", note=f"no-LLM word-overlap grouping on {len(QUESTIONS)} developer-written questions; {general} fell to 'general'"),
        Metric("topic_fallback_wrong_topic", wrong / len(QUESTIONS) * 100, "%", higher_is_better=False, gate=10.0,
               note="questions put under a WRONG topic (worse than 'general')"),
    ]

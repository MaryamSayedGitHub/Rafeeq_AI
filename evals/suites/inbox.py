"""Teacher inbox: when a student asks something a teacher already answered, does the tutor reuse that answer?

Reusing the WRONG teacher answer for a different question is the harmful error, so it is measured on its own. The
questions are developer-written (Arabic and English): each answered question has paraphrases that SHOULD match and
near-misses on the same topic that SHOULD NOT."""
from __future__ import annotations

from evals.core import Metric
from evals.fakes import FakeStore
from rafeeq.teacher import class_report as cr
from rafeeq.teacher import inbox

NAME = "inbox"
DESCRIPTION = "Reuse of a teacher's verified answer: right reuse on paraphrases, wrong reuse on near-miss questions"

ROOM = "r_eval"
# (the question the teacher answered, [paraphrases that should reuse it], [different questions that must NOT reuse it])
BANK = [
    ("ما الفرق بين الجين والكروموسوم؟",
     ["ما هو الفرق بين الجين والكروموسوم", "اشرح الفرق بين الكروموسوم والجين", "الجين والكروموسوم ما الفرق بينهما؟"],
     ["ما وظيفة الجين؟", "كم عدد الكروموسومات في الإنسان؟"]),
    ("ما وظيفة الجدار الخلوي في الخلية النباتية؟",
     ["ما هي وظيفة الجدار الخلوي للخلية النباتية", "وظيفة الجدار الخلوي في الخلية النباتية؟", "ماذا يفعل الجدار الخلوي في الخلية النباتية؟"],
     ["ما وظيفة الغشاء الخلوي في الخلية الحيوانية؟", "ممَّ يتكوّن الجدار الخلوي؟"]),
    ("متى يكون امتحان الفصل الأول؟",
     ["ما موعد امتحان الفصل الأول؟", "امتى امتحان الفصل الأول", "هل تعرف متى امتحان الفصل الأول؟"],
     ["متى يكون امتحان الفصل الثاني؟", "ما مواد امتحان الفصل الأول؟"]),
    ("What is the difference between mitosis and meiosis?",
     ["what is the difference between meiosis and mitosis", "Difference between mitosis and meiosis?", "can you explain the difference between mitosis and meiosis"],
     ["What happens in the first phase of mitosis?", "Where does meiosis take place?"]),
    ("How is the final grade calculated?",
     ["how is my final grade calculated", "How do you calculate the final grade?", "how is the final grade calculated in this course"],
     ["How is the participation grade calculated?", "When will the final grade be published?"]),
    ("ما نواتج عملية التمثيل الضوئي؟",
     ["ما هي نواتج التمثيل الضوئي", "نواتج عملية التمثيل الضوئي ايه؟", "اذكر نواتج عملية التمثيل الضوئي"],
     ["ما شروط عملية التمثيل الضوئي؟", "أين تحدث عملية التمثيل الضوئي؟"]),
    ("هل يمكن تسليم الواجب متأخرًا؟",
     ["هل ممكن تسليم الواجب متأخرا", "أقدر أسلم الواجب متأخر؟", "هل يقبل تسليم الواجب متأخرًا؟"],
     ["هل يمكن إعادة الواجب بعد التصحيح؟", "ما موعد تسليم الواجب؟"]),
    ("Which chapters are included in the midterm?",
     ["which chapters are in the midterm", "What chapters does the midterm include?", "Which chapters are covered in the midterm?"],
     ["When is the midterm?", "Which chapters are in the final exam?"]),
]


def _score(threshold: float, bank: list, answered_store: FakeStore) -> tuple[float, float, float]:
    old = inbox.MATCH_THRESHOLD
    inbox.MATCH_THRESHOLD = threshold
    try:
        pos = [(i, p) for i, (_, ps, _) in enumerate(bank) for p in ps]
        neg = [(i, n) for i, (_, _, ns) in enumerate(bank) for n in ns]

        def hit(q):
            r = inbox.find_verified(ROOM, q, store=answered_store)
            return r["question"] if r else None
        recall = sum(hit(p) == bank[i][0] for i, p in pos) / len(pos)
        wrong = sum(hit(n) is not None for _, n in neg) / len(neg)
        exact = sum(hit(q) == q for q, _, _ in bank) / len(bank)
        return recall, wrong, exact
    finally:
        inbox.MATCH_THRESHOLD = old


def run(ctx: dict) -> list[Metric]:
    store = FakeStore()
    for k, (q, _, _) in enumerate(BANK):
        qid = f"q_{k}"
        store.put(inbox._ns(ROOM), qid, {"question_id": qid, "user_id": "u1", "text": q, "status": "answered", "answer": "جواب المدرس", "answered_at": "x"})
    recall, wrong, exact = _score(inbox.MATCH_THRESHOLD, BANK, store)
    sweep = {f"{t:.2f}": dict(zip(("recall", "wrong_reuse"), [round(x * 100, 1) for x in _score(t, BANK, store)[:2]])) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}
    n_pos, n_neg = sum(len(p) for _, p, _ in BANK), sum(len(n) for _, _, n in BANK)
    best = max((t for t, v in sweep.items() if v["wrong_reuse"] <= 5.0), key=lambda t: sweep[t]["recall"], default=None)
    return [
        Metric("repeat_question_reuse", exact * 100, "%", gate=100.0, note="the exact same question must always get the teacher's answer"),
        Metric("paraphrase_reuse_recall", recall * 100, "%", gate=70.0,
               note=f"{n_pos} rewordings of an answered question that get the teacher's answer (threshold {inbox.MATCH_THRESHOLD}, word overlap)",
               detail={"threshold_sweep": sweep}),
        Metric("wrong_reuse_rate", wrong * 100, "%", higher_is_better=False, gate=5.0,
               note=f"{n_neg} DIFFERENT questions on the same topic that wrongly got a teacher's answer", detail={"threshold_sweep": sweep}),
        Metric("best_threshold_under_5pct_wrong_reuse", float(best) if best else None, "", note="from the sweep: the threshold with the best recall while wrong reuse stays at or under 5%"),
    ]

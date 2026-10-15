"""Fill one room with realistic student activity so the lecturer's reports have data in a demo.

M4's quiz backend (POST /quiz, /quiz/{id}/submit) is not in main yet, so no real quiz results
exist. This script records demo questions and quiz results through the same functions the real
code uses (class_report.record_question / record_quiz_result), so the /app Overview, Students
and Class insights pages show real computed reports instead of sample files.

    uv run python -m rafeeq.teacher.seed_class_demo <room_id>
    uv run python -m rafeeq.teacher.seed_class_demo <room_id> --add-students 4
    uv run python -m rafeeq.teacher.seed_class_demo <room_id> --clear

Needs DATABASE_URL (Postgres): an in-memory store would be gone when this script exits.
Topics: the lecturer's sessions for this room if there are any, otherwise three biology topics.
--add-students N adds demo student accounts until the room has N students (password: demo1234).
"""

import argparse
import random
import sys
from datetime import datetime, timedelta, timezone

from rafeeq.teacher import class_report

DEFAULT_TOPICS = {
    "البناء الضوئي": {
        "questions": ["ما معنى البناء الضوئي؟", "ليه النبات محتاج ضوء الشمس؟", "إيه دور الكلوروفيل؟",
                      "البناء الضوئي بيحصل فين في الورقة؟"],
        "quiz": [("ما الغاز الذي يمتصه النبات في البناء الضوئي؟", "ثاني أكسيد الكربون", "الأكسجين"),
                 ("أين يحدث البناء الضوئي؟", "البلاستيدات الخضراء", "الميتوكوندريا"),
                 ("ما الناتج الرئيسي للبناء الضوئي؟", "الجلوكوز", "البروتين")],
    },
    "امتصاص النبات للغذاء": {
        "questions": ["إزاي الجذر بيمتص الماء؟", "يعني إيه الخاصية الأسموزية؟", "الأملاح بتطلع للورقة إزاي؟"],
        "quiz": [("ما الجزء المسؤول عن امتصاص الماء؟", "الشعيرات الجذرية", "الأوراق"),
                 ("انتقال الماء من التركيز الأعلى للأقل يسمى؟", "الخاصية الأسموزية", "النقل النشط")],
    },
    "تركيب الخلية": {
        "questions": ["الفرق بين الخلية النباتية والحيوانية؟", "وظيفة الجدار الخلوي إيه؟"],
        "quiz": [("ما الذي يميز الخلية النباتية؟", "الجدار الخلوي", "الغشاء البلازمي"),
                 ("أين توجد المادة الوراثية؟", "النواة", "الفجوة العصارية")],
    },
}


def _generic_topic(topic: str) -> dict:
    return {
        "questions": [f"مش فاهم {topic}، ممكن مثال بسيط؟", f"إيه أهم حاجة في {topic}؟",
                      f"إزاي أفرّق بين المفاهيم في {topic}؟"],
        "quiz": [(f"أي عبارة صحيحة عن {topic}؟", "العبارة الأولى", "العبارة الثانية"),
                 (f"ما التطبيق العملي الأقرب لـ {topic}؟", "التطبيق أ", "التطبيق ب")],
    }


def _members(room_id: str, add: int) -> dict[str, str]:
    from rafeeq.platform.service import get_store

    platform = get_store()
    room = platform.db["rooms"].get(room_id)
    if room is None:
        sys.exit(f"Room '{room_id}' not found. Open the room in /app and copy its id from the address bar.")
    members = {uid: platform.db["users"][uid]["name"]
               for uid in platform.db["members"].get(room_id, []) if uid in platform.db["users"]}
    if len(members) < add:
        names = ["عمر خالد", "سارة محمود", "يوسف أحمد", "منى علي", "كريم حسن", "ليلى سمير"]
        taken = {u["email"] for u in platform.db["users"].values()}
        created, i = 0, 0
        while len(members) < add:
            i += 1
            email = f"demo.student{i}.{room_id}@rafeeq.local"
            if email in taken:
                continue
            name = names[(i - 1) % len(names)]
            login = platform.register(name, email, "demo1234", "student", room["join_code"])
            members[login["user_id"]] = name
            created += 1
        print(f"Created {created} demo students (password demo1234).")
    if not members:
        sys.exit("The room has no students. Join it with a student account, or use --add-students 4.")
    return members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("room_id")
    parser.add_argument("--add-students", type=int, default=0, metavar="N")
    parser.add_argument("--clear", action="store_true", help="delete this room's recorded activity first")
    args = parser.parse_args()

    from rafeeq.teacher.lecturer.memory import open_memory
    from rafeeq.teacher.lecturer.settings import DATABASE_URL

    if not DATABASE_URL:
        sys.exit("DATABASE_URL is not set: start Postgres (docker compose up -d db) and set it in .env.")

    members = _members(args.room_id, args.add_students)
    rng = random.Random(42)
    with open_memory() as (_, store):
        if args.clear:
            for kind in ("questions", "quiz"):
                for item in class_report._all(store, ("class", args.room_id, kind)):
                    store.delete(("class", args.room_id, kind), item.key)
            print("Cleared earlier activity.")

        topics = {t: _generic_topic(t) for t in class_report._room_session_topics(store, args.room_id)} \
            or DEFAULT_TOPICS
        # one student struggles, one is strong, the rest in between
        skill = {uid: s for uid, s in zip(members, [0.3, 0.9] + [rng.uniform(0.5, 0.8) for _ in members])}
        now = datetime.now(timezone.utc)
        n_q = n_r = 0
        for uid in members:
            for topic, bank in topics.items():
                hardness = rng.uniform(-0.15, 0.15)
                for text in rng.sample(bank["questions"], k=min(len(bank["questions"]), rng.randint(1, 3))):
                    when = (now - timedelta(days=rng.randint(0, 20), hours=rng.randint(0, 23))).isoformat()
                    n_q += class_report.record_question(args.room_id, uid, text, time=when, store=store, topic_id=topic)
                for attempt in range(rng.randint(1, 3)):
                    answers = []
                    for question, correct, wrong in bank["quiz"]:
                        ok = rng.random() < max(0.05, min(0.95, skill[uid] - hardness + 0.1 * attempt))
                        answers.append({"question": question, "chosen": correct if ok else wrong, "correct": correct})
                    when = (now - timedelta(days=15 - 5 * attempt, hours=rng.randint(0, 23))).isoformat()
                    n_r += class_report.record_quiz_result(args.room_id, uid, topic, answers, time=when, store=store)
        print(f"Recorded {n_q} questions and {n_r} quiz results for {len(members)} students "
              f"in {len(topics)} topics: {', '.join(topics)}")
        print("Open the room in /app as the teacher: Overview, Students and Class insights now use this data.")


if __name__ == "__main__":
    main()

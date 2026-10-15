"""Real class report and insights for the lecturer (M5), built from what students actually do.

Replaces the sample data behind GET /teacher/rooms/{room_id}/report and /insights, which
M1's /app Overview, Students and Class insights pages already show to the lecturer.

Inputs, recorded as they happen:
    record_question(room_id, user_id, text)
        called by the platform router for every student copilot message (M1/M3 side).
    record_quiz_result(room_id, user_id, topic_id, answers)
        called by M4's POST /quiz/{quiz_id}/submit, once per topic of the quiz. answers = [{"question", "chosen", "correct"}].

Outputs, in the shapes of contracts/openapi.yaml:
    build_insights(store, room_id) -> Insights     what students ask about, mistakes that repeat
    build_report(store, room_id)   -> ClassReport  mastery per topic, students at risk
    student_levels(store, room_id) -> {user_id: (mastery, status)} for /rooms/{room_id}/students

Storage: the teacher service's LangGraph store (Postgres when DATABASE_URL works, memory
otherwise), the same store as the teacher profile and sessions:
    ("class", room_id, "questions")  key <id>  {user_id, text, time, topic_id}
    ("class", room_id, "quiz")       key <id>  {user_id, topic_id, score, total, wrong, time}

Topics: the room's own topics from M2 (learned from the uploaded material) when the knowledge
base is on, so quiz results (M4 uses the same topic ids) and questions line up; otherwise a topic
id is the topic's name. Questions are put under a topic the first time a report is built: the topics of the
lecturer's own sessions for that room are the candidates, the LLM picks one (or names a new
one), and the answer is stored so each question is classified only once. Without an LLM a
simple word-overlap match is used, and anything unmatched goes under GENERAL.

Mastery is a stand-in until M4's learner profile exists: recent quiz scores weigh more
(exponential average). When M4 lands, read mastery from M4 instead and keep the rest.
"""

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

GENERAL = "عام"
WINDOW_DAYS = 30
ON_TRACK, WATCH = 0.7, 0.5          # mastery >= 0.7 on_track, >= 0.5 watch, else struggling
EMA_ALPHA = 0.5                       # weight of the newest quiz in a student's mastery
MAX_TOP_QUESTIONS = 8
MAX_MISTAKES = 6
CLASSIFY_BATCH = 30

_store = None


def set_store(store) -> None:
    """Called once by the teacher service at startup (rafeeq/teacher/service.py)."""
    global _store
    _store = store


def get_store():
    return _store


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ns(room_id: str, kind: str) -> tuple:
    return ("class", room_id, kind)


# ---------------------------------------------------------------- recording

def record_question(room_id: str, user_id: str, text: str, time: str | None = None, store=None,
                    topic_id: str | None = None) -> bool:
    """Never raises: the student's answer must not fail because a report could not be updated.
    topic_id is optional: left empty, the question is put under a topic when a report is built."""
    store = store or _store
    text = (text or "").strip()
    if store is None or not room_id or not user_id or not text:
        return False
    try:
        store.put(_ns(room_id, "questions"), uuid.uuid4().hex, {
            "user_id": user_id, "text": text[:500], "time": time or _now().isoformat(), "topic_id": topic_id,
        })
        return True
    except Exception as e:
        log.warning("class_report: could not record a question: %s: %s", type(e).__name__, str(e)[:200])
        return False


def record_quiz_result(room_id: str, user_id: str, topic_id: str, answers: list[dict],
                       time: str | None = None, store=None) -> bool:
    """answers: one dict per question, {"question", "chosen", "correct"}. Never raises."""
    store = store or _store
    if store is None or not room_id or not user_id or not topic_id or not answers:
        return False
    wrong = [a for a in answers if str(a.get("chosen", "")).strip() != str(a.get("correct", "")).strip()]
    try:
        store.put(_ns(room_id, "quiz"), uuid.uuid4().hex, {
            "user_id": user_id, "topic_id": topic_id, "score": len(answers) - len(wrong),
            "total": len(answers), "wrong": wrong[:20], "time": time or _now().isoformat(),
        })
        return True
    except Exception as e:
        log.warning("class_report: could not record a quiz result: %s: %s", type(e).__name__, str(e)[:200])
        return False


# ---------------------------------------------------------------- reading

def _all(store, namespace: tuple, page: int = 500) -> list:
    items, offset = [], 0
    while True:
        batch = store.search(namespace, limit=page, offset=offset)
        items.extend(batch)
        if len(batch) < page:
            return items
        offset += page


def _recent(items: list) -> list:
    cutoff = (_now() - timedelta(days=WINDOW_DAYS)).isoformat()
    return [i for i in items if (i.value.get("time") or "") >= cutoff]


def room_topic_names(room_id: str) -> dict[str, str]:
    """topic_id -> name of the room's own topics (M2's knowledge base, learned from the uploaded
    material). Empty when M2 is off or nothing is indexed: the lecturer's session topics are used then."""
    try:
        from rafeeq.knowledge import integration

        if not integration.enabled():
            return {}
        from rafeeq.knowledge.service import get_service

        return {t["topic_id"]: t["name"] for t in get_service().list_topics(room_id)["topics"]}
    except Exception as e:
        log.warning("class_report: could not read the room's topics: %s", e)
        return {}


def _room_session_topics(store, room_id: str) -> list[str]:
    """Topics of the lecturer sessions linked to this room: the natural topic list for its questions."""
    topics: list[str] = []
    try:
        for ns in store.list_namespaces(prefix=("lecturer",), max_depth=2):
            for item in store.search((*ns, "sessions"), limit=200):
                topic = (item.value.get("topic") or "").strip()
                if item.value.get("room_id") == room_id and topic and topic not in topics:
                    topics.append(topic)
    except Exception as e:
        log.warning("class_report: could not read session topics: %s", e)
    return topics


def _students(room_id: str) -> dict[str, str]:
    """user_id -> name for the room's members (M1's platform store). Empty if unavailable."""
    try:
        from rafeeq.platform.service import get_store

        db = get_store().db
        return {uid: db["users"][uid]["name"] for uid in db["members"].get(room_id, []) if uid in db["users"]}
    except Exception as e:
        log.warning("class_report: could not read room members: %s", e)
        return {}


# ---------------------------------------------------------------- topics for questions

_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def _keyword_topic(text: str, candidates: list[str]) -> str:
    words = _words(text)
    best, best_score = GENERAL, 0
    for c in candidates:
        score = len(words & _words(c))
        if score > best_score:
            best, best_score = c, score
    return best


def _llm_topics(texts: list[str], candidates: list[str]) -> list[str] | None:
    """One topic per question, or None if no LLM is configured or the call fails."""
    try:
        from pydantic import BaseModel, Field

        from rafeeq.teacher.lecturer.llm import get_llm

        class Topics(BaseModel):
            topics: list[str] = Field(description="Exactly one topic per question, in the same order.")

        known = "\n".join(f"- {c}" for c in candidates) or "(none yet)"
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        prompt = (
            "You group students' questions by lesson topic for their teacher.\n"
            f"Known topics of this course:\n{known}\n\n"
            "For each question return the known topic it belongs to, copied exactly. If none fits, "
            "return a short new topic name (2 to 4 words) in the question's language. "
            f"Return exactly {len(texts)} topics.\n\nQuestions:\n{numbered}"
        )
        result = get_llm(temperature=0).with_structured_output(Topics).invoke(prompt)
        topics = [t.strip()[:60] or GENERAL for t in result.topics]
        return topics if len(topics) == len(texts) else None
    except Exception as e:
        log.info("class_report: LLM topic grouping unavailable (%s: %s), using word overlap",
                 type(e).__name__, str(e)[:150])
        return None


def _classify_pending(store, room_id: str, questions: list) -> None:
    pending = [q for q in questions if not q.value.get("topic_id")]
    if not pending:
        return
    room_topics = room_topic_names(room_id)                 # M2: {topic_id: name}
    by_name = {name: tid for tid, name in room_topics.items()}
    candidates = list(by_name) or _room_session_topics(store, room_id)
    for start in range(0, len(pending), CLASSIFY_BATCH):
        batch = pending[start:start + CLASSIFY_BATCH]
        texts = [q.value["text"] for q in batch]
        topics = _llm_topics(texts, candidates) or [_keyword_topic(t, candidates) for t in texts]
        for item, topic in zip(batch, topics):
            item.value["topic_id"] = by_name.get(topic, topic)  # the room's topic id when it is one of them
            try:
                store.put(_ns(room_id, "questions"), item.key, item.value)
            except Exception as e:
                log.warning("class_report: could not save a question topic: %s", e)
        for t in topics:   # topics named by the LLM become candidates for the next batch
            if t != GENERAL and t not in candidates:
                candidates.append(t)


# ---------------------------------------------------------------- mastery

def _status(mastery: float) -> str:
    return "on_track" if mastery >= ON_TRACK else "watch" if mastery >= WATCH else "struggling"


def _mastery_by_student_topic(results: list) -> dict[tuple[str, str], float]:
    """(user_id, topic_id) -> mastery in [0, 1], newest quizzes weighing most."""
    mastery: dict[tuple[str, str], float] = {}
    for item in sorted(results, key=lambda i: i.value.get("time") or ""):
        v = item.value
        key = (v["user_id"], v["topic_id"])
        frac = v["score"] / max(v["total"], 1)
        mastery[key] = frac if key not in mastery else (1 - EMA_ALPHA) * mastery[key] + EMA_ALPHA * frac
    return mastery


# ---------------------------------------------------------------- outputs

def build_insights(store, room_id: str) -> dict:
    """Insights: what students ask about (by topic) and which quiz mistakes repeat."""
    if store is None:
        return {"top_questions": [], "common_mistakes": []}
    questions = _recent(_all(store, _ns(room_id, "questions")))
    _classify_pending(store, room_id, questions)

    by_topic: dict[str, list] = {}
    for q in questions:
        by_topic.setdefault(q.value.get("topic_id") or GENERAL, []).append(q.value)
    top = []
    for topic, items in by_topic.items():
        newest = max(items, key=lambda v: v.get("time") or "")
        top.append({"topic_id": topic, "count": len(items), "example": newest["text"]})
    top.sort(key=lambda t: -t["count"])

    mistakes: dict[tuple, dict] = {}
    for r in _recent(_all(store, _ns(room_id, "quiz"))):
        for w in r.value.get("wrong", []):
            key = (r.value["topic_id"], w.get("question", ""), w.get("chosen", ""))
            entry = mistakes.setdefault(key, {"count": 0, "correct": w.get("correct", "")})
            entry["count"] += 1
    common = []
    for (topic, question, chosen), e in sorted(mistakes.items(), key=lambda kv: -kv[1]["count"])[:MAX_MISTAKES]:
        who = f"{e['count']} طلاب اختاروا" if e["count"] > 1 else "طالب واحد اختار"
        common.append({"topic_id": topic,
                       "description": f"«{question}»: {who} «{chosen}» بدل «{e['correct']}»"})
    return {"top_questions": top[:MAX_TOP_QUESTIONS], "common_mistakes": common}


def build_report(store, room_id: str) -> dict:
    """ClassReport: class mastery per topic (weakest first) and the students struggling in any topic."""
    if store is None:
        return {"topics": [], "students_at_risk": []}
    mastery = _mastery_by_student_topic(_recent(_all(store, _ns(room_id, "quiz"))))

    per_topic: dict[str, list[float]] = {}
    for (_, topic), m in mastery.items():
        per_topic.setdefault(topic, []).append(m)
    topics = [{"topic_id": t, "mastery": round(sum(ms) / len(ms), 2), "status": _status(sum(ms) / len(ms))}
              for t, ms in per_topic.items()]
    topics.sort(key=lambda t: t["mastery"])

    weak: dict[str, list[str]] = {}
    for (uid, topic), m in mastery.items():
        if _status(m) == "struggling":
            weak.setdefault(uid, []).append(topic)
    names = _students(room_id) if weak else {}
    at_risk = [{"user_id": uid, "name": names.get(uid, uid), "topic_ids": sorted(ts)} for uid, ts in weak.items()]
    at_risk.sort(key=lambda s: -len(s["topic_ids"]))
    return {"topics": topics, "students_at_risk": at_risk}


def student_levels(store, room_id: str) -> dict[str, tuple[float, str]]:
    """user_id -> (mastery, status) averaged over the topics the student was quizzed on."""
    if store is None:
        return {}
    per_student: dict[str, list[float]] = {}
    for (uid, _), m in _mastery_by_student_topic(_recent(_all(store, _ns(room_id, "quiz")))).items():
        per_student.setdefault(uid, []).append(m)
    return {uid: (round(sum(ms) / len(ms), 2), _status(sum(ms) / len(ms))) for uid, ms in per_student.items()}

"""tag_topic: which topic of the room is this question about? Returns a topic_id (M2's list) or "other".

Order (cheapest first): the topic M2 already put on the passages it found  ->  words shared with the topic names  ->  the AI model picks
one from the list  ->  "other". M4 and M5 use these ids, so only ids from M2's topic list (or "other") are ever returned.
"""
from __future__ import annotations

import re
from collections import Counter

from . import llm, m2

OTHER = "other"


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z0-9؀-ۿ]{3,}", text.lower())}


def tag_topic(room_id: str, question: str, chunks: list[dict] | None = None, topics: list[dict] | None = None) -> str:
    topics = topics if topics is not None else m2.topics(room_id)[0]
    ids = [t["topic_id"] for t in topics]
    seen = Counter(c.get("topic_id") for c in (chunks or []) if c.get("topic_id") in ids)
    if seen:                                                       # the passages that answered the question know their topic
        return seen.most_common(1)[0][0]
    q = _words(question)
    scored = sorted(((len(q & _words(t["name"])), t["topic_id"]) for t in topics), reverse=True)
    if scored and scored[0][0] >= 1 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    if not ids:
        return OTHER
    names = "; ".join(f"{t['topic_id']} = {t['name']}" for t in topics)
    res = llm.ask_json(f"Pick the one topic that best fits this student question.\nQuestion: {question}\nTopics: {names}; {OTHER}\n"
                       f'Return JSON only: {{"topic": "<one of the ids>"}}', max_tokens=60)
    if isinstance(res, dict) and res.get("topic") in ids + [OTHER]:
        return res["topic"]
    return OTHER

"""Animated video (M5, section A): the story planner that writes the scenes before any clip is generated.

Live cases: developer-written topics (Arabic and English). The real planner runs (LLM_PROVIDER, a facts blueprint
then the scenes). Measured: the exact scene count, narration length inside the 18-32 word rule, scenes that are not
copies of each other, the narration language, and a judge grade of the teaching arc (hook -> ordered steps -> recap,
facts correct). Clip generation (paid video model) is not evaluated here.

Offline checks (always run): the planner's guards with a scripted fake model."""
from __future__ import annotations

import json
import os

from evals import judge, observe
from evals.core import Metric, Spec, pct

NAME = "story_video"
DESCRIPTION = "Story planner: scene count, narration rule, distinct scenes, language, judge grade of the teaching arc; planner guards"
LIVE = True

TOPICS = [
    ("en-water-cycle", "The water cycle", "en", 5),
    ("en-how-vaccines-work", "How vaccines train the immune system", "en", 6),
    ("en-binary-numbers", "How computers count in binary", "en", 4),
    ("ar-photosynthesis", "كيف يصنع النبات غذاءه: التمثيل الضوئي", "ar", 5),
    ("ar-earth-layers", "طبقات الأرض", "ar", 4),
]


def cases() -> list[dict]:
    return [{"id": cid, "input": {"topic": t, "language": lang, "scenes": n, "style": "storybook"}, "expected": {"scenes": n}}
            for cid, t, lang, n in TOPICS]


def missing() -> str:
    provider = (os.getenv("STORY_LLM_PROVIDER") or os.getenv("LLM_PROVIDER") or "").lower()
    return "GROQ_API_KEY is not set" if provider in ("", "groq") and not os.getenv("GROQ_API_KEY") else ""


def target(inp: dict) -> dict:
    from rafeeq.teacher import story_video as sv

    return sv.plan_story(inp["topic"], scenes=inp["scenes"], language=inp["language"], style=inp["style"])


def scenes_exact(i, o, e):
    return float(len(o["scenes"]) == e["scenes"])


def narration_rule(i, o, e):
    from rafeeq.teacher.story_video import NARRATION_WORDS

    lo, hi = NARRATION_WORDS
    return sum(lo <= len(s["narration"].split()) <= hi for s in o["scenes"]) / len(o["scenes"])


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.lower().split()), set(b.lower().split())
    return len(x & y) / len(x | y) if x | y else 1.0


def distinct_scenes(i, o, e):
    """A scene whose picture description is ≥ 70 % the same words as another scene's is a copy."""
    v = [s["visual"] for s in o["scenes"]]
    copies = {a for a in range(len(v)) for b in range(len(v)) if a != b and _jaccard(v[a], v[b]) >= 0.7}
    return 1 - len(copies) / len(v)


def language_match(i, o, e):
    return sum(observe.in_language(s["narration"], i["language"]) for s in o["scenes"]) / len(o["scenes"])


def judge_arc(i, o, e):
    if not judge.available():
        return None
    criteria = (f"- it teaches '{i['topic']}' to school students: scene 1 is a hook, the middle scenes add one correct idea or step each, "
                "in a sensible order, the last scene is a short recap\n- each narration contains a real teaching point (no filler)\n"
                "- the visual descriptions show the real thing being explained")
    return judge.grade(criteria, json.dumps(o["scenes"], ensure_ascii=False))


SCORERS = {f.__name__: f for f in (scenes_exact, narration_rule, distinct_scenes, language_match, judge_arc)}


SPEC = [
    pct("scene_count_exact", "scenes_exact", 100.0, "exactly the scenes asked for"),
    pct("narration_in_rule", "narration_rule", 80.0, "narrations of 18-32 words (fits one clip)"),
    pct("distinct_scenes", "distinct_scenes", 90.0, "scenes whose picture is not a copy of another scene"),
    pct("language_match", "language_match", 95.0, "narration in the chosen language"),
    Spec("judge_arc", "judge_arc", "", 0.7, True, "Gemini judge, 1-5 mapped to 0-1: hook, ordered steps, recap, correct facts"),
    Spec("median_plan_seconds", "@median_seconds", "s", None, False, "topic -> scene plan (LangSmith run latency)"),
    Spec("crashed", "@crashed", "%", 10.0, False, "topics where the planner failed"),
]

class _Scripted:
    """A fake model: answers every prompt with the same text."""
    def __init__(self, text: str):
        self.text = text

    def invoke(self, prompt):
        return self.text


def offline(ctx: dict) -> list[Metric]:
    from rafeeq.teacher import story_video as sv

    many = json.dumps({"character": "a drop", "setting": "blue sky",
                       "scenes": [{"visual": f"scene {k} shows step {k}", "narration": "word " * 20} for k in range(9)]})
    one = json.dumps({"character": "", "setting": "", "scenes": [{"visual": "only one", "narration": "x"}]})
    checks = {}
    try:
        checks["trims_extra_scenes"] = len(sv.plan_story("the water cycle", scenes=4, language="en", llm=_Scripted(many))["scenes"]) == 4
    except Exception:                                                              # noqa: BLE001
        checks["trims_extra_scenes"] = False
    for name, call in (("rejects_too_few_scenes", lambda: sv.plan_story("the water cycle", 4, "en", llm=_Scripted(one))),
                       ("rejects_unknown_style", lambda: sv.plan_story("the water cycle", 4, "en", style="neon", llm=_Scripted(many))),
                       ("rejects_non_json_answer", lambda: sv.plan_story("the water cycle", 4, "en", llm=_Scripted("sorry, I can't"))),
                       ("rejects_empty_topic", lambda: sv.plan_story("  ", 4, "en", llm=_Scripted(many)))):
        try:
            call()
            checks[name] = False
        except sv.StoryError:
            checks[name] = True
        except Exception:                                                          # noqa: BLE001 - a crash is not a clean refusal
            checks[name] = False
    passed = sum(checks.values())
    return [Metric("planner_guards", 100 * passed / len(checks), "%", gate=100.0,
                   note="scripted fake model: extra scenes trimmed; too few scenes, unknown style, non-JSON answer, empty topic refused with a clear message",
                   detail=checks)]


def run(ctx: dict) -> list[Metric]:
    import sys

    return observe.run_case_suite(sys.modules[__name__], ctx)

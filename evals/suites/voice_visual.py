"""Smart montage › Voice to Visual (M5, FUTURE feature): which spoken moments of a lecture get a picture.

Live cases: three developer-written lecture transcripts (caption lines of 30 s, English and Arabic). Every line is
labelled by hand: TEACHING lines (a process, a structure, an object worth showing) vs OTHER lines (greetings,
announcements, questions to the class). The planner the montage page uses runs on them (the app's LLM, Gemini as the
fallback). Measured: moments that land on teaching lines (precision), how many teaching lines are covered, kinds and
prompts that follow the rules, and a judge grade of the picture prompts.

Offline checks (always run): clean_plan() repairs or drops bad moments (overlap, outside the video, mm:ss times,
unknown kind, missing prompt)."""
from __future__ import annotations

import json
import os

from evals import judge, observe
from evals.core import Metric

NAME = "voice_visual"
DESCRIPTION = "Voice to Visual (future): planned moments on teaching lines, coverage, rule-following prompts, judge grade; plan clean-up"
LIVE = True
LINE = 30.0

# (case id, language, [(text, is_teaching_line)])
LECTURES = [
    ("en-heart", "en", [
        ("Good morning everyone, welcome back, I hope you had a nice weekend.", False),
        ("Today we talk about the human heart and how blood moves through it.", False),
        ("The heart has four chambers: two atria at the top and two ventricles at the bottom.", True),
        ("Blood without oxygen comes from the body into the right atrium, then passes down to the right ventricle.", True),
        ("The right ventricle pumps it to the lungs, where it picks up oxygen and gets rid of carbon dioxide.", True),
        ("Before I continue, remember the quiz is next Tuesday, so please revise chapter three.", False),
        ("Oxygen-rich blood comes back to the left atrium and then flows into the left ventricle.", True),
        ("The left ventricle has the thickest wall because it pumps blood to the whole body through the aorta.", True),
        ("Valves between the chambers close like one-way doors, and that closing is the lub-dub sound you hear.", True),
        ("Any questions so far? Okay, good, let's keep going.", False),
        ("So to summarise the path: body, right side, lungs, left side, and back to the body.", True),
        ("That is all for today, see you next week and don't forget your homework.", False),
    ]),
    ("ar-water-cycle", "ar", [
        ("السلام عليكم يا شباب، أهلاً بكم في درس اليوم.", False),
        ("درسنا اليوم عن دورة الماء في الطبيعة.", False),
        ("أولاً تسخن الشمس مياه البحار والمحيطات فيتبخر الماء ويصعد إلى الجو على شكل بخار.", True),
        ("عندما يرتفع البخار يبرد، فيتكاثف ويتحول إلى قطرات صغيرة تكوّن السحب.", True),
        ("تذكروا أن الواجب يسلم يوم الخميس على المنصة.", False),
        ("عندما تثقل القطرات في السحب تسقط على الأرض مطراً أو ثلجاً أو برداً، وهذا هو الهطول.", True),
        ("جزء من الماء يجري على سطح الأرض في الأنهار ويعود إلى البحر، وجزء يتسرب إلى باطن الأرض كمياه جوفية.", True),
        ("هل هناك أي سؤال؟ ممتاز.", False),
        ("النباتات أيضاً تطلق بخار الماء من أوراقها، وهذا يسمى النتح.", True),
        ("إذن الدورة هي: تبخر، ثم تكاثف، ثم هطول، ثم جريان، وتتكرر من جديد.", True),
        ("شكراً لكم، نلتقي في الحصة القادمة إن شاء الله.", False),
    ]),
    ("en-hash-table", "en", [
        ("Hi everyone, can you all hear me? Great, let's start.", False),
        ("Today's topic is the hash table, one of the most used data structures.", False),
        ("A hash table is an array of buckets; a hash function turns each key into the index of a bucket.", True),
        ("For example the key 'apple' might hash to 3, so its value is stored in bucket number 3.", True),
        ("Sometimes two different keys hash to the same bucket. That is called a collision.", True),
        ("One fix is chaining: each bucket keeps a small linked list of all the entries that landed there.", True),
        ("By the way, the lab this week is in room 204, not the usual room.", False),
        ("When the table gets too full we resize it: make a bigger array and re-insert every key.", True),
        ("Thanks to this, looking up a key takes constant time on average.", False),
        ("Okay, that's it for the theory, any questions before the break?", False),
    ]),
]


def _segments(lines):
    return [{"start": k * LINE, "end": k * LINE + LINE - 2, "text": t} for k, (t, _) in enumerate(lines)]


def cases() -> list[dict]:
    return [{"id": cid, "input": {"language": lang, "segments": _segments(lines), "total": LINE * len(lines)},
             "expected": {"teaching_lines": [k for k, (_, teach) in enumerate(lines) if teach]}}
            for cid, lang, lines in LECTURES]


def _planners():
    from rafeeq.teacher import voice_visual as vv

    out = []
    try:
        from rafeeq.knowledge.topics.room_topics import default_propose

        if (main := default_propose()) is not None:
            out.append(("app LLM", main))
    except Exception:                                                              # noqa: BLE001
        pass
    if vv.gemini_available():
        out.append(("gemini", vv.gemini_propose))
    return out


def missing() -> str:
    return "" if _planners() else "no planner LLM (LLM_API_KEY or GOOGLE_API_KEY) is set"


def target(inp: dict) -> dict:
    """Same order as the montage page: the app's LLM first, Gemini when it fails or finds nothing."""
    from rafeeq.teacher import voice_visual as vv

    last = None
    for name, propose in _planners():
        try:
            plan = vv.plan_visuals(inp["segments"], inp["total"], propose)
        except vv.PlanError as e:
            last = e
            continue
        if plan:
            return {"planner": name, "plan": plan}
    if last is not None:
        raise last
    return {"planner": "", "plan": []}


def _line(t: float) -> int:
    return int(t // LINE)


def has_plan(i, o, e):
    return float(bool(o["plan"]))


def on_teaching_line(i, o, e):
    if not o["plan"]:
        return None
    teach = set(e["teaching_lines"])
    return sum(_line(m["start"]) in teach for m in o["plan"]) / len(o["plan"])


def teaching_coverage(i, o, e):
    hit = {_line(m["start"]) for m in o["plan"]} & set(e["teaching_lines"])
    wanted = max(3, min(12, round(i["total"] / 300)))            # what the planner is asked for (one per 5 min, 3 to 12)
    return len(hit) / min(wanted, len(e["teaching_lines"]))


def rules_followed(i, o, e):
    """diagram/illustration have an image prompt, animation has a GIF query, each moment 3-12 s, no overlaps."""
    if not o["plan"]:
        return None
    good = []
    for k, m in enumerate(o["plan"]):
        ok = m["kind"] in ("diagram", "illustration", "animation") and 3 <= m["end"] - m["start"] <= 12
        ok &= bool(m["query"]) if m["kind"] == "animation" else bool(m["prompt"])
        ok &= k == 0 or m["start"] >= o["plan"][k - 1]["end"]
        good.append(ok)
    return sum(good) / len(good)


def label_language(i, o, e):
    if not o["plan"]:
        return None
    return sum(observe.in_language(m["label"], i["language"]) for m in o["plan"]) / len(o["plan"])


def judge_prompts(i, o, e):
    if not o["plan"] or not judge.available():
        return None
    said = {_line(m["start"]): i["segments"][_line(m["start"])]["text"] for m in o["plan"] if _line(m["start"]) < len(i["segments"])}
    material = json.dumps([{"said": said.get(_line(m["start"]), ""), "kind": m["kind"], "label": m["label"],
                            "image_prompt": m["prompt"], "gif_query": m["query"]} for m in o["plan"]], ensure_ascii=False)
    return judge.grade("- each picture shows exactly the idea that was said at that moment\n"
                       "- diagrams are used for processes and structures, illustrations for things\n"
                       "- the image prompt describes shapes and layout and does not ask for written words", material)


SCORERS = {f.__name__: f for f in (has_plan, on_teaching_line, teaching_coverage, rules_followed, label_language, judge_prompts)}


def summarize(rows: list[dict]) -> list[Metric]:
    return [
        Metric("lectures_with_plan", observe.pct(rows, "has_plan"), "%", gate=100.0, note=f"{len(rows)} labelled lectures that got at least one moment"),
        Metric("moments_on_teaching_lines", observe.pct(rows, "on_teaching_line"), "%", gate=80.0,
               note="precision: planned moments that start on a line labelled as teaching (not greetings / announcements)"),
        Metric("teaching_coverage", observe.pct(rows, "teaching_coverage"), "%", gate=80.0,
               note="of the pictures the planner is asked for, how many land on different teaching lines"),
        Metric("rules_followed", observe.pct(rows, "rules_followed"), "%", gate=95.0, note="kind, prompt or GIF query, 3-12 s, no overlap"),
        Metric("label_language", observe.pct(rows, "label_language"), "%", gate=90.0, note="labels in the lecture's language"),
        Metric("judge_prompts", observe.mean(rows, "judge_prompts"), "", gate=0.7, note="Gemini judge, 1-5 mapped to 0-1: the picture matches what was said"),
        Metric("median_plan_seconds", observe.median_seconds(rows), "s", higher_is_better=False, note="captions -> plan"),
        Metric("crashed", observe.error_rate(rows), "%", higher_is_better=False, gate=10.0, note="lectures where planning failed"),
    ]


def offline(ctx: dict) -> list[Metric]:
    from rafeeq.teacher.voice_visual import clean_plan

    raw = [
        {"start": "1:00", "end": "1:06", "kind": "diagram", "label": "heart", "prompt": "four chambers"},     # mm:ss -> 60 s
        {"start": 62, "end": 70, "kind": "diagram", "label": "overlap", "prompt": "x"},                     # overlaps -> dropped
        {"start": 500, "kind": "diagram", "label": "late", "prompt": "x"},                                   # after the video -> dropped
        {"start": 100, "end": 200, "kind": "sticker", "label": "valves", "prompt": ""},                     # unknown kind, long, no prompt
        {"start": 150, "end": 151, "kind": "animation", "label": "pump", "query": ""},                       # animation without query
        {"end": 20, "kind": "diagram"},                                                                      # no start -> dropped
        "not a moment",
    ]
    plan = clean_plan(raw, 300)
    checks = {
        "mm:ss read": bool(plan) and plan[0]["start"] == 60,
        "overlap dropped": not any(m["label"] == "overlap" for m in plan),
        "outside video dropped": not any(m["label"] == "late" for m in plan),
        "unknown kind -> illustration, label as prompt, end capped": any(m["label"] == "valves" and m["kind"] == "illustration"
                                                                          and m["prompt"] == "valves" and m["end"] - m["start"] <= 12 for m in plan),
        "animation without query -> illustration": any(m["label"] == "pump" and m["kind"] == "illustration" for m in plan),
        "every kept moment ≥ 3 s": all(m["end"] - m["start"] >= 3 for m in plan),
        "never raises on junk": clean_plan("junk", 10) == [] and clean_plan(None, 10) == [],
    }
    return [Metric("plan_cleanup", 100 * sum(checks.values()) / len(checks), "%", gate=100.0,
                   note="bad LLM moments are repaired or dropped before they reach the editor", detail=checks)]


def run(ctx: dict) -> list[Metric]:
    import sys

    return observe.run_case_suite(sys.modules[__name__], ctx)

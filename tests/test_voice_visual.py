"""Voice-to-Visual: the plan the LLM returns is cleaned before anything is drawn."""
from rafeeq.teacher import voice_visual as vv

SEGS = [{"start": 0.0, "end": 4.0, "text": "اهلا بيكم"},
        {"start": 10.0, "end": 16.0, "text": "the linked list is a chain of nodes, each node points to the next"},
        {"start": 40.0, "end": 45.0, "text": "and when we push to the stack it goes on top"}]


def test_clean_plan_keeps_good_moments_in_order():
    raw = [{"start": 40, "end": 46, "kind": "diagram", "label": "Stack push", "prompt": "boxes stacked, arrow on top"},
           {"start": 10, "end": 18, "kind": "diagram", "label": "Linked list", "prompt": "circles joined by arrows"}]
    out = vv.clean_plan(raw, 60.0)
    assert [o["start"] for o in out] == [10.0, 40.0]
    assert all(o["kind"] == "diagram" for o in out)


def test_clean_plan_drops_bad_and_overlapping_and_clamps():
    raw = [{"start": "x"}, {"start": 5, "end": 100, "kind": "illustration", "prompt": "a tree"},
           {"start": 6, "end": 9, "kind": "diagram", "prompt": "overlaps the first"},
           {"start": 70, "kind": "diagram", "prompt": "after the end"},
           {"start": 30, "kind": "weird", "label": "unknown kind"}, "not a dict"]
    out = vv.clean_plan(raw, 60.0)
    assert [o["start"] for o in out] == [5.0, 30.0]
    assert out[0]["end"] == 5.0 + vv.MAX_SECONDS
    assert out[1]["kind"] == "illustration" and out[1]["prompt"] == "unknown kind"


def test_animation_without_query_becomes_illustration():
    out = vv.clean_plan([{"start": 3, "end": 8, "kind": "animation", "label": "clap"}], 60.0)
    assert out[0]["kind"] == "illustration"


def test_plan_visuals_uses_llm_and_survives_failure():
    seen = {}

    def propose(system, user):
        seen["user"] = user
        return {"visuals": [{"start": 10, "end": 16, "kind": "diagram", "label": "Linked list", "prompt": "nodes and arrows"}]}

    assert len(vv.plan_visuals(SEGS, 60.0, propose)) == 1 and "linked list" in seen["user"]
    for broken in (lambda s, u: 1 / 0, lambda s, u: None, lambda s, u: "not json"):
        try:
            vv.plan_visuals(SEGS, 60.0, broken)
            raise AssertionError("should raise")
        except vv.PlanError:
            pass
    assert vv.plan_visuals(SEGS, 60.0, None) == []
    assert vv.plan_visuals([], 60.0, propose) == []


def test_prompt_never_asks_for_text_and_key_is_stable():
    item = vv.clean_plan([{"start": 10, "kind": "diagram", "prompt": "nodes and arrows"}], 60.0)[0]
    assert "no text" in vv.image_prompt(item)
    assert vv.job_key("a1", item) == vv.job_key("a1", dict(item)) != vv.job_key("a2", item)


def test_times_and_shapes_the_models_like_to_send():
    out = vv.plan_visuals(SEGS, 120.0, lambda s, u: {"moments": [{"start": "0:10", "end": "0:16s", "kind": "Diagram",
                                                                   "label": "x", "prompt": "nodes"}]})
    assert out[0]["start"] == 10.0 and out[0]["end"] == 16.0 and out[0]["kind"] == "diagram"
    assert len(vv.plan_visuals(SEGS, 120.0, lambda s, u: [{"start": 40, "kind": "illustration", "prompt": "a stack"}])) == 1


def test_cut_off_captions_keep_the_complete_lines():
    from rafeeq.teacher.montage import salvage_segments

    cut = '{"segments": [{"start": 0, "end": 3.5, "text": "النهارده \\"lists\\""}, {"start": 3.5, "end": 7, "text": "index"}, {"start": 7, "end": 1'
    out = salvage_segments(cut)
    assert [s["text"] for s in out] == ['النهارده "lists"', "index"]

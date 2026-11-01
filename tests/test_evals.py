"""The evaluation pipeline itself: its metric helpers must be right, or every report is wrong."""
import shutil

import pytest

from evals import core

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def test_wer_counts_word_errors_and_ignores_arabic_spelling_noise():
    assert core.wer("the cat sat", "the cat sat") == 0.0
    assert core.wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert core.wer("a b c d", "") == 1.0 and core.wer("", "") == 0.0
    assert core.wer("مَرْحَبًا بِكُمْ في الدرس", "مرحبا بكم في الدرس") == 0.0       # diacritics are not errors
    assert core.wer("المدرسة إلى", "المدرسه الى") == 0.0                              # ة/ه and أإآ/ا


def test_range_helpers():
    assert core.overlap([(0, 5), (9, 14)], [(4, 10)]) == 2.0
    assert core.length([(1, 3), (5, 6)]) == 3.0
    assert core.complement([(2, 4), (6, 7)], 10) == [(0, 2), (4, 6), (7, 10)]
    assert core.complement([], 5) == [(0, 5)]


def test_chapter_offset():
    assert core.mean_abs_offset([0, 60], [0, 70]) == 5.0
    assert core.mean_abs_offset([], [1]) is None and core.mean_abs_offset([1], []) is None


def test_metric_status_follows_the_gate_direction():
    m = core.Metric
    assert m("a", 0.9, gate=0.8).status == "PASS" and m("a", 0.7, gate=0.8).status == "FAIL"
    assert m("a", 3, higher_is_better=False, gate=5).status == "PASS" and m("a", 6, higher_is_better=False, gate=5).status == "FAIL"
    assert m("a", 1).status == "INFO" and m("a", None, gate=1).status == "SKIPPED"


def test_change_since_last_run():
    assert core.change({"value": 1.0}, {"value": 1.5}) == "+0.5"
    assert core.change({"value": 1.0}, {"value": 1.0}) == "" and core.change(None, {"value": 1}) == ""
    assert core.change({"value": None}, {"value": 1}) == ""


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_montage_suite_passes_on_the_current_code():
    from evals.suites import montage
    bad = [m.name for m in montage.run({"live": False}) if m.status == "FAIL"]
    assert bad == []


def test_every_suite_loads_and_returns_metrics_that_can_fail():
    """A suite that crashes would hide itself from the report; one that cannot FAIL measures nothing."""
    import importlib
    import pkgutil

    from evals import suites
    for info in pkgutil.iter_modules(suites.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"evals.suites.{info.name}")
        assert mod.NAME and mod.DESCRIPTION
        if info.name == "montage" and not HAS_FFMPEG:
            continue
        metrics = mod.run({"live": False})
        assert metrics and all(isinstance(m, core.Metric) for m in metrics)
        assert any(m.gate is not None for m in metrics), f"{mod.NAME} has no gated metric"


def test_language_check_used_by_the_live_suites():
    from evals import observe
    assert observe.in_language("الحلقات التكرارية في بايثون: for و while", "Arabic")
    assert not observe.in_language("Loops in Python: for and while", "Arabic")
    assert observe.in_language("Loops in Python: for and while", "English")
    assert not observe.in_language("التمثيل الضوئي", "English")


def test_case_runner_scores_each_case_and_keeps_crashes_as_results():
    """The same scoring the LangSmith evaluator uses: a crash is a result, a broken scorer is 'no score'."""
    import types

    from evals import observe

    def target(inp):
        if inp["x"] < 0:
            raise ValueError("negative")
        return {"y": inp["x"] * 2}

    mod = types.SimpleNamespace(
        NAME="fake", LIVE=True, target=target,
        cases=lambda: [{"id": "a", "input": {"x": 1}, "expected": 2}, {"id": "b", "input": {"x": 2}, "expected": 5},
                       {"id": "c", "input": {"x": -1}, "expected": 0}],
        SCORERS={"exact": lambda i, o, e: float(o["y"] == e), "broken": lambda i, o, e: 1 / 0},
        SPEC=[core.pct("exact", "exact", 50.0), core.Spec("crashed", "@crashed", "%", 10.0, False)])
    skipped = observe.run_case_suite(mod, {"live": False})
    assert [m.status for m in skipped] == ["SKIPPED", "SKIPPED"] and all(m.gate is not None for m in skipped)
    ctx = {"live": True}
    exact, crashed = observe.run_case_suite(mod, ctx)
    assert exact.value == 50.0 and exact.status == "PASS"                    # a: right, b: wrong, c: crashed (no score)
    assert crashed.value == pytest.approx(33.3) and crashed.status == "FAIL"
    rows = ctx["rows"]["fake"]
    assert all("broken" not in r["scores"] for r in rows) and rows[2]["error"].startswith("ValueError")


def test_langsmith_dataset_and_evaluator_shape(monkeypatch):
    """What is sent to LangSmith: one example per case (+ a checks example), feedback keys = the scorers."""
    import types

    from evals import judge, langsmith_eval
    monkeypatch.setattr(judge, "available", lambda: False)                    # no real Gemini call in a unit test
    from evals.suites import story_video

    rows = langsmith_eval.wanted_examples(story_video)
    assert [r["inputs"]["case_id"] for r in rows][-1] == "checks" and len(rows) == len(story_video.cases()) + 1
    ev = langsmith_eval.make_evaluator(story_video)
    case = story_video.cases()[0]
    out = {"scenes": [{"visual": f"shot {k} of the sea", "narration": "word " * 20} for k in range(case["expected"]["scenes"])]}
    res = ev(types.SimpleNamespace(outputs=out, error=None),
             types.SimpleNamespace(inputs={"case_id": case["id"], "input": case["input"]}, outputs={"expected": case["expected"]}))
    keys = {r["key"] for r in res["results"]}
    assert {"scenes_exact", "narration_rule", "language_match"} <= keys
    checks = ev(types.SimpleNamespace(outputs={"metrics": [core.Metric("planner_guards", 100.0, "%", gate=100.0).as_dict()]}, error=None),
                types.SimpleNamespace(inputs={"case_id": "checks", "checks": True}, outputs={}))
    assert checks["results"][0] == {"key": "planner_guards", "score": 100.0, "comment": ""}


def test_report_follows_the_presentation_order():
    from evals import flow
    names = [n for n, _, _ in flow.FLOW]
    assert names.index("lesson_studio") < names.index("parent_batch") < names.index("learning_platform") < names.index("voice_visual")
    assert flow.place("unknown_suite")[0] == flow.SECTIONS["Z"]


def test_learning_platform_rules_hold():
    from evals.suites import learning_platform
    bad = {m.name: m.detail for m in learning_platform.run({"live": False}) if m.status != "PASS"}
    assert bad == {}

"""The evaluation report: real runs only, and "NOT RUN" (never an invented number) when there is no data."""
import json

import pandas as pd
import pytest

from rafeeq.snap import report
from rafeeq.tutor import llm as llm_mod


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path / "study"))
    monkeypatch.setattr(llm_mod, "ask_json", lambda *a, **k: None)
    monkeypatch.setenv("TUTOR_SAMPLE_FALLBACK", "1")

    def down(*a, **k):
        raise ConnectionError("down")
    monkeypatch.setattr(report, "__name__", report.__name__)
    from rafeeq.tutor import m2
    monkeypatch.setattr(m2.requests, "post", down)
    monkeypatch.setattr(m2.requests, "get", down)


def test_no_cer_data_is_reported_as_not_run():
    assert report.cer_section(None, "x", "no data")["status"] == "NOT RUN"
    assert report.cer_section(pd.DataFrame(), "x")["status"] == "NOT RUN"


def test_cer_section_computes_the_preprocessing_effect_from_the_rows_it_is_given():
    df = pd.DataFrame([{"reader": r, "preset": p, "image": i, "type": "en", "cer": c} for r, p, i, c in [
        ("vlm", "raw", "a", 0.5), ("vlm", "pre", "a", 0.1), ("vlm", "raw", "b", 0.7), ("vlm", "pre", "b", 0.3), ("easyocr", "raw", "a", 0.2), ("easyocr", "pre", "a", 0.3)]])
    sec = report.cer_section(df, "synthetic")
    assert sec["preprocessing_effect"]["vlm"] == {"raw": 0.6, "pre": 0.2, "pre_minus_raw": -0.4, "images": 2}
    assert sec["preprocessing_effect"]["easyocr"]["pre_minus_raw"] == 0.1 and sec["baseline_available"] is True


def test_work_section_counts_real_runs_and_leaks_nothing(tmp_path):
    cases = tmp_path / "c.json"
    cases.write_text(json.dumps({"cases": [
        {"id": "a", "lang": "en", "lines": ["2 + 3 = 5", "4 × 2 = 9"], "statuses": [["ok"], ["error"]], "first_error": 2, "hint_keywords": ["arithmetic"], "solution": ["= 8"]},
        {"id": "b", "lang": "en", "lines": ["2 + 3 = 5", "x = [?]"], "statuses": [["ok"], ["check"]], "first_error": None, "expect_clarification": True}]}), encoding="utf-8")
    sec = report.work_section(cases)
    assert sec["status"] == "RAN" and sec["llm_available"] is False and sec["cases"] == 2
    assert sec["step_status_accuracy"] == {"correct": 4, "total": 4, "rate": 1.0}
    assert sec["first_error_localization"]["rate"] == 1.0 and sec["uncertainty_handling"]["rate"] == 1.0
    assert sec["hint_leaks"] == {"hints_containing_a_solution_string": 0, "hints_containing_a_line_of_working": 0}


def test_missing_files_are_not_run(tmp_path):
    assert report.work_section(tmp_path / "none.json")["status"] == "NOT RUN"
    assert report.citation_section(tmp_path / "none.json")["status"] == "NOT RUN"


def test_citation_section_says_when_the_material_is_the_sample(tmp_path):
    q = tmp_path / "q.json"
    q.write_text(json.dumps({"room_id": "r_bio101", "questions": [{"q": "ما وظيفة الميتوكوندريا؟", "lang": "ar", "answerable": True}]}), encoding="utf-8")
    sec = report.citation_section(q)
    assert sec["material_source"] == ["sample"] and "not your database" in sec["note"]
    assert sec["citation_validity"]["rate"] == 1.0


def test_the_markdown_renders_every_section():
    md = report.to_markdown({"generated": "now", "platform": "p", "cer": {"status": "NOT RUN", "reason": "r"},
                             "work": {"status": "NOT RUN", "reason": "r"}, "citations": {"status": "NOT RUN", "reason": "r"}})
    assert md.count("**NOT RUN**") == 3

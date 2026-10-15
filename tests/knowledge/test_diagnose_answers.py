"""scripts/diagnose_answers.py with a fake embedder and a fake HTTP post. No network, no weights, no key."""
import io
import json

import pytest

from eval_helpers import load_script
from kb_fakes import HashingEmbedder
from test_run_eval_hybrid import ROOM, setup

diag = load_script("diagnose_answers")


def body(content, finish="stop", reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    return json.dumps({"choices": [{"message": msg, "finish_reason": finish}], "usage": {"completion_tokens": 7, "total_tokens": 99}})


@pytest.mark.parametrize("text,stage", [
    ("", "empty"), ("NOT_COVERED", "sentinel_exact"), ("`NOT_COVERED`.", "sentinel_exact"),
    ("The excerpts lack it, so NOT_COVERED [1]", "sentinel_substring_only"),
    ("Pairs are n(n-1)/2.", "no_valid_marker"), ("Answer (1)", "no_valid_marker"), ("Answer [9]", "no_valid_marker"),
    ("Answer [1]", "accepted"), ("Answer \\[1\\]", "accepted"), ("Answer [1, 2]", "accepted"),
])
def test_classify_repeats_the_answerer_conditions(text, stage):
    assert diag.classify(text, available=2)["stage"] == stage


def run(tmp_path, post, *extra):
    import os
    os.environ["LLM_API_KEY"] = "test-key-not-real"
    store_dir, bench = setup(tmp_path)
    out, err = io.StringIO(), io.StringIO()
    code = diag.main(["--benchmark", str(bench), "--store-dir", str(store_dir), "--room-id", ROOM, "--llm-min-interval", "0", "--llm-max-retries", "0", *extra],
                     embedder=HashingEmbedder(), post=post, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_dry_run_makes_no_llm_call_and_writes_nothing(tmp_path):
    calls = []
    code, out, _ = run(tmp_path, lambda *a: calls.append(a) or (200, "{}"), "--ids", "q1", "--dry-run")
    assert code == 0 and calls == [] and "DRY RUN" in out and not (tmp_path / "d").exists()


def test_records_raw_response_and_matches_the_answerer(tmp_path):
    post = lambda url, h, b, t: (200, body("Gradient descent updates weights [1].", reasoning="thinking"))
    code, out, _ = run(tmp_path, post, "--ids", "q1", "--out-dir", str(tmp_path / "d"))
    row = json.loads((tmp_path / "d" / "diagnosis.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert code == 0 and row["stage"] == "accepted" and row["grounded_by_answerer"] and "WARNING" not in row
    assert row["raw_reasoning"] == "thinking" and row["finish_reason"] == "stop" and row["usage"]["total_tokens"] == 99


def test_sentinel_in_longer_text_is_diagnosed(tmp_path):
    post = lambda url, h, b, t: (200, body("I should reply NOT_COVERED only if missing. Gradient is here [1]."))
    run(tmp_path, post, "--ids", "q1", "--out-dir", str(tmp_path / "d"))
    row = json.loads((tmp_path / "d" / "diagnosis.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["stage"] == "sentinel_substring_only" and row["grounded_by_answerer"] is False


def test_http_429_is_an_error_not_an_abstention(tmp_path):
    code, out, _ = run(tmp_path, lambda *a: (429, "rate limited"), "--ids", "q1", "--out-dir", str(tmp_path / "d"))
    row = json.loads((tmp_path / "d" / "diagnosis.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["stage"] == "http_error" and "429" in row["error"]


def test_quota_guard_and_no_overwrite_and_no_key_saved(tmp_path):
    # the fake key must never reach the saved files
    assert run(tmp_path, None, "--ids", "q1", "o1", "--repeat", "4", "--out-dir", str(tmp_path / "d"))[0] == 1   # 8 calls > 6
    (tmp_path / "d").mkdir()
    assert run(tmp_path, None, "--ids", "q1", "--out-dir", str(tmp_path / "d"))[0] == 1
    post = lambda url, h, b, t: (200, body("ok [1]"))
    run(tmp_path, post, "--ids", "q1", "--out-dir", str(tmp_path / "d2"))
    assert "test-key-not-real" not in "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "d2").iterdir())

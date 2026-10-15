"""Evaluation logic: per-question scoring, aggregation, RAG-layer scoring, judge, LLM client, report.

Scripted retrieval results are used wherever an exact number is asserted, so every expectation can be
checked by hand from the comments. No model, network or LLM is used.
"""

import json

import pytest

from rafeeq.knowledge.evaluation.dataset import BenchmarkItem
from rafeeq.knowledge.evaluation.judge import (
    LLMError,
    LLMJudge,
    OpenAICompatibleLLM,
    parse_verdict,
)
from rafeeq.knowledge.evaluation.report import csv_text, recommendations, render_report
from rafeeq.knowledge.evaluation.runner import (
    RecordingRetriever,
    aggregate_rag,
    aggregate_retrieval,
    evaluate_rag_item,
    run_evaluation,
    score_retrieval,
    threshold_sweep,
)

KS = (1, 3, 5)


def item(id="q1", pages=(10, 20), chunk_ids=(), qtype="explanation", language="en", **kw):
    base = dict(id=id, language=language, qtype=qtype, question=f"question {id}", answerable=True, status="verified",
                file_name="book.pdf", expected_pages=list(pages), expected_chunk_ids=list(chunk_ids),
                evidence_quotes=["x" * 30], reference_answer="reference", key_facts=[["fact"]])
    base.update(kw)
    return BenchmarkItem(**base)


def oos(id="o1", language="en"):
    return BenchmarkItem(id=id, language=language, qtype="out_of_scope", question=f"off topic {id}", answerable=False,
                         status="verified", absent_terms=["zebra"])


def hit(file_name, page, score=0.5, chunk_id=None, text="chunk text"):
    return {"file_name": file_name, "page": page, "score": score, "chunk_id": chunk_id or f"{file_name}-p{page}", "text": text, "topic_id": "topic_general"}


# ------------------------------------------------------------- score_retrieval

def test_scored_example_matches_the_hand_computation():
    # expected pages {10, 20} in book.pdf. Ranks: 1 book p5, 2 book p10 (relevant), 3 other p20 (wrong doc), 4 book p20 (relevant), 5 book p7
    hits = [hit("book.pdf", 5), hit("book.pdf", 10), hit("other.pdf", 20), hit("book.pdf", 20), hit("book.pdf", 7)]
    r = score_retrieval(item(), hits, KS, min_score=0.35)
    assert [h["relevant_page"] for h in r["retrieved"]] == [False, True, False, True, False]
    assert (r["hit@1"], r["hit@3"], r["hit@5"]) == (0.0, 1.0, 1.0)
    assert r["precision@3"] == pytest.approx(1 / 3) and r["precision@5"] == pytest.approx(2 / 5)
    assert (r["recall@1"], r["recall@3"], r["recall@5"]) == (0.0, 0.5, 1.0)     # pages 10 then 20 found
    assert (r["full_recall@3"], r["full_recall@5"]) == (0.0, 1.0)
    assert r["rr"] == 0.5 and r["first_relevant_rank"] == 2
    assert r["other_document_share@3"] == pytest.approx(1 / 3) and r["other_document_share@5"] == pytest.approx(1 / 5)
    assert r["failure_category"] == "late_hit"


def test_rank_one_hit_is_ok_and_partial_coverage_is_flagged():
    full = score_retrieval(item(pages=(10,)), [hit("book.pdf", 10)], KS, 0.35)
    assert full["failure_category"] == "ok" and full["rr"] == 1.0
    partial = score_retrieval(item(), [hit("book.pdf", 10), hit("book.pdf", 99)], KS, 0.35)
    assert partial["failure_category"] == "ok_partial_coverage" and partial["recall@5"] == 0.5


def test_miss_categories():
    near = score_retrieval(item(pages=(10,)), [hit("book.pdf", 12), hit("book.pdf", 90)], KS, 0.35)
    assert near["failure_category"] == "miss_near_page" and near["hit@5"] == 0.0 and near["rr"] == 0.0
    far = score_retrieval(item(pages=(10,)), [hit("book.pdf", 13), hit("book.pdf", 90)], KS, 0.35)
    assert far["failure_category"] == "miss"
    wrong = score_retrieval(item(pages=(10,)), [hit("other.pdf", 10), hit("other.pdf", 11)], KS, 0.35)
    assert wrong["failure_category"] == "miss_wrong_document" and wrong["other_document_share@5"] == 1.0
    nothing = score_retrieval(item(pages=(10,)), [], KS, 0.35)
    assert nothing["failure_category"] == "miss_wrong_document" and nothing["top1_score"] is None and nothing["gate_passed"] is False


def test_only_the_top_depth_chunks_are_considered():
    hits = [hit("book.pdf", p) for p in (1, 2, 3, 4, 5)] + [hit("book.pdf", 10)]       # relevant chunk is at rank 6
    r = score_retrieval(item(pages=(10,)), hits, KS, 0.35)
    assert r["hit@5"] == 0.0 and len(r["retrieved"]) == 5


def test_chunk_level_metrics_are_computed_only_when_chunk_ids_exist():
    hits = [hit("book.pdf", 10, chunk_id="c1"), hit("book.pdf", 10, chunk_id="c2"), hit("book.pdf", 20, chunk_id="c3")]
    strict = score_retrieval(item(chunk_ids=("c2", "c9")), hits, KS, 0.35)
    assert (strict["hit_chunk@1"], strict["hit_chunk@3"]) == (0.0, 1.0)
    assert strict["recall_chunk@3"] == 0.5 and strict["rr_chunk"] == 0.5
    assert strict["hit@1"] == 1.0                          # page level is lenient: c1 is also on page 10
    assert "rr_chunk" not in score_retrieval(item(), hits, KS, 0.35)


def test_gate_and_out_of_scope_records():
    r = score_retrieval(oos(), [hit("book.pdf", 1, score=0.30), hit("book.pdf", 2, score=0.20)], KS, min_score=0.35)
    assert r["top1_score"] == 0.30 and r["gate_passed"] is False and "hit@1" not in r
    passed = score_retrieval(oos(), [hit("book.pdf", 1, score=0.36)], KS, min_score=0.35)
    assert passed["gate_passed"] is True
    exact = score_retrieval(oos(), [hit("book.pdf", 1, score=0.35)], KS, min_score=0.35)
    assert exact["gate_passed"] is True                    # the Answerer keeps chunks with score >= min_score


# ----------------------------------------------------------------- aggregation

def test_aggregate_by_group_and_overall_means():
    a = score_retrieval(item("a", pages=(10,), qtype="definition", language="en"), [hit("book.pdf", 10)], KS, 0.35)              # rank 1
    b = score_retrieval(item("b", pages=(10,), qtype="definition", language="ar"), [hit("book.pdf", 3), hit("book.pdf", 10)], KS, 0.35)   # rank 2
    c = score_retrieval(item("c", pages=(10,), qtype="comparison", language="ar"), [hit("book.pdf", 3)], KS, 0.35)                # miss
    o = score_retrieval(oos(), [hit("book.pdf", 1)], KS, 0.35)
    out = aggregate_retrieval([a, b, c, o], KS)
    overall = out["overall"]
    assert overall["n"] == 3                                                     # out-of-scope is not a retrieval question
    assert overall["hit@1"] == pytest.approx(1 / 3) and overall["hit@3"] == pytest.approx(2 / 3)
    assert overall["mrr@5"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert out["by_qtype"]["definition"]["n"] == 2 and out["by_qtype"]["definition"]["hit@1"] == 0.5
    assert out["by_language"]["ar"]["n"] == 2 and out["by_language"]["ar"]["hit@3"] == 0.5
    low, high = overall["hit@3_ci95"]
    assert low < 2 / 3 < high
    assert out["failure_categories"] == {"late_hit": 1, "miss": 1, "ok": 1}


def test_chunk_level_aggregate_only_covers_items_with_chunk_ids():
    with_ids = score_retrieval(item("a", chunk_ids=("c1",)), [hit("book.pdf", 10, chunk_id="c1")], KS, 0.35)
    without = score_retrieval(item("b"), [hit("book.pdf", 10)], KS, 0.35)
    out = aggregate_retrieval([with_ids, without], KS)
    assert out["chunk_level"]["n"] == 1 and out["chunk_level"]["hit_chunk@1"] == 1.0
    assert "chunk_level" not in aggregate_retrieval([without], KS)


def test_threshold_sweep_is_descriptive_shares_of_top1_scores():
    records = [
        score_retrieval(item("a", pages=(1,)), [hit("book.pdf", 1, score=0.8)], KS, 0.35),
        score_retrieval(item("b", pages=(1,)), [hit("book.pdf", 1, score=0.4)], KS, 0.35),
        score_retrieval(oos("o1"), [hit("book.pdf", 1, score=0.3)], KS, 0.35),
        score_retrieval(oos("o2"), [hit("book.pdf", 1, score=0.5)], KS, 0.35),
    ]
    rows = {r["threshold"]: r for r in threshold_sweep(records, [0.0, 0.35, 0.6, 0.9])}
    assert (rows[0.0]["answerable_pass_rate"], rows[0.0]["out_of_scope_pass_rate"]) == (1.0, 1.0)
    assert (rows[0.35]["answerable_pass_rate"], rows[0.35]["out_of_scope_pass_rate"]) == (1.0, 0.5)
    assert (rows[0.6]["answerable_pass_rate"], rows[0.6]["out_of_scope_pass_rate"]) == (0.5, 0.0)
    assert (rows[0.9]["answerable_pass_rate"], rows[0.9]["out_of_scope_pass_rate"]) == (0.0, 0.0)


# ------------------------------------------------------------------- run_evaluation

class ScriptedRetriever:
    """Returns a fixed hit list per question text."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def retrieve(self, room_id, query, k=6):
        self.calls.append((room_id, query, k))
        return list(self.script.get(query, []))[:k]


def test_only_verified_items_are_scored_and_drafts_are_listed():
    draft = item("d1", status="draft", expected_pages=[])
    retriever = ScriptedRetriever({"question q1": [hit("book.pdf", 10)]})
    results = run_evaluation([item("q1", pages=(10,)), draft], retriever, room_id="r", ks=KS, min_score=0.35)
    assert results["items"]["scored"] == 1 and results["items"]["skipped_draft"] == ["d1"]
    assert [c[1] for c in retriever.calls] == ["question q1"] and retriever.calls[0][2] == 5   # depth = max K
    assert results["rag"]["status"] == "not_run"


def test_nothing_verified_refuses_to_produce_numbers():
    with pytest.raises(ValueError, match="no verified items"):
        run_evaluation([item("d1", status="draft", expected_pages=[])], ScriptedRetriever({}), room_id="r", ks=KS, min_score=0.35)


# -------------------------------------------------------------------- RAG layer

class ScriptedAnswerer:
    """Stands in for Answerer: sets what the recorder saw and returns a scripted result."""

    def __init__(self, recorder, seen, result):
        self.recorder, self.seen, self.result = recorder, seen, result

    def answer(self, room_id, question):
        self.recorder.last = list(self.seen)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def rag(item_, seen, result, judge=None, min_score=0.35):
    recorder = RecordingRetriever(None)
    return evaluate_rag_item(item_, ScriptedAnswerer(recorder, seen, result), recorder, "r", min_score, judge)


GOOD = {"answer_md": "It fetches documents [1].\n\n**Sources:**\n- [1] book.pdf, page 10", "grounded": True,
        "citations": [{"file_name": "book.pdf", "page": 10}]}
DECLINED = {"answer_md": "This topic is not covered in the provided course material.", "grounded": False, "citations": []}


def test_answered_question_gets_citation_and_key_fact_scores():
    q = item(pages=(10,), key_facts=[["documents"], ["retrieval", "retrieve"]])
    r = rag(q, [hit("book.pdf", 10, score=0.6)], GOOD)
    assert r["outcome"] == "answered" and r["grounded"] is True
    assert r["citation"]["hit"] == 1.0 and r["citation"]["precision"] == 1.0
    assert r["key_fact_coverage"] == 0.5 and r["missing_facts"] == [["retrieval", "retrieve"]]   # the Sources footer is not scanned
    assert r["excerpts_given"] == [{"chunk_id": "book.pdf-p10", "file_name": "book.pdf", "page": 10, "score": 0.6}]


def test_wrong_page_citation_scores_zero():
    bad = {**GOOD, "citations": [{"file_name": "book.pdf", "page": 99}]}
    r = rag(item(pages=(10,)), [hit("book.pdf", 99, score=0.6)], bad)
    assert r["citation"]["hit"] == 0.0 and r["citation"]["precision"] == 0.0


def test_declined_answerable_question_is_a_false_abstention_with_its_stage():
    gate = rag(item(), [hit("book.pdf", 10, score=0.2)], DECLINED)                  # nothing reaches min_score
    assert gate["outcome"] == "false_abstention" and gate["abstention_stage"] == "retrieval_gate" and gate["key_fact_coverage"] == 0.0
    llm = rag(item(), [hit("book.pdf", 10, score=0.6)], DECLINED)                   # LLM or citation check declined
    assert llm["abstention_stage"] == "llm_or_citation_check"


def test_out_of_scope_outcomes():
    ok = rag(oos(), [hit("book.pdf", 1, score=0.5)], DECLINED)
    assert ok["outcome"] == "correct_abstention"
    bad = rag(oos(), [hit("book.pdf", 1, score=0.5)], GOOD)
    assert bad["outcome"] == "unsupported_answer" and "citation" not in bad


def test_a_failing_answerer_is_recorded_not_hidden():
    r = rag(item(), [], RuntimeError("HTTP 429"))
    assert r["error"] == "RuntimeError: HTTP 429" and "outcome" not in r


class FixedJudge:
    def __init__(self, verdict=None, error=None):
        self.verdict, self.error, self.calls = verdict, error, []

    def judge(self, question, reference, excerpts, answer):
        self.calls.append((question, reference, [e["page"] for e in excerpts], answer))
        if self.error:
            raise self.error
        return self.verdict


def test_judge_sees_reference_excerpts_and_the_clean_answer():
    judge = FixedJudge({"correctness": "correct", "faithfulness": "faithful", "reason": "ok"})
    r = rag(item(pages=(10,)), [hit("book.pdf", 10, score=0.6), hit("book.pdf", 11, score=0.1)], GOOD, judge)
    assert r["judge"]["correctness"] == "correct"
    question, reference, pages, answer = judge.calls[0]
    assert reference == "reference" and pages == [10] and answer == "It fetches documents ."   # below-gate chunk not shown; no footer


def test_judge_failure_is_recorded_and_not_scored():
    r = rag(item(pages=(10,)), [hit("book.pdf", 10, score=0.6)], GOOD, FixedJudge(error=LLMError("bad json")))
    assert "judge" not in r and r["judge_error"] == "LLMError: bad json"


def test_aggregate_rag_rates():
    judge_ok = FixedJudge({"correctness": "correct", "faithfulness": "faithful", "reason": ""})
    judge_bad = FixedJudge({"correctness": "incorrect", "faithfulness": "unfaithful", "reason": ""})
    records = [
        rag(item("a", pages=(10,), key_facts=[["documents"]]), [hit("book.pdf", 10, score=0.6)], GOOD, judge_ok),   # answered, correct
        rag(item("b", pages=(10,), key_facts=[["zzz"]]), [hit("book.pdf", 10, score=0.6)], GOOD, judge_bad),        # answered, wrong
        rag(item("c", pages=(10,)), [hit("book.pdf", 10, score=0.2)], DECLINED),                                       # false abstention (gate)
        rag(oos("o1"), [hit("book.pdf", 1, score=0.2)], DECLINED),                                                     # correct abstention
        rag(oos("o2"), [hit("book.pdf", 1, score=0.6)], GOOD),                                                         # unsupported answer
        rag(item("d"), [], RuntimeError("boom")),                                                                      # error: excluded
    ]
    out = aggregate_rag(records, judge_used=True)
    assert out["n_scored"] == 5 and out["n_errors"] == 1 and out["errors"][0]["id"] == "d"
    assert out["answerable"]["answered_rate"] == pytest.approx(2 / 3) and out["answerable"]["false_abstention_rate"] == pytest.approx(1 / 3)
    assert out["out_of_scope"]["correct_abstention_rate"] == 0.5 and out["out_of_scope"]["unsupported_answer_ids"] == ["o2"]
    assert out["abstention_stages"] == {"retrieval_gate": 2}
    assert out["abstention_accuracy"] == pytest.approx(3 / 5)                    # right: a, b (answered), o1 (declined); wrong: c (false abstention), o2 (unsupported)
    assert out["citations"]["hit_rate"] == 1.0 and out["citations"]["n_answers"] == 2
    assert out["key_fact_coverage"]["mean_when_answered"] == 0.5                  # a: 1.0, b: 0.0
    assert out["key_fact_coverage"]["mean_end_to_end"] == pytest.approx(1 / 3)    # a 1.0, b 0.0, c 0.0 (declined counts as 0)
    assert out["judge"]["correct_rate_when_judged"] == 0.5 and out["judge"]["correct_rate_end_to_end"] == pytest.approx(1 / 3)
    assert out["judge"]["faithful_rate_when_judged"] == 0.5


def test_aggregate_rag_without_a_judge_says_so():
    out = aggregate_rag([rag(item(pages=(10,)), [hit("book.pdf", 10, score=0.6)], GOOD)], judge_used=False)
    assert out["judge"]["status"] == "not_run" and "NOT evaluated" in out["judge"]["reason"]


def test_run_evaluation_with_an_answerer_factory_records_what_the_answerer_saw():
    retriever = ScriptedRetriever({"question q1": [hit("book.pdf", 10, score=0.7)], "off topic o1": [hit("book.pdf", 4, score=0.2)]})
    seen_by_factory = []

    class EchoAnswerer:
        def __init__(self, recorder):
            self.recorder = recorder

        def answer(self, room_id, question):
            seen_by_factory.append(self.recorder.retrieve(room_id, question, 6))
            return GOOD if question == "question q1" else DECLINED

    results = run_evaluation([item("q1", pages=(10,)), oos("o1")], retriever, room_id="r", ks=KS, min_score=0.35,
                             answerer_factory=lambda recorder: EchoAnswerer(recorder))
    assert results["rag"]["status"] == "run" and len(seen_by_factory) == 2
    by_id = {r["id"]: r for r in results["per_question"]}
    assert by_id["q1"]["rag"]["outcome"] == "answered" and by_id["o1"]["rag"]["outcome"] == "correct_abstention"
    assert results["retrieval_gate"] == {"note": results["retrieval_gate"]["note"], "answerable_pass_rate": 1.0, "out_of_scope_pass_rate": 0.0}


# ------------------------------------------------------------------------ judge

def test_parse_verdict_accepts_wrapped_json_and_strips_reasoning():
    raw = '<think>hmm</think>Here you go: {"correctness": "partially_correct", "faithfulness": "faithful", "reason": "missed one point"}'
    assert parse_verdict(raw) == {"correctness": "partially_correct", "faithfulness": "faithful", "reason": "missed one point"}


@pytest.mark.parametrize("raw", ["no json here", "{not json}", '{"correctness": "great", "faithfulness": "faithful"}', '{"correctness": "correct"}'])
def test_parse_verdict_rejects_unusable_output(raw):
    with pytest.raises(LLMError):
        parse_verdict(raw)


def test_judge_builds_its_prompt_from_all_inputs():
    prompts = []

    class Fake:
        def generate(self, system, prompt):
            prompts.append((system, prompt))
            return '{"correctness": "correct", "faithfulness": "faithful", "reason": "r"}'

    verdict = LLMJudge(Fake()).judge("What is RAG?", "REF TEXT", [{"page": 7, "text": "EXCERPT TEXT"}], "ANSWER TEXT")
    system, prompt = prompts[0]
    assert verdict["correctness"] == "correct" and "ONLY a JSON object" in system
    for needle in ("What is RAG?", "REF TEXT", "EXCERPT TEXT", "ANSWER TEXT", "page 7"):
        assert needle in prompt


# ------------------------------------------------------------- LLM client

class FakePost:
    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, url, headers, body, timeout):
        self.requests.append((url, dict(headers), json.loads(body), timeout))
        return self.responses.pop(0)


def ok(content):
    return 200, json.dumps({"choices": [{"message": {"content": content}}]})


def client(post, **kw):
    kw.setdefault("environ", {"MY_KEY": "secret-123"})
    return OpenAICompatibleLLM("https://api.example.com/v1/", "model-x", "MY_KEY", post=post, sleep=lambda s: None, **kw)


def test_client_sends_a_chat_completion_request_and_strips_reasoning():
    post = FakePost(ok("<think>private reasoning</think>The answer [1]."))
    llm = client(post, max_tokens=300)
    assert llm.generate("SYSTEM", "PROMPT") == "The answer [1]."
    url, headers, body, _ = post.requests[0]
    assert url == "https://api.example.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret-123"
    assert body["model"] == "model-x" and body["temperature"] == 0.0 and body["max_tokens"] == 300
    assert body["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "PROMPT"}]
    assert "secret" not in json.dumps(llm.describe())                       # the key never reaches the results file


def test_client_retries_rate_limits_then_succeeds():
    sleeps = []
    post = FakePost((429, "slow down"), (503, "busy"), ok("fine"))
    llm = OpenAICompatibleLLM("https://x/v1", "m", "K", post=post, sleep=sleeps.append, environ={"K": "k"})
    assert llm.generate("s", "p") == "fine" and len(post.requests) == 3 and sleeps == [2, 4]


def test_client_does_not_retry_auth_errors_and_gives_up_after_the_retry_budget():
    with pytest.raises(LLMError, match="HTTP 401"):
        client(FakePost((401, "bad key"))).generate("s", "p")
    post = FakePost(*[(429, "limit")] * 4)
    with pytest.raises(LLMError, match="HTTP 429"):
        client(post, max_retries=3).generate("s", "p")
    assert len(post.requests) == 4


def test_client_needs_a_key_except_for_local_servers():
    with pytest.raises(LLMError, match="MY_KEY is not set"):
        client(FakePost(), environ={}).generate("s", "p")
    post = FakePost(ok("local"))
    local = OpenAICompatibleLLM("http://localhost:11434/v1", "m", "K", post=post, environ={})
    assert local.generate("s", "p") == "local" and "Authorization" not in post.requests[0][1]


def test_client_rejects_malformed_responses():
    for bad in ((200, "not json"), (200, json.dumps({"choices": []})), (200, json.dumps({"choices": [{"message": {"content": None}}]}))):
        with pytest.raises(LLMError):
            client(FakePost(bad)).generate("s", "p")


def test_client_spaces_calls_when_a_minimum_interval_is_set():
    sleeps, now = [], [100.0]
    post = FakePost(ok("a"), ok("b"))
    llm = OpenAICompatibleLLM("https://x/v1", "m", "K", post=post, sleep=sleeps.append, clock=lambda: now[0], min_interval=5.0, environ={"K": "k"})
    llm.generate("s", "p")
    now[0] = 101.0                                                         # 1 s later: must wait 4 more
    llm.generate("s", "p")
    assert sleeps == [pytest.approx(4.0)]


# ----------------------------------------------------------------------- report

def build_results(with_rag=False):
    retriever = ScriptedRetriever({
        "question q1": [hit("book.pdf", 10, score=0.8)],
        "question q2": [hit("other.pdf", 3, score=0.5), hit("book.pdf", 10, score=0.4)],
        "question q3": [hit("book.pdf", 40, score=0.3)],
        "off topic o1": [hit("book.pdf", 5, score=0.45)],
    })
    items = [item("q1", pages=(10,)), item("q2", pages=(10,), language="ar"), item("q3", pages=(10,)), oos("o1"), item("d9", status="draft", expected_pages=[])]
    factory = (lambda recorder: ScriptedAnswerer(recorder, [hit("book.pdf", 10, score=0.8)], GOOD)) if with_rag else None
    results = run_evaluation(items, retriever, room_id="r", ks=KS, min_score=0.35, answerer_factory=factory)
    results["meta"] = {"benchmark_name": "demo"}
    return results


def test_report_contains_every_section_and_the_failing_questions():
    text = render_report(build_results(), {"benchmark_name": "demo", "room": "r"})
    for section in ("## Run", "## Benchmark status", "## Retrieval", "### Questions that did not hit at rank 1",
                    "## Abstention gate and score separation", "## RAG answer layer", "## Recommended fixes", "## Limitations"):
        assert section in text, section
    assert "**Not run.**" in text and "d9" in text                              # RAG not run; draft listed as skipped
    assert "| q2 |" in text and "| q3 |" in text and "| q1 |" not in text.split("### Questions that did not hit at rank 1")[1].split("##")[0]
    assert "(other doc)" in text


def test_recommendations_are_tied_to_the_observed_failures():
    text = "\n".join(recommendations(build_results()))
    assert "draft item(s) were not scored" in text
    assert "Only 3 answerable questions" in text
    assert "document-aware routing" not in text                                    # no question missed the document entirely
    assert "had their first relevant chunk below rank 1" in text                   # q2 is a late hit
    assert "out-of-scope questions still pass the min_score=0.35 gate" in text     # o1 scored 0.45
    assert "RAG answer layer was not run" in text
    # q3 is a plain miss (page 40 is nowhere near page 10)
    assert "found nothing near the expected pages" in text


def test_threshold_hint_gives_the_whole_separating_range():
    results = build_results()
    # scripted top-1 scores: answerable 0.8 / 0.5 / 0.3, out-of-scope 0.45 -> only cut-offs above 0.45 block it, and those pass 0.8 and 0.5 (2 of 3)
    results["threshold_sweep"] = [
        {"threshold": 0.40, "answerable_pass_rate": 2 / 3, "out_of_scope_pass_rate": 1.0},
        {"threshold": 0.50, "answerable_pass_rate": 2 / 3, "out_of_scope_pass_rate": 0.0},
        {"threshold": 0.55, "answerable_pass_rate": 1 / 3, "out_of_scope_pass_rate": 0.0},
    ]
    hint = [r for r in recommendations(results) if r.startswith("Descriptively")][0]
    assert "from 0.50 to 0.50" in hint and "66.7%" in hint
    results["threshold_sweep"].append({"threshold": 0.45, "answerable_pass_rate": 2 / 3, "out_of_scope_pass_rate": 0.0})
    hint = [r for r in recommendations(results) if r.startswith("Descriptively")][0]
    assert "from 0.45 to 0.50" in hint


def test_csv_has_one_row_per_scored_question_and_marks_other_documents():
    import csv as csv_module
    import io

    rows = list(csv_module.DictReader(io.StringIO(csv_text(build_results(with_rag=True)))))
    assert [r["id"] for r in rows] == ["q1", "q2", "q3", "o1"]
    q2 = next(r for r in rows if r["id"] == "q2")
    assert ":OTHER_DOC" in q2["retrieved"] and q2["first_relevant_rank"] == "2" and q2["hit@1"] == "0.0"
    assert next(r for r in rows if r["id"] == "q1")["rag_outcome"] == "answered"
    assert "hit@5" in rows[0] and "recall@3" in rows[0]


def test_report_with_rag_results_lists_unsupported_answers():
    results = build_results(with_rag=True)
    text = render_report(results, {"benchmark_name": "demo"})
    assert "Out-of-scope answered anyway" in text and "o1" in text                  # the scripted answerer answers everything
    assert "LLM judge not run" in text
    assert any("out-of-scope question(s) (o1)" in r for r in recommendations(results))

# ------------------------------------------------- hybrid retrieval: audit fields and the gate score

def test_hybrid_audit_fields_are_kept_when_present_and_absent_otherwise():
    plain = score_retrieval(item(), [hit("book.pdf", 10, 0.9)], KS, 0.35)["retrieved"][0]
    assert not {"fusion_score", "dense_rank", "lexical_rank", "lexical_score"} & set(plain)
    fused = {**hit("book.pdf", 10, 0.9), "fusion_score": 0.0328, "dense_rank": 1, "lexical_rank": 2, "lexical_score": 3.2}
    kept = score_retrieval(item(), [fused], KS, 0.35)["retrieved"][0]
    assert (kept["fusion_score"], kept["dense_rank"], kept["lexical_rank"], kept["lexical_score"]) == (0.0328, 1, 2, 3.2)
    assert kept["score"] == 0.9                                              # the cosine is not replaced
    out_of_scope = score_retrieval(oos(), [fused], KS, 0.35)["retrieved"][0]
    assert out_of_scope["fusion_score"] == 0.0328


def test_max_score_is_the_best_cosine_retrieved_and_drives_the_gate_sweep():
    # fusion put a 0.30-cosine chunk first, but a 0.60 chunk was retrieved too: the Answerer's gate would let it through
    hits = [hit("book.pdf", 1, 0.30), hit("book.pdf", 2, 0.60)]
    record = score_retrieval(oos(), hits, KS, 0.35)
    assert record["top1_score"] == 0.30 and record["max_score"] == 0.60 and record["gate_passed"] is True
    sweep = {row["threshold"]: row for row in threshold_sweep([record])}
    assert sweep[0.55]["out_of_scope_pass_rate"] == 1.0 and sweep[0.65]["out_of_scope_pass_rate"] == 0.0
    dense = score_retrieval(oos(), [hit("book.pdf", 1, 0.6), hit("book.pdf", 2, 0.3)], KS, 0.35)
    assert dense["max_score"] == dense["top1_score"] == 0.6                    # dense order: identical, nothing changes
    old_run_record = {"answerable": False, "top1_score": 0.5}                  # written before max_score existed
    assert threshold_sweep([old_run_record])[10]["out_of_scope_pass_rate"] == 1.0   # threshold 0.50

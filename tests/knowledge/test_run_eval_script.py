"""scripts/run_eval.py end to end on a real saved store with the deterministic HashingEmbedder.

Flow under test: draft benchmark -> validate_benchmark --write (ground truth derived from the stored
chunks) -> run_eval -> results.json / per_question.csv / report.md. No BGE-M3 and no network.
Questions are the stored page texts verbatim so the expected ranking is certain by construction; this
checks the plumbing and the arithmetic, it says nothing about the quality of the real system.
"""

import io
import json
from pathlib import Path

import pytest

from eval_helpers import BOOK, OTHER, load_script, save_store
from kb_fakes import HashingEmbedder
from rafeeq.knowledge.evaluation.dataset import BenchmarkItem, dump_benchmark

run_eval = load_script("run_eval")
validate_benchmark = load_script("validate_benchmark")

ROOM = "llm_course"


def snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(Path(directory).iterdir())}


def make_item(**kw):
    base = dict(language="en", qtype="definition", answerable=True, file_name="hands_on_llm.pdf")
    base.update(kw)
    return BenchmarkItem(**base)


def draft_benchmark():
    return [
        make_item(id="e1", question=BOOK[3], evidence_quotes=["Retrieval-augmented generation fetches documents"],
                  reference_answer="RAG fetches documents before answering.", key_facts=[["documents"], ["before answering"]]),
        make_item(id="e2", question=BOOK[0], evidence_quotes=["Self-attention lets every token attend to the other tokens"],
                  reference_answer="Every token attends to the others.", key_facts=[["attend"]], language="ar"),
        make_item(id="e3", question=BOOK[3] + " " + BOOK[4], qtype="multi_passage",
                  evidence_quotes=["Retrieval-augmented generation fetches documents", "LoRA adapts large models by training small"],
                  reference_answer="RAG fetches documents; LoRA trains small matrices.", key_facts=[["documents"], ["low-rank"]]),
        make_item(id="o1", question="zebra giraffe savanna migration", qtype="out_of_scope", answerable=False, file_name=None, absent_terms=["zebra"]),
        make_item(id="d1", question="A question nobody verified yet?"),
    ]


class Env:
    def __init__(self, tmp_path):
        self.store_dir = tmp_path / "store"
        save_store(self.store_dir, {ROOM: {"hands_on_llm.pdf": BOOK, "other.pdf": OTHER}})
        self.benchmark = tmp_path / "bench.jsonl"
        dump_benchmark(draft_benchmark(), self.benchmark)
        self.out_dir = tmp_path / "results"

    def validate(self):
        code = validate_benchmark.main(["--benchmark", str(self.benchmark), "--store-dir", str(self.store_dir), "--write"], out=io.StringIO(), err=io.StringIO())
        assert code == 0

    def run(self, *extra, llm=None, judge_llm=None, embedder=None):
        out, err = io.StringIO(), io.StringIO()
        argv = ["--benchmark", str(self.benchmark), "--store-dir", str(self.store_dir), "--room-id", ROOM, "--out-dir", str(self.out_dir), *extra]
        code = run_eval.main(argv, embedder=embedder or HashingEmbedder(), llm=llm, judge_llm=judge_llm, out=out, err=err)
        return code, out.getvalue(), err.getvalue()

    def results(self):
        return json.loads((self.out_dir / "results.json").read_text(encoding="utf-8"))


class QuotingLLM:
    """Answers with the first excerpt it was given and cites it, like a perfectly extractive model."""

    def __init__(self):
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        first = prompt.split("[1] ", 1)[1].split("\n", 1)[1].split("\n\n", 1)[0]
        return f"{first} [1]"

    def describe(self):
        return {"model": "quoting-fake", "temperature": 0.0}


class VerdictLLM:
    def generate(self, system, prompt):
        return '{"correctness": "correct", "faithfulness": "faithful", "reason": "matches"}'

    def describe(self):
        return {"model": "judge-fake"}


# ---------------------------------------------------------------------- refusals

def test_a_benchmark_of_only_drafts_cannot_be_scored(tmp_path):
    env = Env(tmp_path)                                               # the file written by Env has NOT been through validate_benchmark
    out, err = io.StringIO(), io.StringIO()
    code = run_eval.main(["--benchmark", str(env.benchmark), "--store-dir", str(env.store_dir), "--out-dir", str(env.out_dir)],
                         embedder=HashingEmbedder(), out=out, err=err)
    assert code == 1 and "no verified items" in err.getvalue() and "validate_benchmark.py --write" in err.getvalue()
    assert not env.out_dir.exists()                                   # no numbers, no files


def test_draft_items_never_enter_the_scores(tmp_path):
    env = Env(tmp_path)
    env.validate()
    assert env.run()[0] == 0
    assert env.results()["items"]["skipped_draft"] == ["d1"]
    assert {r["id"] for r in env.results()["per_question"]} == {"e1", "e2", "e3", "o1"}


@pytest.mark.parametrize(
    "extra,message",
    [(["--judge"], "needs an LLM"), (["--llm-base-url", "https://x/v1"], "needs --llm-model"),
     (["--room-id", "nowhere"], "has no chunks"), (["--min-score", "2"], "between -1 and 1")],
)
def test_bad_configuration_fails_clearly(tmp_path, extra, message):
    env = Env(tmp_path)
    env.validate()
    code, _, err = env.run(*extra)
    assert code == 1 and message in err and not env.out_dir.exists()


def test_benchmark_for_a_document_the_room_does_not_hold_is_refused(tmp_path):
    env = Env(tmp_path)
    env.validate()
    other = tmp_path / "other_room_store"
    save_store(other, {ROOM: {"different_book.pdf": BOOK}})
    out, err = io.StringIO(), io.StringIO()
    code = run_eval.main(["--benchmark", str(env.benchmark), "--store-dir", str(other), "--room-id", ROOM, "--out-dir", str(env.out_dir)],
                         embedder=HashingEmbedder(), out=out, err=err)
    assert code == 1 and "expects hands_on_llm.pdf" in err.getvalue()


def test_missing_store_and_missing_benchmark(tmp_path):
    env = Env(tmp_path)
    env.validate()
    out, err = io.StringIO(), io.StringIO()
    assert run_eval.main(["--benchmark", str(env.benchmark), "--store-dir", str(tmp_path / "none")], embedder=HashingEmbedder(), out=out, err=err) == 1
    assert "no vector store found" in err.getvalue()
    err = io.StringIO()
    assert run_eval.main(["--benchmark", str(tmp_path / "none.jsonl"), "--store-dir", str(env.store_dir)], embedder=HashingEmbedder(), out=out, err=err) == 1
    assert "cannot read benchmark" in err.getvalue()


# ------------------------------------------------------------ retrieval-only run

def test_retrieval_only_run_writes_all_artifacts_and_leaves_the_store_untouched(tmp_path):
    env = Env(tmp_path)
    env.validate()
    before = snapshot(env.store_dir)

    code, out, err = env.run()

    assert code == 0 and err == ""
    assert snapshot(env.store_dir) == before                          # read only
    assert sorted(p.name for p in env.out_dir.iterdir()) == ["per_question.csv", "report.md", "results.json"]
    results = env.results()
    overall = results["retrieval"]["overall"]
    assert overall["n"] == 3 and results["items"]["scored_out_of_scope"] == 1
    by_id = {r["id"]: r for r in results["per_question"]}
    assert by_id["e1"]["first_relevant_rank"] == 1 and by_id["e2"]["first_relevant_rank"] == 1       # the question IS the page text
    assert by_id["e1"]["hit@1"] == 1.0 and by_id["e1"]["expected_pages"] == [4]
    assert by_id["e3"]["expected_pages"] == [4, 5] and by_id["e3"]["recall@5"] == 1.0
    # aggregates must equal the per-question numbers they summarise
    answerable = [r for r in results["per_question"] if r["answerable"]]
    assert overall["hit@1"] == pytest.approx(sum(r["hit@1"] for r in answerable) / 3)
    assert overall["mrr@5"] == pytest.approx(sum(r["rr"] for r in answerable) / 3)
    assert results["rag"]["status"] == "not_run"
    assert "RAG answer layer: not_run" in out and "Hit@1=" in out
    assert by_id["o1"]["gate_passed"] is False                        # nothing in the store resembles it
    meta = results["meta"]
    assert meta["room"] == ROOM and "sha256" in meta["benchmark_file"] and "(7 vectors)" in meta["store"]
    report = (env.out_dir / "report.md").read_text(encoding="utf-8")
    assert "**Not run.**" in report and "d1" in report


def test_ks_are_configurable(tmp_path):
    env = Env(tmp_path)
    env.validate()
    assert env.run("--ks", "2", "4")[0] == 0
    overall = env.results()["retrieval"]["overall"]
    assert "hit@2" in overall and "hit@4" in overall and "hit@3" not in overall and "mrr@4" in overall


# --------------------------------------------------------------------- RAG run

def test_rag_run_scores_answers_citations_and_abstention(tmp_path):
    env = Env(tmp_path)
    env.validate()
    llm = QuotingLLM()

    code, out, err = env.run(llm=llm)

    assert code == 0 and err == ""
    rag = env.results()["rag"]
    assert rag["status"] == "run" and rag["n_errors"] == 0
    assert rag["answerable"]["answered_rate"] == 1.0
    assert rag["out_of_scope"]["correct_abstention_rate"] == 1.0                  # declined at the retrieval gate
    assert rag["abstention_stages"] == {"retrieval_gate": 1}
    assert llm.calls == 3                                                          # never called for the out-of-scope question
    assert rag["citations"]["hit_rate"] == 1.0 and rag["citations"]["wrong_document_citations"] == 0
    by_id = {r["id"]: r["rag"] for r in env.results()["per_question"]}
    assert by_id["e1"]["citations"] == [{"file_name": "hands_on_llm.pdf", "page": 4}]
    assert by_id["e1"]["key_fact_coverage"] == 1.0                                 # answer repeats the excerpt
    assert by_id["e3"]["citation"]["recall"] == 0.5                                # only one of the two pages was cited
    assert rag["judge"]["status"] == "not_run"
    assert env.results()["meta"]["answer LLM"].count("quoting-fake") == 1


def test_judge_results_are_added_only_when_requested(tmp_path):
    env = Env(tmp_path)
    env.validate()
    code, _, _ = env.run("--judge", llm=QuotingLLM(), judge_llm=VerdictLLM())
    assert code == 0
    judge = env.results()["rag"]["judge"]
    assert judge["n_judged"] == 3 and judge["correctness"] == {"correct": 3} and judge["faithfulness"] == {"faithful": 3}
    assert "judge-fake" in env.results()["meta"]["judge"]
    assert "LLM judge" in (env.out_dir / "report.md").read_text(encoding="utf-8")


def test_min_score_zero_exposes_unsupported_answers(tmp_path):
    env = Env(tmp_path)
    env.validate()
    code, _, _ = env.run("--min-score", "0", llm=QuotingLLM())
    assert code == 0
    rag = env.results()["rag"]
    assert rag["out_of_scope"]["unsupported_answer_ids"] == ["o1"]                  # the always-answering fake LLM, gate removed
    assert rag["out_of_scope"]["unsupported_answer_rate"] == 1.0
    assert "o1" in (env.out_dir / "report.md").read_text(encoding="utf-8")


def test_an_llm_failure_on_one_question_is_reported_not_fatal(tmp_path):
    env = Env(tmp_path)
    env.validate()

    class Flaky(QuotingLLM):
        def generate(self, system, prompt):
            if "LoRA" in prompt:
                raise RuntimeError("HTTP 429")
            return super().generate(system, prompt)

    code, _, _ = env.run(llm=Flaky())
    assert code == 0
    rag = env.results()["rag"]
    assert rag["n_errors"] == 1 and rag["errors"][0]["id"] == "e3" and "HTTP 429" in rag["errors"][0]["error"]
    assert "e3 (RuntimeError: HTTP 429" in (env.out_dir / "report.md").read_text(encoding="utf-8")

# ------------------------------------------- answer-layer evaluation: groups and presets

def test_rag_results_are_split_by_question_group(tmp_path):
    env = Env(tmp_path)
    env.validate()
    code, _, _ = env.run(llm=QuotingLLM())
    assert code == 0
    groups = env.results()["rag"]["by_group"]
    assert set(groups) == {"single_passage", "multi_passage", "out_of_scope"}
    assert groups["single_passage"]["n"] == 2 and groups["single_passage"]["full_citation_recall"] == 1.0
    assert groups["multi_passage"]["n"] == 1 and groups["multi_passage"]["answered_rate"] == 1.0
    assert groups["multi_passage"]["full_citation_recall"] == 0.0                   # e3 cited only one of its two pages
    assert groups["multi_passage"]["citation_mean_recall"] == 0.5
    assert groups["out_of_scope"] == {"n": 1, "correct_abstention_rate": 1.0, "unsupported_answer_ids": []}
    assert "By question group" in (env.out_dir / "report.md").read_text(encoding="utf-8")


def test_provisional_rag_config_selects_hybrid_with_exclusion(tmp_path):
    env = Env(tmp_path)
    env.validate()
    code, _, err = env.run("--provisional-rag-config", llm=QuotingLLM())
    assert code == 0, err
    retrieval = env.results()["meta"]["retrieval"]
    assert retrieval.startswith("hybrid") and "candidate pool 50" in retrieval and "index-like pages excluded" in retrieval


def test_provisional_rag_config_enforces_candidate_pool_50_even_if_another_is_given(tmp_path):
    args = run_eval.build_parser().parse_args(["--provisional-rag-config", "--candidate-pool", "7"])
    run_eval._apply_presets(args)
    assert (args.retrieval_mode, args.exclude_non_content, args.candidate_pool) == ("hybrid", True, 50)
    env = Env(tmp_path)
    env.validate()
    code, _, err = env.run("--provisional-rag-config", "--candidate-pool", "7", llm=QuotingLLM())
    assert code == 0, err
    retrieval = env.results()["meta"]["retrieval"]
    assert "candidate pool 50" in retrieval and "candidate pool 7" not in retrieval


def test_gemini_provider_uses_the_openai_compatible_endpoint_and_its_own_key_variable():
    args = run_eval.build_parser().parse_args(["--llm-provider", "gemini", "--llm-model", "m"])
    run_eval._apply_presets(args)
    assert args.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert args.llm_api_key_env == "GEMINI_API_KEY"
    other = run_eval.build_parser().parse_args(["--llm-base-url", "https://x/v1", "--llm-model", "m"])
    run_eval._apply_presets(other)
    assert other.llm_base_url == "https://x/v1" and other.llm_api_key_env == "LLM_API_KEY"
    own = run_eval.build_parser().parse_args(["--llm-provider", "gemini", "--llm-api-key-env", "MY_KEY"])
    run_eval._apply_presets(own)
    assert own.llm_api_key_env == "MY_KEY"

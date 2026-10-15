"""scripts/replay_hybrid.py must give exactly what run_eval.py --retrieval-mode hybrid gives for the same pool size."""

import io
import json

import pytest

from eval_helpers import load_script, save_store
from kb_fakes import HashingEmbedder
from rafeeq.knowledge.evaluation.dataset import BenchmarkItem, dump_benchmark
from test_content_filter import index_page

run_eval = load_script("run_eval")
replay_hybrid = load_script("replay_hybrid")
validate_benchmark = load_script("validate_benchmark")
compare_runs = load_script("compare_runs")

ROOM = "llm_course"
PAGES = [
    "Convolutional neural networks process images with learned filters and pooling layers.",
    "Self-attention lets every token attend to the other tokens in the sequence.",
    "Gradient descent optimizes the weights of neural networks step by step using learning rates.",
    "Retrieval-augmented generation fetches documents before answering a question with a language model.",
    "LoRA adapts large models by training small low-rank matrices only.",
    "Tokenizers split raw text into tokens before embedding.",
    "Quantization stores neural network weights with fewer bits to save memory.",
    index_page([("quantization", "Quantization Basics, Quantization Memory"), ("tokenizer", "Tokenizer Vocabulary, Token Splitting"),
                ("gradient", "Gradient Descent, Gradient Clipping")] * 6),
    "Transformers use attention heads in parallel to model different relations between tokens.",
    "Embedding models map sentences to vectors so that similar meaning gives nearby vectors.",
    "Neural networks neural networks neural networks are trained by gradient methods.",
    "Evaluation of language models uses perplexity and human judgement.",
]
QUESTIONS = [("q1", "What is self-attention?", 2), ("q2", "How does quantization save memory?", 7), ("q3", "How are neural networks trained with gradient descent?", 3),
             ("q4", "What does retrieval-augmented generation do before answering?", 4), ("q5", "How do tokenizers split text into tokens?", 6),
             ("q6", "Which models use attention heads?", 9)]


@pytest.fixture()
def env(tmp_path):
    store_dir = tmp_path / "store"
    save_store(store_dir, {ROOM: {"hands_on_llm.pdf": PAGES}})
    items = [BenchmarkItem(id=qid, language="en", qtype="definition", question=q, answerable=True, file_name="hands_on_llm.pdf",
                           evidence_quotes=[PAGES[page - 1]], reference_answer="reference", key_facts=[["x"]]) for qid, q, page in QUESTIONS]
    items.append(BenchmarkItem(id="o1", language="en", qtype="out_of_scope", question="Who won the football world cup?", answerable=False, absent_terms=["zebra"]))
    benchmark = tmp_path / "bench.jsonl"
    dump_benchmark(items, benchmark)
    assert validate_benchmark.main(["--benchmark", str(benchmark), "--store-dir", str(store_dir), "--write"], out=io.StringIO(), err=io.StringIO()) == 0
    return tmp_path, store_dir, benchmark


def run_real(env, name, *extra):
    tmp, store_dir, benchmark = env
    out_dir = tmp / name
    code = run_eval.main(["--benchmark", str(benchmark), "--store-dir", str(store_dir), "--room-id", ROOM, "--out-dir", str(out_dir), *extra],
                         embedder=HashingEmbedder(), out=io.StringIO(), err=io.StringIO())
    assert code == 0
    return json.loads((out_dir / "results.json").read_text(encoding="utf-8")), out_dir


def run_replay(env, source_dir, name, *extra):
    tmp, store_dir, benchmark = env
    out, err = io.StringIO(), io.StringIO()
    code = replay_hybrid.main([str(source_dir / "results.json"), "--benchmark", str(benchmark), "--store-dir", str(store_dir), "--room-id", ROOM,
                               "--out-dir", str(tmp / name), *extra], out=out, err=err)
    return code, out.getvalue(), err.getvalue(), tmp / name


def rows(results):
    return {r["id"]: r for r in results["per_question"] if r["answerable"]}


@pytest.mark.parametrize("flags", [[], ["--exclude-non-content"]])
def test_the_replay_equals_the_real_hybrid_run_for_the_same_pool(env, flags):
    _, dense_dir = run_real(env, "dense", *flags)
    real, _ = run_real(env, "real", "--retrieval-mode", "hybrid", "--candidate-pool", "5", *flags)
    code, _, err, replay_dir = run_replay(env, dense_dir, "replay", *flags)
    assert code == 0, err
    replayed = json.loads((replay_dir / "results.json").read_text(encoding="utf-8"))
    for qid, real_row in rows(real).items():
        got = rows(replayed)[qid]
        assert [h["chunk_id"] for h in got["retrieved"]] == [h["chunk_id"] for h in real_row["retrieved"]], qid
        for field in ("fusion_score", "dense_rank", "lexical_rank"):
            for g, r in zip(got["retrieved"], real_row["retrieved"]):
                assert (g[field] is None) == (r[field] is None), (qid, field)
                if r[field] is not None:
                    assert g[field] == pytest.approx(r[field]), (qid, field)
        assert got["first_relevant_rank"] == real_row["first_relevant_rank"]
        assert {k: got[k] for k in got if k.startswith(("hit@", "recall@", "full_recall@"))} == {k: real_row[k] for k in real_row if k.startswith(("hit@", "recall@", "full_recall@"))}
    assert replayed["retrieval"]["overall"] == real["retrieval"]["overall"]


def test_the_fixture_is_meaningful_hybrid_reorders_at_least_one_question(env):
    dense, dense_dir = run_real(env, "dense")
    real, _ = run_real(env, "real", "--retrieval-mode", "hybrid", "--candidate-pool", "5")
    changed = [qid for qid, r in rows(dense).items() if [h["page"] for h in r["retrieved"]] != [h["page"] for h in rows(real)[qid]["retrieved"]]]
    assert changed, "the replay test would pass trivially if hybrid never changed anything"


def test_the_replay_does_not_claim_the_abstention_gate_and_works_with_compare_runs(env):
    _, dense_dir = run_real(env, "dense")
    code, out, err, replay_dir = run_replay(env, dense_dir, "replay")
    assert code == 0 and "hybrid REPLAY" in out
    replayed = json.loads((replay_dir / "results.json").read_text(encoding="utf-8"))
    assert replayed["threshold_sweep"] == [] and "not computed" in replayed["retrieval_gate"]["note"]
    o1 = next(r for r in replayed["per_question"] if r["id"] == "o1")
    assert o1["max_score"] is None and o1["gate_passed"] is None
    text = compare_runs.render(compare_runs.compare(json.loads((dense_dir / "results.json").read_text(encoding="utf-8")), replayed))
    assert "Retrieval comparison" in text and "hybrid REPLAY" in text


def test_replay_refuses_what_it_cannot_do_and_never_overwrites(env):
    tmp, store_dir, benchmark = env
    _, dense_dir = run_real(env, "dense")
    assert run_replay(env, dense_dir, "ok")[0] == 0
    code, _, err, _ = run_replay(env, dense_dir, "ok")
    assert code == 1 and "never overwritten" in err
    code, _, err, out_dir = run_replay(env, dense_dir, "deep", "--pool", "6")
    assert code == 1 and "between 1 and the stored depth 5" in err and not out_dir.exists()
    other = tmp / "other.jsonl"
    other.write_bytes(benchmark.read_bytes() + b"\n")
    code = replay_hybrid.main([str(dense_dir / "results.json"), "--benchmark", str(other), "--store-dir", str(store_dir), "--room-id", ROOM, "--out-dir", str(tmp / "x")],
                              out=io.StringIO(), err=io.StringIO())
    assert code == 1 and not (tmp / "x").exists()               # a different benchmark file is not silently accepted


def test_a_run_made_on_windows_with_crlf_endings_still_matches_the_same_benchmark(env):
    tmp, store_dir, benchmark = env
    _, dense_dir = run_real(env, "dense")
    results = json.loads((dense_dir / "results.json").read_text(encoding="utf-8"))
    crlf = benchmark.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    import hashlib

    results["meta"]["benchmark_file"] = f"D:\\\\x\\\\bench.jsonl (sha256 {hashlib.sha256(crlf).hexdigest()[:12]})"
    (dense_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    assert run_replay(env, dense_dir, "win")[0] == 0

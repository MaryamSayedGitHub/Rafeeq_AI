"""Cross-Encoder reranking: everything here uses a mocked scorer. No model, no download, no network, no API key."""

import io
import json
import sys

import pytest

from eval_helpers import load_script
from kb_fakes import HashingEmbedder
from rafeeq.knowledge.retrieval.factory import build_retriever
from rafeeq.knowledge.retrieval.hybrid_retriever import HybridRetriever
from rafeeq.knowledge.retrieval.reranker import CrossEncoderReranker, RerankingRetriever
from rafeeq.knowledge.retrieval.retriever import Retriever
from test_run_eval_hybrid import ROOM, row, run, setup

run_eval = load_script("run_eval")


class FakeScorer:
    """Stands in for sentence_transformers.CrossEncoder: scores by a rule, records every call."""

    def __init__(self, rule):
        self.rule, self.calls = rule, []

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.calls.append(([list(p) for p in pairs], batch_size))
        return [self.rule(q, p) for q, p in pairs]


class StubInner:
    def __init__(self, hits):
        self.hits, self.asked = hits, []

    def retrieve(self, room_id, query, k=6):
        self.asked.append((room_id, query, k))
        return [dict(h) for h in self.hits[:k]]


def hit(cid, text, score=0.5, page=1):
    return {"chunk_id": cid, "text": text, "file_name": "b.pdf", "page": page, "topic_id": "t", "score": score}


HITS = [hit(f"c{i}", f"text {i}", score=0.9 - i * 0.01, page=i + 1) for i in range(8)]


def make(rule, pool=5):
    scorer = FakeScorer(rule)
    return RerankingRetriever(StubInner(HITS), CrossEncoderReranker(scorer=scorer), pool=pool), scorer


def test_reorders_by_reranker_and_keeps_all_metadata_and_cosine():
    retriever, _ = make(lambda q, p: {"text 3": 9.0, "text 0": 5.0}.get(p, 0.0))
    out = retriever.retrieve("r", "q", k=3)
    assert [h["chunk_id"] for h in out] == ["c3", "c0", "c1"]
    top = out[0]
    assert (top["file_name"], top["page"], top["topic_id"], top["text"]) == ("b.pdf", 4, "t", "text 3")
    assert top["score"] == pytest.approx(0.87)           # cosine untouched: the Answerer's gate keeps its meaning
    assert top["rerank_score"] == 9.0 and top["pre_rerank_rank"] == 4


def test_only_pool_candidates_are_scored_and_k_is_respected():
    retriever, scorer = make(lambda q, p: 0.0, pool=5)
    assert len(retriever.retrieve("r", "q", k=2)) == 2
    assert retriever._inner.asked == [("r", "q", 5)]
    assert len(scorer.calls[0][0]) == 5


def test_pool_never_smaller_than_k():
    retriever, _ = make(lambda q, p: 0.0, pool=2)
    assert len(retriever.retrieve("r", "q", k=4)) == 4


def test_ties_are_deterministic_inner_rank_then_chunk_id():
    retriever, _ = make(lambda q, p: 1.0)
    assert [h["chunk_id"] for h in retriever.retrieve("r", "q", k=5)] == ["c0", "c1", "c2", "c3", "c4"]
    assert retriever.retrieve("r", "q", k=5) == retriever.retrieve("r", "q", k=5)


def test_empty_candidates_never_load_the_model():
    reranker = CrossEncoderReranker()
    assert RerankingRetriever(StubInner([]), reranker).retrieve("r", "q") == [] and not reranker.is_loaded


def test_scores_the_query_and_chunk_text_pairs():
    retriever, scorer = make(lambda q, p: 0.0, pool=3)
    retriever.retrieve("r", "my question", k=3)
    assert scorer.calls[0][0] == [["my question", "text 0"], ["my question", "text 1"], ["my question", "text 2"]]


def test_wrong_score_count_fails_loudly():
    class Bad:
        def predict(self, pairs, **kw):
            return [1.0]
    with pytest.raises(RuntimeError):
        RerankingRetriever(StubInner(HITS), CrossEncoderReranker(scorer=Bad()), pool=3).retrieve("r", "q")


@pytest.mark.parametrize("k", [0, -1, True, "6"])
def test_invalid_k_rejected(k):
    with pytest.raises(ValueError):
        make(lambda q, p: 0.0)[0].retrieve("r", "q", k=k)


def test_construction_and_import_load_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)   # any import attempt would raise
    reranker = CrossEncoderReranker()
    assert not reranker.is_loaded and reranker.load_count == 0
    with pytest.raises(ImportError):
        reranker.load()


def test_model_loaded_lazily_once_with_cpu_resolution(monkeypatch):
    created = []

    class FakeCE:
        def __init__(self, name, device=None, max_length=None):
            created.append((name, device, max_length))

        def predict(self, pairs, **kw):
            return [0.0] * len(pairs)

    class FakeST:
        CrossEncoder = FakeCE

    monkeypatch.setitem(sys.modules, "sentence_transformers", FakeST)
    monkeypatch.setitem(sys.modules, "torch", type("T", (), {"cuda": type("C", (), {"is_available": staticmethod(lambda: False)})})())
    reranker = CrossEncoderReranker(device="auto", max_length=256)
    assert created == []
    reranker.score("q", ["a"]); reranker.score("q", ["b", "c"])
    assert created == [("BAAI/bge-reranker-v2-m3", "cpu", 256)] and reranker.load_count == 1 and reranker.device == "cpu"


def test_gpu_failure_falls_back_to_cpu(monkeypatch):
    created = []

    class FakeCE:
        def __init__(self, name, device=None, max_length=None):
            created.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")

        def predict(self, pairs, **kw):
            return [0.0] * len(pairs)

    monkeypatch.setitem(sys.modules, "sentence_transformers", type("ST", (), {"CrossEncoder": FakeCE}))
    reranker = CrossEncoderReranker(device="cuda")
    reranker.load()
    assert created == ["cuda", "cpu"] and reranker.device == "cpu"


# ------------------------------------------------------------------ factory: defaults unchanged

def test_factory_default_and_explicit_off_return_the_same_plain_classes():
    assert type(build_retriever(object(), HashingEmbedder())) is Retriever
    assert type(build_retriever(object(), HashingEmbedder(), rerank=False)) is Retriever


def test_factory_rerank_wraps_dense_or_hybrid_without_loading_a_model():
    class Store:
        def room_chunks(self, r): return []
        def similarities(self, *a): return {}
        def __len__(self): return 0
    dense = build_retriever(Store(), HashingEmbedder(), rerank=True, reranker=CrossEncoderReranker(scorer=FakeScorer(lambda q, p: 0)))
    hybrid = build_retriever(Store(), HashingEmbedder(), mode="hybrid", rerank=True, reranker=CrossEncoderReranker(scorer=FakeScorer(lambda q, p: 0)))
    assert isinstance(dense, RerankingRetriever) and isinstance(dense._inner, Retriever)
    assert isinstance(hybrid._inner, HybridRetriever)
    default = build_retriever(Store(), HashingEmbedder(), rerank=True)   # real reranker object, still not loaded
    assert not default._reranker.is_loaded


def test_factory_rejects_rerank_options_without_rerank():
    with pytest.raises(ValueError):
        build_retriever(object(), HashingEmbedder(), rerank_pool=10)
    with pytest.raises(ValueError):
        build_retriever(object(), HashingEmbedder(), reranker=object())


# ------------------------------------------------------------------ run_eval end to end (fake embedder, fake scorer)

def _eval(tmp_path, name, *extra, reranker=None):
    store_dir, benchmark = setup(tmp_path) if not (tmp_path / "store").exists() else (tmp_path / "store", tmp_path / "bench.jsonl")
    out_dir = tmp_path / name
    out, err = io.StringIO(), io.StringIO()
    code = run_eval.main(["--benchmark", str(benchmark), "--store-dir", str(store_dir), "--room-id", ROOM, "--out-dir", str(out_dir), *extra],
                         embedder=HashingEmbedder(), reranker=reranker, out=out, err=err)
    return code, (json.loads((out_dir / "results.json").read_text(encoding="utf-8")) if code == 0 else None), err.getvalue()


def test_run_eval_without_rerank_is_identical_to_before_and_records_no_rerank_fields(tmp_path):
    code, res, _ = _eval(tmp_path, "plain", "--retrieval-mode", "hybrid")
    assert code == 0 and "reranker" not in res["meta"] and "rerank" not in res["meta"]["retrieval"]
    assert all("rerank_score" not in h for r in res["per_question"] for h in r["retrieved"])


def test_run_eval_rerank_flag_uses_scorer_and_records_audit_fields(tmp_path):
    scorer = FakeScorer(lambda q, p: 10.0 if "Gradient" in p else 0.0)
    code, res, err = _eval(tmp_path, "rr", "--retrieval-mode", "hybrid", "--rerank", "--rerank-pool", "5", reranker=CrossEncoderReranker(scorer=scorer))
    assert code == 0, err
    assert "Cross-Encoder rerank (BAAI/bge-reranker-v2-m3, pool 5)" in res["meta"]["retrieval"]
    top = row(res, "q1")["retrieved"][0]
    assert "Gradient" in top["text_preview"] and top["rerank_score"] == 10.0 and "pre_rerank_rank" in top
    assert len(scorer.calls) == 2                                   # one scorer call per question (q1, o1)


def test_run_eval_rerank_does_not_change_the_cosine_gate_values(tmp_path):
    _, base, _ = _eval(tmp_path, "base", "--retrieval-mode", "hybrid")
    _, rr, _ = _eval(tmp_path, "rr2", "--retrieval-mode", "hybrid", "--rerank", reranker=CrossEncoderReranker(scorer=FakeScorer(lambda q, p: -len(p))))
    for a, b in zip(base["per_question"], rr["per_question"]):
        assert a["max_score"] == b["max_score"] and a["gate_passed"] == b["gate_passed"]


def test_run_eval_rerank_options_need_the_flag_and_pool_must_cover_ks(tmp_path):
    assert _eval(tmp_path, "a", "--rerank-pool", "10")[0] == 1
    code, _, err = _eval(tmp_path, "b", "--rerank", "--rerank-pool", "3", reranker=CrossEncoderReranker(scorer=FakeScorer(lambda q, p: 0)))
    assert code == 1 and "at least" in err


def test_run_eval_never_overwrites_existing_results(tmp_path):
    scorer = CrossEncoderReranker(scorer=FakeScorer(lambda q, p: 0))
    assert _eval(tmp_path, "same", "--rerank", reranker=scorer)[0] == 0
    code, _, err = _eval(tmp_path, "same", "--rerank", reranker=scorer)
    assert code == 1 and "never overwritten" in err

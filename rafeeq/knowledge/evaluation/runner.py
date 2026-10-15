"""Run the benchmark against a retriever (and, optionally, an Answerer) and aggregate the results.

Retrieval and answering are evaluated SEPARATELY:

* Retrieval: ``Retriever.retrieve(room_id, question, depth)`` for every scored question; relevance is
  defined only by the verified pages / chunk ids (see ``metrics``). No LLM is involved.
* RAG answer layer: ``Answerer.answer(room_id, question)`` with whatever LLM it was built with. The
  retriever is wrapped so we record exactly which chunks the answerer saw. Nothing is re-implemented.

Only items with ``status == "verified"`` are scored; drafts are listed as skipped so an unfinished
benchmark can never silently produce numbers. A failure of the system under test on one question (for
example an LLM timeout) is recorded on that question and the run continues; such questions are counted
as errors and excluded from the rates, and the report says so.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from rafeeq.knowledge.evaluation.dataset import BenchmarkItem
from rafeeq.knowledge.evaluation.metrics import (
    abstention_outcome,
    answer_body,
    citation_scores,
    first_relevant_rank,
    hit_at_k,
    key_fact_coverage,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
    wilson_interval,
)

DEFAULT_KS = (1, 3, 5)
NEAR_PAGE_DISTANCE = 2
PREVIEW_CHARS = 240
DEFAULT_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(0, 19))  # 0.00 .. 0.90


class RecordingRetriever:
    """Delegates to a retriever and remembers the last result, i.e. what the answerer was shown."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last: list[dict[str, Any]] = []

    def retrieve(self, room_id: str, query: str, k: int = 6) -> list[dict[str, Any]]:
        self.last = self._inner.retrieve(room_id, query, k)
        return self.last


# ------------------------------------------------------------------ retrieval

_HYBRID_FIELDS = ("fusion_score", "dense_rank", "lexical_rank", "lexical_score", "rerank_score", "pre_rerank_rank")


def _hybrid_fields(hit: Mapping[str, Any]) -> dict[str, Any]:
    """Extra per-hit fields a hybrid retriever returns, kept in the record so a run can be audited; absent for dense."""
    return {name: hit[name] for name in _HYBRID_FIELDS if name in hit}


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def score_retrieval(
    item: BenchmarkItem,
    hits: Sequence[Mapping[str, Any]],
    ks: Sequence[int],
    min_score: float,
) -> dict[str, Any]:
    """Per-question retrieval record from the chunks returned for ``item.question`` (rank order)."""
    depth = max(ks)
    hits = list(hits)[:depth]
    record: dict[str, Any] = {
        "id": item.id, "language": item.language, "qtype": item.qtype, "question": item.question,
        "answerable": item.answerable, "expected_file": item.file_name,
        "expected_pages": list(item.expected_pages), "expected_chunk_ids": list(item.expected_chunk_ids),
        "top1_score": float(hits[0]["score"]) if hits else None,
        # best cosine among the retrieved chunks: what the Answerer's gate looks at. Equal to top1_score for dense
        # retrieval (sorted by cosine); can be higher for hybrid, where the top-ranked chunk need not have the best cosine.
        "max_score": max(float(h["score"]) for h in hits) if hits else None,
        "gate_passed": any(float(h["score"]) >= min_score for h in hits),
    }
    if not item.answerable:
        record["retrieved"] = [
            {"rank": r, "chunk_id": h["chunk_id"], "file_name": h["file_name"], "page": h["page"],
             "topic_id": h.get("topic_id"), "score": float(h["score"]), "text_preview": _preview(h["text"]), **_hybrid_fields(h)}
            for r, h in enumerate(hits, start=1)
        ]
        return record

    page_flags = relevance_flags(hits, item.file_name, item.expected_pages, level="page")
    chunk_flags = relevance_flags(hits, item.file_name, expected_chunk_ids=item.expected_chunk_ids, level="chunk") if item.expected_chunk_ids else None
    record["retrieved"] = [
        {"rank": r, "chunk_id": h["chunk_id"], "file_name": h["file_name"], "page": h["page"],
         "topic_id": h.get("topic_id"), "score": float(h["score"]), "text_preview": _preview(h["text"]),
         "relevant_page": page_flags[r - 1], "relevant_chunk": chunk_flags[r - 1] if chunk_flags else None, **_hybrid_fields(h)}
        for r, h in enumerate(hits, start=1)
    ]
    units = [(h["file_name"], h["page"]) for h in hits]
    expected_units = {(item.file_name, p) for p in item.expected_pages}
    rank = first_relevant_rank(page_flags)
    record["first_relevant_rank"] = rank
    record["rr"] = reciprocal_rank(page_flags)
    for k in ks:
        record[f"hit@{k}"] = hit_at_k(page_flags, k)
        record[f"precision@{k}"] = precision_at_k(page_flags, k)
        recall = recall_at_k(units, expected_units, k)
        record[f"recall@{k}"] = recall
        record[f"full_recall@{k}"] = 1.0 if recall == 1.0 else 0.0
        top = hits[:k]
        record[f"other_document_share@{k}"] = (sum(1 for h in top if h["file_name"] != item.file_name) / len(top)) if top else 0.0
    if chunk_flags is not None:
        chunk_units = [h["chunk_id"] for h in hits]
        record["rr_chunk"] = reciprocal_rank(chunk_flags)
        for k in ks:
            record[f"hit_chunk@{k}"] = hit_at_k(chunk_flags, k)
            record[f"precision_chunk@{k}"] = precision_at_k(chunk_flags, k)
            record[f"recall_chunk@{k}"] = recall_at_k(chunk_units, item.expected_chunk_ids, k)
    record["failure_category"] = _failure_category(item, hits, rank, record[f"recall@{depth}"])
    return record


def _failure_category(item: BenchmarkItem, hits: Sequence[Mapping[str, Any]], rank: int | None, recall_at_depth: float) -> str:
    if rank == 1:
        return "ok" if recall_at_depth == 1.0 else "ok_partial_coverage"
    if rank is not None:
        return "late_hit" if recall_at_depth == 1.0 else "late_hit_partial_coverage"
    same_file = [h for h in hits if h["file_name"] == item.file_name]
    if not same_file:
        return "miss_wrong_document"
    if any(abs(h["page"] - p) <= NEAR_PAGE_DISTANCE for h in same_file for p in item.expected_pages):
        return "miss_near_page"
    return "miss"


def _group_stats(records: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    stats: dict[str, Any] = {"n": len(records)}
    if not records:
        return stats
    for k in ks:
        hits = [r[f"hit@{k}"] for r in records]
        stats[f"hit@{k}"] = mean(hits)
        interval = wilson_interval(sum(hits), len(hits))
        stats[f"hit@{k}_ci95"] = [round(interval[0], 4), round(interval[1], 4)] if interval else None
        stats[f"precision@{k}"] = mean([r[f"precision@{k}"] for r in records])
        stats[f"recall@{k}"] = mean([r[f"recall@{k}"] for r in records])
        stats[f"full_recall@{k}"] = mean([r[f"full_recall@{k}"] for r in records])
        stats[f"other_document_share@{k}"] = mean([r[f"other_document_share@{k}"] for r in records])
    stats[f"mrr@{max(ks)}"] = mean([r["rr"] for r in records])
    return stats


def aggregate_retrieval(records: Sequence[Mapping[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    """Overall / by question type / by language, over the answerable scored questions."""
    answerable = [r for r in records if r["answerable"]]
    out: dict[str, Any] = {
        "relevance": "page level: chunk.file_name == expected file and chunk.page in expected_pages",
        "ks": list(ks),
        "overall": _group_stats(answerable, ks),
        "by_qtype": {t: _group_stats([r for r in answerable if r["qtype"] == t], ks) for t in sorted({r["qtype"] for r in answerable})},
        "by_language": {l: _group_stats([r for r in answerable if r["language"] == l], ks) for l in sorted({r["language"] for r in answerable})},
    }
    with_chunks = [r for r in answerable if "rr_chunk" in r]
    if with_chunks:
        chunk: dict[str, Any] = {"n": len(with_chunks), "relevance": "chunk level: chunk_id in expected_chunk_ids"}
        for k in ks:
            chunk[f"hit_chunk@{k}"] = mean([r[f"hit_chunk@{k}"] for r in with_chunks])
            chunk[f"precision_chunk@{k}"] = mean([r[f"precision_chunk@{k}"] for r in with_chunks])
            chunk[f"recall_chunk@{k}"] = mean([r[f"recall_chunk@{k}"] for r in with_chunks])
        chunk[f"mrr_chunk@{max(ks)}"] = mean([r["rr_chunk"] for r in with_chunks])
        out["chunk_level"] = chunk
    categories: dict[str, int] = {}
    for r in answerable:
        categories[r["failure_category"]] = categories.get(r["failure_category"], 0) + 1
    out["failure_categories"] = dict(sorted(categories.items()))
    return out


def threshold_sweep(records: Sequence[Mapping[str, Any]], thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> list[dict[str, Any]]:
    """For each cosine threshold t: share of answerable / out-of-scope questions whose best retrieved cosine is >= t.

    Descriptive only. Scores are not probabilities; this shows whether a single cut-off could separate
    the two groups on THIS benchmark, which with ~30 questions is a rough guide, not a calibration.
    """
    def gate_score(r: Mapping[str, Any]) -> Any:
        return r.get("max_score", r["top1_score"])           # runs written before max_score existed are dense: top-1 is the max

    answerable = [gate_score(r) for r in records if r["answerable"] and gate_score(r) is not None]
    oos = [gate_score(r) for r in records if not r["answerable"] and gate_score(r) is not None]
    rows = []
    for t in thresholds:
        rows.append({
            "threshold": t,
            "answerable_pass_rate": (sum(1 for s in answerable if s >= t) / len(answerable)) if answerable else None,
            "out_of_scope_pass_rate": (sum(1 for s in oos if s >= t) / len(oos)) if oos else None,
        })
    return rows


# ------------------------------------------------------------------------ RAG

def evaluate_rag_item(
    item: BenchmarkItem,
    answerer: Any,
    recorder: RecordingRetriever,
    room_id: str,
    min_score: float,
    judge: Any | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"id": item.id, "language": item.language, "qtype": item.qtype, "question": item.question, "answerable": item.answerable}
    recorder.last = []
    try:
        result = answerer.answer(room_id, item.question)
    except Exception as exc:  # recorded on the question; the run continues and the report counts it
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    excerpts = [c for c in recorder.last if c["score"] >= min_score]
    grounded = bool(result["grounded"])
    record.update({
        "grounded": grounded,
        "outcome": abstention_outcome(item.answerable, grounded),
        "answer_md": result["answer_md"],
        "citations": result["citations"],
        "excerpts_given": [{"chunk_id": c["chunk_id"], "file_name": c["file_name"], "page": c["page"], "score": c["score"]} for c in excerpts],
    })
    if not grounded:
        record["abstention_stage"] = "retrieval_gate" if not excerpts else "llm_or_citation_check"
        if item.answerable:
            record["key_fact_coverage"] = 0.0  # end-to-end: declining covers nothing
        return record
    if not item.answerable:
        return record

    record["citation"] = citation_scores(result["citations"], item.file_name, item.expected_pages)
    facts = key_fact_coverage(answer_body(result["answer_md"]), item.key_facts)
    record["key_fact_coverage"] = facts["coverage"]
    record["missing_facts"] = facts["missing"]
    if judge is not None:
        try:
            record["judge"] = judge.judge(item.question, item.reference_answer, excerpts, answer_body(result["answer_md"]))
        except Exception as exc:
            record["judge_error"] = f"{type(exc).__name__}: {exc}"
    return record


def aggregate_rag(records: Sequence[Mapping[str, Any]], judge_used: bool) -> dict[str, Any]:
    errors = [r for r in records if "error" in r]
    ok = [r for r in records if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    oos = [r for r in ok if not r["answerable"]]
    out: dict[str, Any] = {"status": "run", "n_scored": len(ok), "n_errors": len(errors),
                           "errors": [{"id": r["id"], "error": r["error"]} for r in errors]}

    answered = [r for r in answerable if r["grounded"]]
    out["answerable"] = {"n": len(answerable), "answered_rate": (len(answered) / len(answerable)) if answerable else None,
                         "false_abstention_rate": ((len(answerable) - len(answered)) / len(answerable)) if answerable else None}
    abstained_oos = [r for r in oos if not r["grounded"]]
    out["out_of_scope"] = {"n": len(oos), "correct_abstention_rate": (len(abstained_oos) / len(oos)) if oos else None,
                           "unsupported_answer_rate": ((len(oos) - len(abstained_oos)) / len(oos)) if oos else None,
                           "unsupported_answer_ids": [r["id"] for r in oos if r["grounded"]]}
    stages: dict[str, int] = {}
    for r in ok:
        if not r["grounded"]:
            stages[r["abstention_stage"]] = stages.get(r["abstention_stage"], 0) + 1
    out["abstention_stages"] = dict(sorted(stages.items()))
    if ok:
        correct = sum(1 for r in ok if r["outcome"] in ("answered", "correct_abstention"))
        out["abstention_accuracy"] = correct / len(ok)  # "answered" counts as right here: only the decision to answer is judged

    cited = [r["citation"] for r in answered]
    out["citations"] = {
        "n_answers": len(cited),
        "hit_rate": mean([c["hit"] for c in cited]),
        "mean_precision": mean([c["precision"] for c in cited if c["precision"] is not None]),
        "mean_recall": mean([c["recall"] for c in cited]),
        "wrong_document_citations": sum(c["wrong_document"] for c in cited),
    }
    out["key_fact_coverage"] = {
        "proxy": "share of required facts present in the answer; deterministic, not a correctness judge",
        "mean_when_answered": mean([r["key_fact_coverage"] for r in answered]),
        "mean_end_to_end": mean([r["key_fact_coverage"] for r in answerable]),
    }
    if judge_used:
        judged = [r for r in answered if "judge" in r]
        out["judge"] = {
            "n_judged": len(judged), "n_judge_errors": sum(1 for r in answered if "judge_error" in r),
            "correctness": _label_counts(judged, "correctness"), "faithfulness": _label_counts(judged, "faithfulness"),
            "correct_rate_when_judged": (sum(1 for r in judged if r["judge"]["correctness"] == "correct") / len(judged)) if judged else None,
            "correct_rate_end_to_end": (sum(1 for r in judged if r["judge"]["correctness"] == "correct") / len(answerable)) if answerable else None,
            "faithful_rate_when_judged": (sum(1 for r in judged if r["judge"]["faithfulness"] == "faithful") / len(judged)) if judged else None,
        }
    else:
        out["judge"] = {"status": "not_run", "reason": "no judge configured: answer correctness (beyond key-fact coverage) and faithfulness were NOT evaluated"}
    out["by_group"] = _rag_by_group(ok, judge_used)
    return out


def _rag_group(r: Mapping[str, Any]) -> str:
    if not r["answerable"]:
        return "out_of_scope"
    return "multi_passage" if r["qtype"] == "multi_passage" else "single_passage"


def _rag_by_group(ok: Sequence[Mapping[str, Any]], judge_used: bool) -> dict[str, Any]:
    """Single-passage / multi-passage / out-of-scope, so a pooled average cannot hide a weak group.

    Answerable groups: answered rate, key-fact coverage (end to end), citation hit, mean citation recall and
    ``full_citation_recall`` (answers whose citations cover EVERY expected page). Out-of-scope: declined rate.
    """
    groups: dict[str, Any] = {}
    for name in ("single_passage", "multi_passage", "out_of_scope"):
        rs = [r for r in ok if _rag_group(r) == name]
        if not rs:
            continue
        if name == "out_of_scope":
            groups[name] = {"n": len(rs), "correct_abstention_rate": sum(1 for r in rs if not r["grounded"]) / len(rs),
                            "unsupported_answer_ids": [r["id"] for r in rs if r["grounded"]]}
            continue
        answered = [r for r in rs if r["grounded"]]
        g: dict[str, Any] = {
            "n": len(rs), "answered_rate": len(answered) / len(rs),
            "key_fact_coverage_end_to_end": mean([r["key_fact_coverage"] for r in rs]),
            "citation_hit_rate": mean([r["citation"]["hit"] for r in answered]),
            "citation_mean_recall": mean([r["citation"]["recall"] for r in answered]),
            "full_citation_recall": mean([1.0 if r["citation"]["recall"] >= 1.0 else 0.0 for r in answered]),
        }
        if judge_used:
            judged = [r for r in answered if "judge" in r]
            g["n_judged"] = len(judged)
            g["judge_correct_rate_end_to_end"] = sum(1 for r in judged if r["judge"]["correctness"] == "correct") / len(rs)
            g["judge_faithful_rate_when_judged"] = mean([1.0 if r["judge"]["faithfulness"] == "faithful" else 0.0 for r in judged])
        groups[name] = g
    return groups


def _label_counts(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r["judge"][key]] = counts.get(r["judge"][key], 0) + 1
    return dict(sorted(counts.items()))


# ------------------------------------------------------------------ orchestration

def run_evaluation(
    items: Sequence[BenchmarkItem],
    retriever: Any,
    *,
    room_id: str,
    ks: Sequence[int] = DEFAULT_KS,
    min_score: float,
    answerer_factory: Callable[[Any], Any] | None = None,
    judge: Any | None = None,
) -> dict[str, Any]:
    """Score every verified item. ``answerer_factory(recording_retriever)`` builds the Answerer under test."""
    ks = sorted(set(ks))
    scored = [i for i in items if i.verified]
    skipped = [i.id for i in items if not i.verified]
    if not scored:
        raise ValueError("no verified items to score; run scripts/validate_benchmark.py --write first")

    depth = max(ks)
    retrieval_records = [score_retrieval(i, retriever.retrieve(room_id, i.question, depth), ks, min_score) for i in scored]
    results: dict[str, Any] = {
        "items": {"total": len(items), "scored": len(scored), "skipped_draft": skipped,
                  "scored_answerable": sum(1 for i in scored if i.answerable), "scored_out_of_scope": sum(1 for i in scored if not i.answerable)},
        "retrieval": aggregate_retrieval(retrieval_records, ks),
        "threshold_sweep": threshold_sweep(retrieval_records),
        "min_score_in_use": min_score,
    }
    results["retrieval_gate"] = {
        "note": "would the provisional min_score gate let the question through to the LLM? (top-1 score >= min_score)",
        "answerable_pass_rate": mean([1.0 if r["gate_passed"] else 0.0 for r in retrieval_records if r["answerable"]]),
        "out_of_scope_pass_rate": mean([1.0 if r["gate_passed"] else 0.0 for r in retrieval_records if not r["answerable"]]),
    }

    rag_records: list[dict[str, Any]] = []
    if answerer_factory is None:
        results["rag"] = {"status": "not_run", "reason": "no LLM configured: the Answerer was not run, so answers, citations, faithfulness and LLM-side abstention were NOT evaluated (see retrieval_gate for the retrieval-side gate only)"}
    else:
        recorder = RecordingRetriever(retriever)
        answerer = answerer_factory(recorder)
        rag_records = [evaluate_rag_item(i, answerer, recorder, room_id, min_score, judge) for i in scored]
        results["rag"] = aggregate_rag(rag_records, judge_used=judge is not None)

    rag_by_id = {r["id"]: r for r in rag_records}
    results["per_question"] = [{**r, **({"rag": rag_by_id[r["id"]]} if r["id"] in rag_by_id else {})} for r in retrieval_records]
    return results
"""Replay hybrid fusion OFFLINE from a stored dense run, without loading any embedding model.

    uv run python scripts/replay_hybrid.py eval/results/baseline_again/results.json --out-dir eval/results/replay_hybrid_baseline
    uv run python scripts/replay_hybrid.py eval/results/exclude_index/results.json --exclude-non-content --out-dir eval/results/replay_hybrid_exclude

What it does: takes the dense candidates a finished run_eval.py run stored for every question (its top
``--ks`` maximum, normally 5), computes the BM25 ranking over the same room's stored chunk texts with the
shipped HybridRetriever code, fuses the two lists with the same Reciprocal Rank Fusion function, and scores
the result with the same metric code run_eval.py uses. The output ``results.json`` can be given to
``scripts/compare_runs.py``.

What it is NOT: a replacement for ``run_eval.py --retrieval-mode hybrid`` at the default pool. The dense list
is only as deep as the stored run (normally 5), so the replay equals a hybrid with ``--candidate-pool`` equal to
that depth, not the default of 50: a chunk dense search would rank 6th or lower cannot contribute here. For that
pool size the replay is exact (tests/test_replay_hybrid.py checks it against the real HybridRetriever): a chunk
found only by BM25 lies outside the dense candidates, so its cosine cannot beat any dense candidate's and ties are
decided the same way the real retriever decides them. What is NOT available is the cosine of such chunks, so the
abstention gate and threshold sweep are left empty (the real run computes them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
try:
    import rafeeq.knowledge  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))

from rafeeq.knowledge.evaluation.dataset import BenchmarkError, load_benchmark  # noqa: E402
from rafeeq.knowledge.evaluation.ground_truth import load_store_chunks  # noqa: E402
from rafeeq.knowledge.evaluation.runner import aggregate_retrieval, score_retrieval  # noqa: E402
from rafeeq.knowledge.retrieval.hybrid_retriever import RRF_K, _RoomLexicon, reciprocal_rank_fusion  # noqa: E402
from rafeeq.knowledge.store_persistence import DEFAULT_ROOM_ID, DEFAULT_STORE_SUBDIR, PersistenceError  # noqa: E402

DEFAULT_BENCHMARK = KB_DIR / "eval" / "benchmark" / "hands_on_llm_v1.jsonl"
UNKNOWN = float("-inf")


class ReplayError(Exception):
    """The replay cannot be done as requested."""


def fuse(dense: list[dict[str, Any]], lexical: list[tuple[dict[str, Any], float]], k: int) -> list[dict[str, Any]]:
    """Fuse the stored dense hits with BM25 hits in the order HybridRetriever would return them.

    A BM25-only chunk has no known cosine; it is placed after dense candidates on equal fusion score, which is what
    the real tie-break (higher cosine first) does because it lies outside the dense candidate list.
    """
    entries: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(dense, start=1):
        entries[hit["chunk_id"]] = {"hit": hit, "cosine": hit["score"], "dense_rank": rank, "lexical_rank": None, "lexical_score": None}
    for rank, (chunk, bm25) in enumerate(lexical, start=1):
        entry = entries.get(chunk["chunk_id"])
        if entry is None:
            entry = entries[chunk["chunk_id"]] = {
                "hit": {"chunk_id": chunk["chunk_id"], "file_name": chunk["file_name"], "page": chunk["page"], "topic_id": chunk.get("topic_id"),
                        "text_preview": chunk["text"]},
                "cosine": None, "dense_rank": None, "lexical_rank": None, "lexical_score": None}
        entry["lexical_rank"], entry["lexical_score"] = rank, bm25
    scores = reciprocal_rank_fusion([[h["chunk_id"] for h in dense], [c["chunk_id"] for c, _ in lexical]], k)
    order = sorted(entries, key=lambda cid: (-scores[cid], -(entries[cid]["cosine"] if entries[cid]["cosine"] is not None else UNKNOWN), cid))
    rows = []
    for cid in order:
        e = entries[cid]
        rows.append({"chunk_id": cid, "file_name": e["hit"]["file_name"], "page": e["hit"]["page"], "topic_id": e["hit"].get("topic_id"),
                     "score": e["cosine"], "text": e["hit"].get("text_preview", ""), "fusion_score": scores[cid],
                     "dense_rank": e["dense_rank"], "lexical_rank": e["lexical_rank"], "lexical_score": e["lexical_score"]})
    return rows


def replay(source: dict[str, Any], items: list[Any], chunks: list[dict[str, Any]], room_id: str, *, pool: int, k: int, exclude_non_content: bool,
           ks: list[int]) -> list[dict[str, Any]]:
    """Per-question records for the hybrid replay (same record format as run_eval.py)."""
    room_chunks = [dict(c, document_id=c.get("document_id", c["file_name"])) for c in chunks if c.get("room_id") == room_id]
    if not room_chunks:
        raise ReplayError(f"room {room_id!r} has no chunks in the store")
    lexicon = _RoomLexicon(room_chunks)
    stored = {r["id"]: r for r in source["per_question"]}
    depth = max(ks)
    records: list[dict[str, Any]] = []
    for item in items:
        if not item.verified:
            continue
        row = stored.get(item.id)
        if row is None or row.get("question") != item.question:
            raise ReplayError(f"question {item.id} is missing from the source run or its text differs; the replay needs the same benchmark")
        dense = [dict(h) for h in row["retrieved"][:pool]]
        if len(dense) < pool:
            raise ReplayError(f"question {item.id}: the source run stored {len(dense)} chunks, fewer than --pool {pool}")
        lexical = lexicon.rank(item.question, pool, exclude_non_content)
        fused = fuse(dense, lexical, k)
        hits = [dict(h, score=h["score"] if h["score"] is not None else 0.0) for h in fused[:depth]]
        record = score_retrieval(item, hits, ks, 0.35)
        record["retrieved"] = [dict(r, score=fused[i]["score"], **{f: fused[i][f] for f in ("fusion_score", "dense_rank", "lexical_rank", "lexical_score")})
                               for i, r in enumerate(record["retrieved"])]
        # The cosine of BM25-only chunks is unknown here, so nothing about the abstention gate can be claimed.
        record["top1_score"] = record["retrieved"][0]["score"] if record["retrieved"] and record["retrieved"][0]["score"] is not None else None
        record["max_score"], record["gate_passed"] = None, None
        records.append(record)
    return records


def benchmark_hashes(path: str | Path) -> list[str]:
    """sha256 prefixes of the benchmark file as stored and with Windows (CRLF) line endings.

    run_eval.py hashes the raw bytes, so a run made on Windows records the CRLF hash while the same content
    checked out with LF endings (git, Linux) hashes differently. Both count as "the same benchmark".
    """
    raw = Path(path).read_bytes()
    crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return sorted({hashlib.sha256(raw).hexdigest()[:12], hashlib.sha256(crlf).hexdigest()[:12]})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="replay_hybrid.py", description=__doc__.split("\n\n")[0])
    p.add_argument("source", help="results.json of a finished dense run_eval.py run")
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    p.add_argument("--store-dir", default=str(PROJECT_ROOT / DEFAULT_STORE_SUBDIR), help="saved store (only metadata.json is read)")
    p.add_argument("--room-id", default=DEFAULT_ROOM_ID)
    p.add_argument("--pool", type=int, default=None, help="candidates per list; default and maximum: the depth stored in the source run")
    p.add_argument("--rrf-k", type=int, default=RRF_K)
    p.add_argument("--exclude-non-content", action="store_true", help="drop index-like pages from the BM25 list (use with a source run made with the same option)")
    p.add_argument("--out-dir", required=True, help="new directory for results.json (an existing results.json is never overwritten)")
    return p


def run(args: argparse.Namespace, out: TextIO) -> Path:
    out_dir = Path(args.out_dir)
    if (out_dir / "results.json").exists():
        raise ReplayError(f"{out_dir / 'results.json'} already exists; results are never overwritten")
    try:
        source = json.loads(Path(args.source).read_text(encoding="utf-8"))
        ks = [int(x) for x in source["retrieval"]["ks"]]
        depth = max(len(r["retrieved"]) for r in source["per_question"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReplayError(f"cannot read a run from {args.source}: {exc}") from exc
    pool = args.pool or min(len(r["retrieved"]) for r in source["per_question"])
    if pool > depth or pool <= 0:
        raise ReplayError(f"--pool must be between 1 and the stored depth {depth}")
    items = load_benchmark(args.benchmark)
    shas = benchmark_hashes(args.benchmark)
    src_meta = source.get("meta", {})
    if not any(sha in str(src_meta.get("benchmark_file", "")) for sha in shas):
        raise ReplayError(f"the source run used a different benchmark ({src_meta.get('benchmark_file')}); this file hashes to {' / '.join(shas)}")
    chunks = load_store_chunks(args.store_dir)
    records = replay(source, items, chunks, args.room_id, pool=pool, k=args.rrf_k, exclude_non_content=args.exclude_non_content, ks=ks)

    label = (f"hybrid REPLAY (dense ranks from stored run, BM25 recomputed; RRF k={args.rrf_k}, equal weights, candidate pool {pool})"
             + (", index-like pages excluded" if args.exclude_non_content else ""))
    results = {
        "items": source.get("items", {}),
        "retrieval": aggregate_retrieval(records, ks),
        "threshold_sweep": [],
        "min_score_in_use": None,
        "retrieval_gate": {"note": "not computed in a replay: the cosine of BM25-only chunks is unknown without the query embedding"},
        "rag": {"status": "not_run", "reason": "offline replay"},
        "per_question": records,
        "replay": {"pool": pool, "rrf_k": args.rrf_k, "source_run": str(args.source)},
        "meta": {**{k: v for k, v in src_meta.items() if k in ("benchmark_name", "benchmark_file", "store", "room", "documents in room", "ks")},
                 "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                 "embedder": f"{src_meta.get('embedder', '?')} (dense ranks replayed from {Path(args.source).parent.name}; no model loaded)",
                 "retrieval": label, "answerer": "not run", "answer LLM": "none", "judge": "none"},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = results["retrieval"]["overall"]
    print(f"{label}\nOver {overall['n']} answerable question(s): " + ", ".join(f"Hit@{k}={overall[f'hit@{k}']:.3f}" for k in sorted(ks))
          + f", MRR@{max(ks)}={overall[f'mrr@{max(ks)}']:.3f}", file=out)
    print(f"Wrote {out_dir / 'results.json'} (compare it with scripts/compare_runs.py)", file=out)
    return out_dir


def main(argv: list[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    args = build_parser().parse_args(argv)
    try:
        run(args, out)
    except (ReplayError, BenchmarkError, PersistenceError) as exc:
        print(f"error: {exc}", file=err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Evaluate retrieval (always) and the RAG answer layer (only if an LLM is configured).

    # retrieval only: needs the saved store and BGE-M3, no LLM, no network
    uv run python scripts/run_eval.py

    # also run the existing Answerer with an OpenAI-compatible LLM (Groq shown; the key comes from the environment)
    $env:LLM_API_KEY = "..."
    uv run python scripts/run_eval.py --llm-base-url https://api.groq.com/openai/v1 --llm-model <model name>

    # Gemini, provisional RAG configuration (Hybrid + exclusion, pool 50); the key comes from the environment
    $env:GEMINI_API_KEY = "..."
    uv run python scripts/run_eval.py --llm-provider gemini --llm-model <model name> --provisional-rag-config --llm-min-interval 6

    # additionally grade correctness / faithfulness with an LLM judge (optional; see EVALUATION.md for limits)
    uv run python scripts/run_eval.py --llm-base-url ... --llm-model ... --judge

Only items with status "verified" are scored (see scripts/validate_benchmark.py). The vector store is
only read. Writes <out-dir>/results.json, per_question.csv and report.md. Exit status: 0 = evaluation
completed, 1 = it could not run (reason on stderr), 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ lives here)
KB_DIR = Path(__file__).resolve().parents[1]  # rafeeq/knowledge (eval/ lives here)
try:
    import rafeeq.knowledge  # noqa: F401
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))

from rafeeq.knowledge.answering.answerer import DEFAULT_K as DEFAULT_ANSWER_K  # noqa: E402
from rafeeq.knowledge.answering.answerer import DEFAULT_MIN_SCORE, Answerer  # noqa: E402
from rafeeq.knowledge.evaluation.dataset import BenchmarkError, load_benchmark  # noqa: E402
from rafeeq.knowledge.evaluation.ground_truth import load_store_chunks  # noqa: E402
from rafeeq.knowledge.evaluation.judge import LLMJudge, OpenAICompatibleLLM  # noqa: E402
from rafeeq.knowledge.evaluation.report import csv_text, render_report  # noqa: E402
from rafeeq.knowledge.evaluation.runner import DEFAULT_KS, run_evaluation  # noqa: E402
from rafeeq.knowledge.retrieval.factory import DEFAULT_RETRIEVAL_MODE, RETRIEVAL_MODES, build_retriever  # noqa: E402
from rafeeq.knowledge.retrieval.hybrid_retriever import DEFAULT_CANDIDATE_POOL, RRF_K  # noqa: E402
from rafeeq.knowledge.retrieval.reranker import (  # noqa: E402
    DEFAULT_RERANK_BATCH_SIZE,
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_POOL,
    DEFAULT_RERANKER_MODEL,
)
from rafeeq.knowledge.store_persistence import (  # noqa: E402
    DEFAULT_ROOM_ID,
    DEFAULT_STORE_SUBDIR,
    PersistenceError,
    ensure_dimension,
    load_existing_store,
)

DEFAULT_BENCHMARK = KB_DIR / "eval" / "benchmark" / "hands_on_llm_v1.jsonl"


class EvalSetupError(Exception):
    """The evaluation cannot run as configured."""


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_eval.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK), help="JSONL benchmark (default: %(default)s)")
    p.add_argument("--store-dir", default=str(PROJECT_ROOT / DEFAULT_STORE_SUBDIR), help="saved vector store (read only)")
    p.add_argument("--room-id", default=DEFAULT_ROOM_ID, help="room to search (default: %(default)s)")
    p.add_argument("--ks", type=_positive_int, nargs="+", default=list(DEFAULT_KS), help="cut-offs K (default: 1 3 5)")
    p.add_argument("--out-dir", default=None, help="where to write results (default: eval/results/<UTC timestamp>)")
    p.add_argument("--device", default="cpu", help="embedding device (default: %(default)s)")
    p.add_argument("--batch-size", type=_positive_int, default=8)
    p.add_argument("--answer-k", type=_positive_int, default=DEFAULT_ANSWER_K, help="chunks the Answerer retrieves (default: the Answerer's own default)")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Answerer's provisional cosine gate (default: the Answerer's own default)")
    p.add_argument("--exclude-non-content", action="store_true", help="skip index-like pages when retrieving (A/B switch; default: off = unchanged baseline behaviour)")
    p.add_argument("--retrieval-mode", choices=RETRIEVAL_MODES, default=DEFAULT_RETRIEVAL_MODE,
                   help="dense (default, unchanged) or hybrid = dense + BM25 fused with Reciprocal Rank Fusion; combine with --exclude-non-content for the 4 modes")
    p.add_argument("--rrf-k", type=_positive_int, default=RRF_K, help="RRF constant, hybrid mode only (fixed design value: %(default)s)")
    p.add_argument("--candidate-pool", type=_positive_int, default=DEFAULT_CANDIDATE_POOL, help="candidates taken from each list before fusion, hybrid mode only (default: %(default)s)")
    r = p.add_argument_group("Cross-Encoder reranking (optional, off by default)")
    r.add_argument("--rerank", action="store_true", help="rerank the candidate pool of the selected retrieval mode with a Cross-Encoder (default: off = unchanged behaviour)")
    r.add_argument("--rerank-pool", type=_positive_int, default=DEFAULT_RERANK_POOL, help="candidates handed to the reranker (default: %(default)s)")
    r.add_argument("--rerank-model", default=DEFAULT_RERANKER_MODEL, help="reranker model (default: %(default)s)")
    r.add_argument("--rerank-device", default=DEFAULT_RERANK_DEVICE, help="auto = cuda if available else cpu (default: %(default)s)")
    r.add_argument("--rerank-batch-size", type=_positive_int, default=DEFAULT_RERANK_BATCH_SIZE, help="default: %(default)s")
    r.add_argument("--rerank-max-length", type=_positive_int, default=DEFAULT_RERANK_MAX_LENGTH, help="max tokens per (query, chunk) pair (default: %(default)s)")
    g = p.add_argument_group("RAG answer layer (optional)")
    g.add_argument("--llm-provider", choices=("openai-compatible", "gemini"), default="openai-compatible",
                   help="gemini = Google's OpenAI-compatible endpoint; sets --llm-base-url and, unless given, --llm-api-key-env GEMINI_API_KEY")
    g.add_argument("--provisional-rag-config", action="store_true",
                   help="retrieve with Hybrid + index-like pages excluded (candidate pool always 50, overriding --candidate-pool): the provisional configuration for the answer layer")
    g.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL, e.g. https://api.groq.com/openai/v1")
    g.add_argument("--llm-model", default=None, help="model name for answering (required with --llm-base-url)")
    g.add_argument("--llm-api-key-env", default=None, help="environment variable holding the API key (default: LLM_API_KEY, or GEMINI_API_KEY with --llm-provider gemini)")
    g.add_argument("--llm-temperature", type=float, default=0.0)
    g.add_argument("--llm-max-tokens", type=_positive_int, default=None)
    g.add_argument("--llm-min-interval", type=float, default=0.0, help="minimum seconds between LLM calls (free tiers rate-limit)")
    g.add_argument("--judge", action="store_true", help="grade correctness and faithfulness with an LLM judge")
    g.add_argument("--judge-model", default=None, help="judge model (default: same as --llm-model; a different model is better)")
    g.add_argument("--debug", action="store_true")
    return p


def _make_llm(args: argparse.Namespace, model: str) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(args.llm_base_url, model, args.llm_api_key_env, temperature=args.llm_temperature,
                               max_tokens=args.llm_max_tokens, min_interval=args.llm_min_interval)


def _retrieval_label(args: argparse.Namespace) -> str:
    """The `retrieval` line of the run metadata. The two dense labels are unchanged from earlier runs."""
    if args.retrieval_mode == "dense":
        return "dense, index-like pages excluded" if args.exclude_non_content else "dense (unchanged baseline behaviour)"
    label = f"hybrid (dense + BM25, RRF k={args.rrf_k}, equal weights, candidate pool {args.candidate_pool})"
    return label + (", index-like pages excluded" if args.exclude_non_content else "")


def _with_rerank(label: str, args: argparse.Namespace) -> str:
    return label + f" + Cross-Encoder rerank ({args.rerank_model}, pool {args.rerank_pool})" if args.rerank else label


GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
PROVISIONAL_CANDIDATE_POOL = 50


def _apply_presets(args: argparse.Namespace) -> None:
    """Fill the settings implied by --llm-provider / --provisional-rag-config (an explicit --llm-base-url / key env wins)."""
    if args.llm_provider == "gemini" and not args.llm_base_url:
        args.llm_base_url = GEMINI_OPENAI_BASE_URL
    if args.llm_api_key_env is None:
        args.llm_api_key_env = "GEMINI_API_KEY" if args.llm_provider == "gemini" else "LLM_API_KEY"
    if args.provisional_rag_config:
        args.retrieval_mode, args.exclude_non_content = "hybrid", True
        args.candidate_pool = PROVISIONAL_CANDIDATE_POOL   # enforced: overrides any --candidate-pool


def run(args: argparse.Namespace, *, embedder: Any = None, store_cls: Any = None, llm: Any = None, judge_llm: Any = None, reranker: Any = None, out: TextIO) -> Path:
    _apply_presets(args)
    if args.min_score < -1 or args.min_score > 1:
        raise EvalSetupError("--min-score must be between -1 and 1")
    if llm is None and args.llm_base_url:
        if not args.llm_model:
            raise EvalSetupError("--llm-base-url needs --llm-model")
        llm = _make_llm(args, args.llm_model)
    if (args.judge or args.judge_model) and llm is None:
        raise EvalSetupError("--judge needs an LLM to answer with (--llm-base-url and --llm-model)")
    if args.judge and judge_llm is None:
        judge_llm = _make_llm(args, args.judge_model or args.llm_model)

    if args.out_dir and (Path(args.out_dir) / "results.json").exists():
        raise EvalSetupError(f"{Path(args.out_dir) / 'results.json'} already exists; results are never overwritten. Choose a new --out-dir.")
    if args.retrieval_mode == "dense" and (args.rrf_k, args.candidate_pool) != (RRF_K, DEFAULT_CANDIDATE_POOL):
        raise EvalSetupError("--rrf-k and --candidate-pool only apply with --retrieval-mode hybrid")

    if not args.rerank and (args.rerank_pool != DEFAULT_RERANK_POOL or args.rerank_model != DEFAULT_RERANKER_MODEL):
        raise EvalSetupError("--rerank-pool / --rerank-model only apply with --rerank")
    if args.rerank and args.rerank_pool < max(args.ks):
        raise EvalSetupError(f"--rerank-pool ({args.rerank_pool}) must be at least max(--ks) ({max(args.ks)})")

    items = load_benchmark(args.benchmark)
    scored = [i for i in items if i.verified]
    if not scored:
        raise EvalSetupError(f"{args.benchmark} has no verified items ({len(items)} draft). Run scripts/validate_benchmark.py --write first.")

    store = load_existing_store(args.store_dir, store_cls=store_cls)
    if len(store) == 0:
        raise EvalSetupError(f"the store in {args.store_dir} is empty")
    if embedder is None:
        from rafeeq.knowledge.embeddings.embedder import Embedder

        embedder = Embedder(device=args.device, batch_size=args.batch_size)
    ensure_dimension(store, embedder.dimension, args.store_dir)

    room_files = sorted({r["file_name"] for r in load_store_chunks(args.store_dir) if r.get("room_id") == args.room_id})
    if not room_files:
        raise EvalSetupError(f"room {args.room_id!r} has no chunks in the store (use --room-id)")
    absent = sorted({i.file_name for i in scored if i.answerable and i.file_name not in room_files})
    if absent:
        raise EvalSetupError(f"benchmark expects {', '.join(absent)} but room {args.room_id!r} holds: {', '.join(room_files)}")

    benchmark_bytes = Path(args.benchmark).read_bytes()
    now = datetime.now(timezone.utc)
    meta: dict[str, Any] = {
        "benchmark_name": Path(args.benchmark).stem,
        "run_at_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_file": f"{args.benchmark} (sha256 {hashlib.sha256(benchmark_bytes).hexdigest()[:12]})",
        "store": f"{args.store_dir} ({len(store)} vectors)",
        "room": args.room_id,
        "documents in room": ", ".join(room_files),
        "embedder": f"{getattr(embedder, 'model_name', type(embedder).__name__)} on {getattr(embedder, 'device', '?')}",
        "ks": " ".join(map(str, sorted(set(args.ks)))),
        "retrieval": _with_rerank(_retrieval_label(args), args),
        "answerer": f"k={args.answer_k}, min_score={args.min_score}" if llm is not None else "not run",
        "answer LLM": json.dumps(llm.describe() if hasattr(llm, "describe") else {"model": type(llm).__name__}) if llm is not None else "none",
        "judge": json.dumps(judge_llm.describe() if hasattr(judge_llm, "describe") else {"model": type(judge_llm).__name__}) if judge_llm is not None else "none",
    }

    if args.rerank and reranker is None:
        from rafeeq.knowledge.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(args.rerank_model, args.rerank_device, args.rerank_batch_size, args.rerank_max_length)
    retriever = build_retriever(store, embedder, mode=args.retrieval_mode, exclude_non_content=args.exclude_non_content,
                                rrf_k=args.rrf_k, candidate_pool=args.candidate_pool,
                                rerank=args.rerank, reranker=reranker if args.rerank else None, rerank_pool=args.rerank_pool)
    factory = (lambda recorder: Answerer(recorder, llm, k=args.answer_k, min_score=args.min_score)) if llm is not None else None
    print(f"Evaluating {len(scored)} verified question(s) in room {args.room_id!r}...", file=out)
    results = run_evaluation(items, retriever, room_id=args.room_id, ks=args.ks, min_score=args.min_score,
                             answerer_factory=factory, judge=LLMJudge(judge_llm) if judge_llm is not None else None)
    if args.rerank:  # the model is loaded by now: record what actually ran
        meta["reranker"] = json.dumps(reranker.describe() if hasattr(reranker, "describe") else {"model": type(reranker).__name__})
    results["meta"] = meta

    out_dir = Path(args.out_dir) if args.out_dir else KB_DIR / "eval" / "results" / now.strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "per_question.csv").write_text(csv_text(results), encoding="utf-8-sig")  # BOM: Excel shows Arabic correctly
    (out_dir / "report.md").write_text(render_report(results, meta), encoding="utf-8")

    overall = results["retrieval"]["overall"]
    kmax = max(args.ks)
    print(f"Retrieval over {overall['n']} answerable question(s): "
          + ", ".join(f"Hit@{k}={overall[f'hit@{k}']:.3f}" for k in sorted(set(args.ks))) + f", MRR@{kmax}={overall[f'mrr@{kmax}']:.3f}", file=out)
    print(f"RAG answer layer: {results['rag'].get('status', 'run')}", file=out)
    print(f"Wrote {out_dir / 'report.md'}, results.json, per_question.csv", file=out)
    return out_dir


def main(argv: list[str] | None = None, *, embedder: Any = None, store_cls: Any = None, llm: Any = None, judge_llm: Any = None,
         reranker: Any = None, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr
    args = build_parser().parse_args(argv)
    try:
        run(args, embedder=embedder, store_cls=store_cls, llm=llm, judge_llm=judge_llm, reranker=reranker, out=out)
    except KeyboardInterrupt:
        print("error: interrupted; nothing was written.", file=err)
        return 130
    except (EvalSetupError, BenchmarkError, PersistenceError) as exc:
        print(f"error: {exc}", file=err)
        return 1
    except Exception as exc:
        if args.debug:
            traceback.print_exc(file=err)
        print(f"error: {type(exc).__name__}: {exc}", file=err)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    sys.exit(main())
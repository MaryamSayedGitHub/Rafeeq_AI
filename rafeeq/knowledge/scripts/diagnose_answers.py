#!/usr/bin/env python
"""Find out WHY a question ends in the generic refusal, using a handful of LLM calls and NO change to the Answerer.

The Answerer, prompts, min_score and citation logic are used exactly as in run_eval.py. This script only
  * wraps the HTTP call (never the Answerer) to keep the FULL provider response: content, reasoning field,
    finish_reason, token usage;
  * classifies the outcome with a pure function that repeats the Answerer's three refusal conditions;
  * writes everything to a NEW directory (an existing one is refused). The API key is never written.

Stages: http_error (NOT an abstention) | empty | sentinel_exact | sentinel_substring_only | no_valid_marker | accepted

    python scripts/diagnose_answers.py --dry-run --ids h03 h04                       # retrieval + prompt only, 0 LLM calls
    python scripts/diagnose_answers.py --ids h03 h04 h01 --rerank --out-dir <new dir> # 3 LLM calls
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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

from rafeeq.knowledge.answering.answerer import (  # noqa: E402
    DEFAULT_K, DEFAULT_MIN_SCORE, NOT_COVERED_SENTINEL, SYSTEM_PROMPT, Answerer, build_prompt,
)
from rafeeq.knowledge.evaluation.dataset import load_benchmark  # noqa: E402
from rafeeq.knowledge.evaluation.judge import _THINK, OpenAICompatibleLLM, _default_post  # noqa: E402
from rafeeq.knowledge.evaluation.runner import RecordingRetriever  # noqa: E402
from rafeeq.knowledge.retrieval.factory import build_retriever  # noqa: E402
from rafeeq.knowledge.store_persistence import DEFAULT_ROOM_ID, DEFAULT_STORE_SUBDIR, ensure_dimension, load_existing_store  # noqa: E402

STAGES = ("http_error", "empty", "sentinel_exact", "sentinel_substring_only", "no_valid_marker", "accepted")
_EXACT = re.compile(r"[\s`*_\"'.]*" + NOT_COVERED_SENTINEL + r"[\s`*_\"'.]*")


def classify(text: str, available: int) -> dict[str, Any]:
    """Repeat the Answerer's refusal conditions on the text it receives (after _THINK stripping), without changing them."""
    text = text.strip()
    cited: list[int] = []
    if text:
        _, cited = Answerer._clean_markers(text, available)
    if not text:
        stage = "empty"
    elif NOT_COVERED_SENTINEL in text:
        stage = "sentinel_exact" if _EXACT.fullmatch(text) else "sentinel_substring_only"
    elif not cited:
        stage = "no_valid_marker"
    else:
        stage = "accepted"
    # hints only; they never change the stage
    raw_markers = re.findall(r"\[[^\]\n]{1,40}\]|【[^】\n]{1,20}】|\(\s*\d+(?:\s*,\s*\d+)*\s*\)", text)
    return {"stage": stage, "valid_cited_numbers": cited, "marker_like_strings": raw_markers[:12],
            "unclosed_think": bool(re.search(r"<think>", text, re.I)), "text_chars": len(text)}


def parse_body(body: str) -> dict[str, Any]:
    """Pull the diagnostic fields out of an OpenAI-compatible response body (tolerates odd shapes)."""
    try:
        data = json.loads(body)
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        return {"finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
                "content": msg.get("content"), "reasoning": msg.get("reasoning") or msg.get("reasoning_content"),
                "message_keys": sorted(msg.keys())}
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return {"parse_error": True}


class RecordingPost:
    """Drop-in for the LLM's `post`: forwards to the real HTTP call and keeps (status, request payload, body). No headers."""

    def __init__(self, inner=_default_post) -> None:
        self._inner, self.calls = inner, []

    def __call__(self, url, headers, body, timeout):
        status, text = self._inner(url, headers, body, timeout)
        self.calls.append({"status": status, "request": json.loads(body.decode("utf-8")), "response_body": text})
        return status, text


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--benchmark", default=str(KB_DIR / "eval" / "benchmark" / "hands_on_llm_heldout_v1.jsonl"))
    p.add_argument("--store-dir", default=str(PROJECT_ROOT / DEFAULT_STORE_SUBDIR))
    p.add_argument("--room-id", default=DEFAULT_ROOM_ID)
    p.add_argument("--out-dir", default=None, help="NEW directory (required unless --dry-run)")
    p.add_argument("--device", default="cpu", help="embedder device (the earlier runs used cpu)")
    p.add_argument("--retrieval-mode", choices=("dense", "hybrid"), default="hybrid")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-pool", type=int, default=50)
    p.add_argument("--rerank-device", default="auto")
    p.add_argument("--answer-k", type=int, default=DEFAULT_K)
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    p.add_argument("--llm-base-url", default="https://api.groq.com/openai/v1")
    p.add_argument("--llm-model", default="openai/gpt-oss-120b")
    p.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    p.add_argument("--llm-temperature", type=float, default=0.0)
    p.add_argument("--llm-max-tokens", type=int, default=None, help="default None = exactly as the earlier runs")
    p.add_argument("--llm-min-interval", type=float, default=20.0)
    p.add_argument("--llm-max-retries", type=int, default=3, help="HTTP retries on 429/5xx (default: as the earlier runs)")
    p.add_argument("--repeat", type=int, default=1, help="calls per question (temperature 0 is not guaranteed deterministic)")
    p.add_argument("--max-calls", type=int, default=6, help="refuse to start if ids x repeat exceeds this (quota guard)")
    p.add_argument("--dry-run", action="store_true", help="retrieval + prompt only; makes NO LLM call")
    return p


def main(argv=None, *, embedder: Any = None, reranker: Any = None, post: Any = None, out: TextIO = sys.stdout, err: TextIO = sys.stderr) -> int:
    args = _parser().parse_args(argv)
    calls_needed = 0 if args.dry_run else len(args.ids) * args.repeat
    if calls_needed > args.max_calls:
        print(f"error: {calls_needed} LLM calls requested, --max-calls is {args.max_calls}. Nothing was called.", file=err)
        return 1
    if not args.dry_run:
        if not args.out_dir:
            print("error: --out-dir is required (a NEW directory) unless --dry-run", file=err)
            return 1
        if Path(args.out_dir).exists():
            print(f"error: {args.out_dir} exists; never overwritten.", file=err)
            return 1

    items = {i.id: i for i in load_benchmark(args.benchmark)}
    missing = [i for i in args.ids if i not in items]
    if missing:
        print(f"error: unknown question id(s): {', '.join(missing)}", file=err)
        return 1

    store = load_existing_store(args.store_dir)
    if embedder is None:
        from rafeeq.knowledge.embeddings.embedder import Embedder

        embedder = Embedder(device=args.device, batch_size=8)
    ensure_dimension(store, embedder.dimension, args.store_dir)
    extra: dict[str, Any] = {}
    if args.rerank:
        if reranker is None:
            from rafeeq.knowledge.retrieval.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(device=args.rerank_device)
        extra = {"rerank": True, "reranker": reranker, "rerank_pool": args.rerank_pool}
    recorder = RecordingRetriever(build_retriever(store, embedder, mode=args.retrieval_mode, **extra))

    recording = post if isinstance(post, RecordingPost) else RecordingPost(post or _default_post)
    llm = None if args.dry_run else OpenAICompatibleLLM(
        args.llm_base_url, args.llm_model, args.llm_api_key_env, temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens, min_interval=args.llm_min_interval, max_retries=args.llm_max_retries, post=recording)
    answerer = Answerer(recorder, llm if llm is not None else _NoLLM(), k=args.answer_k, min_score=args.min_score)

    records: list[dict[str, Any]] = []
    for qid in args.ids:
        item = items[qid]
        for attempt in range(1, args.repeat + 1):
            recorder.last = []
            before = len(recording.calls)
            rec: dict[str, Any] = {"id": qid, "attempt": attempt, "question": item.question, "answerable": item.answerable}
            try:
                retrieved = recorder.retrieve(args.room_id, item.question, args.answer_k)
                relevant = [c for c in retrieved if c["score"] >= args.min_score]
                rec["excerpts_given"] = [{"n": n, "chunk_id": c["chunk_id"], "page": c["page"], "score": round(c["score"], 4),
                                          "is_expected_chunk": c["chunk_id"] in item.expected_chunk_ids} for n, c in enumerate(relevant, 1)]
                prompt = build_prompt(item.question, relevant) if relevant else ""
                rec["prompt_sha256"] = hashlib.sha256((SYSTEM_PROMPT + prompt).encode("utf-8")).hexdigest()[:16]
                rec["prompt_chars"] = len(prompt)
                if args.dry_run:
                    rec["stage"] = "dry_run" if relevant else "retrieval_gate"
                    records.append(rec); continue
                if not relevant:
                    rec["stage"] = "retrieval_gate"
                    records.append(rec); continue
                result = answerer.answer(args.room_id, item.question)
            except Exception as exc:  # LLMError (HTTP 429, ...) is recorded as an ERROR, never as an abstention
                rec["stage"] = "http_error"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["http"] = recording.calls[before:]
                records.append(rec); continue
            call = recording.calls[-1]
            parsed = parse_body(call["response_body"])
            text_after_think = _THINK.sub("", parsed.get("content") or "").strip()
            verdict = classify(text_after_think, len(relevant))
            rec.update({"finish_reason": parsed.get("finish_reason"), "usage": parsed.get("usage"), "message_keys": parsed.get("message_keys"),
                        "raw_content": parsed.get("content"), "raw_reasoning": parsed.get("reasoning"),
                        "text_seen_by_answerer": text_after_think, "grounded_by_answerer": bool(result["grounded"]),
                        "request_max_tokens": call["request"].get("max_tokens"), **verdict})
            if (verdict["stage"] == "accepted") != bool(result["grounded"]):
                rec["WARNING"] = "classifier and Answerer disagree: do not trust this row"
            records.append(rec)

    summary = _summary(args, records)
    print(summary, file=out)
    if not args.dry_run:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True)
        with (out_dir / "diagnosis.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out_dir / "summary.md").write_text(summary, encoding="utf-8")
        (out_dir / "meta.json").write_text(json.dumps({
            "run_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "model": args.llm_model, "base_url": args.llm_base_url,
            "temperature": args.llm_temperature, "max_tokens": args.llm_max_tokens, "retrieval_mode": args.retrieval_mode, "rerank": args.rerank,
            "min_score": args.min_score, "answer_k": args.answer_k, "ids": args.ids, "repeat": args.repeat}, indent=2), encoding="utf-8")
        print(f"\nSaved to {out_dir}", file=out)
    return 0


class _NoLLM:
    def generate(self, system, prompt):  # pragma: no cover - dry run never reaches the LLM
        raise RuntimeError("dry run: no LLM call is allowed")


def _summary(args, records) -> str:
    lines = ["# Answer diagnosis" + (" (DRY RUN: no LLM was called)" if args.dry_run else ""), "",
             "| id | try | stage | finish_reason | completion/total tokens | cited | expected chunk shown at n | note |", "|---|---|---|---|---|---|---|---|"]
    for r in records:
        usage = r.get("usage") or {}
        exp = [e["n"] for e in r.get("excerpts_given", []) if e["is_expected_chunk"]]
        note = r.get("WARNING") or (r.get("error", "")[:80] if r["stage"] == "http_error" else "")
        lines.append(f"| {r['id']} | {r['attempt']} | **{r['stage']}** | {r.get('finish_reason', '-')} | "
                     f"{usage.get('completion_tokens', '-')}/{usage.get('total_tokens', '-')} | {r.get('valid_cited_numbers', '-')} | {exp or '-'} | {note} |")
    lines += ["", "Meaning: http_error = API failure, NOT an abstention; empty = no text after think-stripping; sentinel_exact = the model itself said NOT_COVERED; "
              "sentinel_substring_only = NOT_COVERED appears inside a longer text (the substring check rejects it); no_valid_marker = no [n] in range; accepted = would be grounded."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())

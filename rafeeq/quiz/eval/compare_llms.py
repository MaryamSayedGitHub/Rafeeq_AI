"""Compare LLMs on the same book pages.

Run:  uv run python -m rafeeq.quiz.eval.compare_llms --n 20
"""
from __future__ import annotations

import argparse
import json
import random
import time

from rafeeq.quiz.dev.book_chunks import load_chunks
from rafeeq.quiz.quiz.generator import generate_item, select_chunks
from rafeeq.quiz.quiz.llm import DEFAULT_MODELS, make_llm


def run_one(provider: str, model: str, chunks: list[dict]) -> dict:
    llm = make_llm(provider, model)
    first_ok = produced = errors = 0
    streak = 0  # consecutive errors
    seconds: list[float] = []
    items = []
    for i, chunk in enumerate(chunks, start=1):
        start = time.time()
        try:
            item, first = generate_item(llm, chunk, f"{provider}-{i}")
            streak = 0
        except Exception as exc:  # API error, rate limit, bad key...
            errors += 1
            streak += 1
            print(f"  [{provider}] page {chunk['page']}: {type(exc).__name__}: {str(exc)[:200]}")
            if streak >= 3:
                print(f"  [{provider}] 3 errors in a row, skipping this provider")
                break
            continue
        seconds.append(time.time() - start)
        print(f"  [{provider}] {i}/{len(chunks)} done")
        first_ok += int(first)
        if item is not None:
            produced += 1
            items.append(item.model_dump())
    n = len(chunks)
    return {
        "provider": provider, "model": model, "attempted": n, "errors": errors,
        "first_try_pct": 100 * first_ok / n if n else 0,
        "produced_pct": 100 * produced / n if n else 0,
        "avg_seconds": sum(seconds) / len(seconds) if seconds else 0,
        "items": items,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default="data/Hands-On_Large_Language_Models.pdf")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--providers", nargs="+", default=["openai", "gemini"])
    p.add_argument("--openai-model", default=DEFAULT_MODELS["openai"])
    p.add_argument("--gemini-model", default=DEFAULT_MODELS["gemini"])
    p.add_argument("--out", default="data/compare_samples.json")
    args = p.parse_args()

    chunks = select_chunks(load_chunks(args.pdf), args.n, rng=random.Random(args.seed))
    print(f"{len(chunks)} pages selected (same pages for every model)\n")

    models = {"openai": args.openai_model, "gemini": args.gemini_model}
    results = []
    for provider in args.providers:
        print(f"running {provider} / {models[provider]} ...")
        results.append(run_one(provider, models[provider], chunks))

    print(f"\n{'model':<34}{'first try':>10}{'produced':>10}{'errors':>8}{'sec/item':>10}")
    for r in results:
        print(f"{r['provider'] + '/' + r['model']:<34}{r['first_try_pct']:>9.0f}%"
              f"{r['produced_pct']:>9.0f}%{r['errors']:>8}{r['avg_seconds']:>10.1f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nall generated questions saved to {args.out} for manual review")


if __name__ == "__main__":
    main()
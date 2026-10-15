"""Score prepared RAGAS JSONL samples using an OpenAI-compatible judge."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall, Faithfulness


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return rows


async def score_rows(rows: list[dict], model: str, base_url: str | None) -> list[dict]:
    api_key = os.environ.get("RAGAS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set RAGAS_API_KEY (or OPENAI_API_KEY) in the environment first."
        )

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or os.environ.get("RAGAS_BASE_URL") or None,
    )
    judge = llm_factory(model, client=client)

    faithfulness = Faithfulness(llm=judge)
    context_recall = ContextRecall(llm=judge)
    output = []

    try:
        for row in rows:
            required = ("id", "user_input", "response", "retrieved_contexts", "reference")
            missing = [key for key in required if not row.get(key)]
            if missing:
                raise ValueError(
                    f"{row.get('id', '<unknown>')}: missing fields {missing}"
                )

            faith = await faithfulness.ascore(
                user_input=row["user_input"],
                response=row["response"],
                retrieved_contexts=row["retrieved_contexts"],
            )
            recall = await context_recall.ascore(
                user_input=row["user_input"],
                reference=row["reference"],
                retrieved_contexts=row["retrieved_contexts"],
            )

            output.append({
                "id": row["id"],
                "faithfulness": faith.value,
                "context_recall": recall.value,
                "retrieved_chunk_ids": row.get("retrieved_chunk_ids", []),
            })
    finally:
        await client.close()

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prepared RAGAS input JSONL")
    parser.add_argument("--model", required=True, help="Judge model name")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible API base URL")
    parser.add_argument("--out-dir", default="eval/results/ragas")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    scores = asyncio.run(score_rows(rows, args.model, args.base_url))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = out_dir / f"scores_{timestamp}.json"

    with output_path.open("x", encoding="utf-8") as stream:
        json.dump({
            "created_at_utc": timestamp,
            "model": args.model,
            "input": str(Path(args.input)),
            "sample_count": len(scores),
            "scores": scores,
            "mean_faithfulness": sum(
                r["faithfulness"] for r in scores
                if r["faithfulness"] is not None
            ) / max(1, sum(r["faithfulness"] is not None for r in scores)),
            "mean_context_recall": sum(
                r["context_recall"] for r in scores
                if r["context_recall"] is not None
            ) / max(1, sum(r["context_recall"] is not None for r in scores)),
        }, stream, ensure_ascii=False, indent=2)

    print(f"Saved {len(scores)} scored samples to: {output_path}")


if __name__ == "__main__":
    main()

"""Prepare RAGAS samples from saved answers and exact stored chunk IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rafeeq.knowledge.evaluation.dataset import load_benchmark
from rafeeq.knowledge.evaluation.ground_truth import load_store_chunks


def _load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("per_question")
        if not isinstance(rows, list):
            raise ValueError(
                f"{path}: expected a 'per_question' list. "
                "This file may be retrieval-only."
            )
    else:
        raise ValueError(f"{path}: unsupported results format")

    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every result row must be an object")

    return rows


def prepare_samples(
    benchmark_path: str | Path,
    results_path: str | Path,
    store_dir: str | Path,
) -> list[dict[str, Any]]:
    """Join verified answerable benchmark items to saved answers and exact contexts.

    Result rows must contain an answer (answer, answer_md, or response) and
    retrieved chunk IDs (retrieved, excerpts_given, or retrieved_chunk_ids).
    No LLM calls are made here.
    """
    benchmark = load_benchmark(benchmark_path)
    result_rows = _load_result_rows(results_path)

    results_by_id = {}
    for row in result_rows:
        item_id = row.get("id", row.get("item_id"))
        if not item_id:
            raise ValueError("A result row has no id")
        if item_id in results_by_id:
            raise ValueError(f"Duplicate result id: {item_id}")
        results_by_id[item_id] = row

    records = load_store_chunks(store_dir)
    chunks_by_id = {record["chunk_id"]: record for record in records}

    samples: list[dict[str, Any]] = []
    missing_answers: list[str] = []

    for item in benchmark:
        if not item.verified or not item.answerable:
            continue

        row = results_by_id.get(item.id)
        if row is None:
            continue

        answer = row.get("answer") or row.get("answer_md") or row.get("response")
        if not isinstance(answer, str) or not answer.strip():
            missing_answers.append(item.id)
            continue

        retrieved = (
            row.get("retrieved_chunk_ids")
            or row.get("excerpts_given")
            or row.get("retrieved")
        )
        if not isinstance(retrieved, list) or not retrieved:
            raise ValueError(
                f"{item.id}: no retrieved chunk IDs. "
                "Do not substitute text_preview for full contexts."
            )

        chunk_ids = []
        for entry in retrieved:
            chunk_id = entry if isinstance(entry, str) else (
                entry.get("chunk_id") if isinstance(entry, dict) else None
            )
            if not chunk_id:
                raise ValueError(f"{item.id}: invalid retrieved chunk entry")
            if chunk_id not in chunks_by_id:
                raise ValueError(
                    f"{item.id}: chunk_id {chunk_id!r} not found in store metadata"
                )
            if chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)

        contexts = [chunks_by_id[cid]["text"] for cid in chunk_ids]
        if not contexts or any(not text.strip() for text in contexts):
            raise ValueError(f"{item.id}: one or more retrieved contexts are empty")

        if not item.reference_answer:
            raise ValueError(f"{item.id}: verified item has no reference_answer")

        samples.append({
            "id": item.id,
            "user_input": item.question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item.reference_answer,
            "reference_contexts": [
                chunks_by_id[cid]["text"]
                for cid in item.expected_chunk_ids
                if cid in chunks_by_id
            ],
            "retrieved_chunk_ids": chunk_ids,
        })

    if missing_answers:
        raise ValueError(
            "Saved results are missing generated answers for: "
            + ", ".join(missing_answers)
            + ". This results file may have been retrieval-only."
        )

    if not samples:
        raise ValueError(
            "No matching answerable benchmark items with saved answers were found."
        )

    return samples


def write_jsonl(samples: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write samples without overwriting an existing file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")

    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

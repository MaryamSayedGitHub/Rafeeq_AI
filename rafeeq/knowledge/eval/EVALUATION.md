# Retrieval and RAG evaluation

Goal: an honest baseline of the current retrieval and grounded-answer quality **before** OCR / vision work,
measured without changing the embedder, chunker, retriever, answerer or the saved store (everything here only reads them).

## What gets measured, and how

| Layer | Needs | Metrics |
|---|---|---|
| Retrieval | saved store + BGE-M3 (no LLM) | Hit@K, Precision@K, Recall@K, FullRecall@K, MRR, share of top-K from another document, failure category per question, abstention-gate pass rates and a score-vs-threshold table |
| RAG answers | an LLM behind `Answerer` (OpenAI-compatible endpoint) | abstention accuracy (false abstention, unsupported answers to out-of-scope questions), citation hit / precision / recall, key-fact coverage |
| Judge (optional) | a second LLM call per answer | correctness and faithfulness labels |

**Relevance (used for every retrieval metric).** A retrieved chunk is relevant when it comes from the question's
expected document **and** its page is one of the question's `expected_pages` (page level, primary). Where
`expected_chunk_ids` exist, a strict chunk-level view is reported too. Similarity scores are never used as correctness labels.
Questions needing several passages list several pages; Recall@K counts distinct expected pages covered (several chunks of one
page count once) and FullRecall@K is 1 only when all of them are found. Precision@K divides by K, so it is capped when few
relevant chunks exist; read it with Recall@K. With ~30 questions the 95% Wilson intervals in the report are wide.

**Not computed unless configured.** Without `--llm-*` flags the report says the RAG layer was **not run**; without `--judge`
it says correctness beyond key-fact coverage and faithfulness were **not evaluated**. *Key-fact coverage* is a deterministic
proxy (are the required facts present), not a correctness judge. The LLM judge is an opinion: it can be wrong, favours answers from
its own model family, judges only against the excerpts shown, and judge models are not comparable with each other. It is off by default.

## Ground truth is verified against the real store, not typed from memory

`eval/benchmark/hands_on_llm_v1.jsonl` has 30 answerable questions (definition 8, explanation 9, comparison 6,
multi-passage 7; 20 English, 10 Arabic) and 5 out-of-scope questions (3 English, 2 Arabic; one is "near-domain": GPT-5).
All 35 are `verified`: each answerable question carries 2-3 verbatim `evidence_quotes` copied from the book (Arabic questions use the
English book passages), a `reference_answer` and `key_facts`. `validate_benchmark.py` found every quote on its expected
page in a store built from the PDF text (`pdftotext` extraction of the 598-page file; chunking it with the project chunker gives 595 chunks,
the same count as the real store). Out-of-scope questions are verified only because none of their `absent_terms` occurs in the book.

What this does **not** establish: the quotes were checked against my text extraction, not against your PyMuPDF-built store,
and `expected_chunk_ids` are empty (page-level metrics only) until you run `validate_benchmark.py --write` on your store.
Reference answers are short paraphrases of the quoted passages written for this benchmark; they are not book text.
An Arabic question is answered from English book text, so Arabic retrieval quality (BGE-M3) is part of what is measured.

Pages are the **physical PDF pages** stored in the chunk metadata (1-indexed), not the numbers printed in the book.

### Completing the benchmark

```powershell
# 1. find where a topic is discussed (keyword search over the stored chunks; no model; English book terms)
uv run python scripts/find_evidence.py "key value cache" --file-name hands_on_llm.pdf

# 2. in the JSONL, for each item: add 1-3 verbatim "evidence_quotes" (>= 25 chars), a short "reference_answer",
#    and "key_facts" ([[any spelling, ...], ...]; for Arabic questions list Arabic and English spellings)

# 3. check without writing, then promote what verifies (keeps a .bak copy of the file)
uv run python scripts/validate_benchmark.py
uv run python scripts/validate_benchmark.py --write
```

## Running the evaluation

```powershell
# retrieval only (no LLM, no network)
uv run python scripts/run_eval.py

# plus the RAG answer layer with the existing Answerer; any OpenAI-compatible endpoint works (the key is read from the environment)
$env:LLM_API_KEY = "<your key>"
uv run python scripts/run_eval.py --llm-base-url https://api.groq.com/openai/v1 --llm-model <model name> --llm-min-interval 2

# optional judge (prefer a DIFFERENT model from the one answering)
uv run python scripts/run_eval.py --llm-base-url ... --llm-model <answer model> --judge --judge-model <judge model>
```

Useful options: `--ks 1 3 5 10`, `--room-id`, `--min-score` / `--answer-k` (defaults = the Answerer's own), `--out-dir`.
Output (default `eval/results/<UTC time>/`): `results.json` (everything, machine-readable), `per_question.csv` (one row per question with
the retrieved pages, scores and chunk ids, the answer and the citations; opens in Excel with Arabic intact), `report.md` (metrics, failing
questions, abstention table, rule-based recommendations). Commit the baseline run so later changes can be compared against it.

## Tests

```powershell
uv run pytest tests\test_eval_metrics.py tests\test_eval_dataset_ground_truth.py tests\test_eval_runner.py tests\test_run_eval_script.py -v
uv run pytest -v
```

## Files

- `src/knowledge_base/evaluation/`: `dataset.py` (schema, validation), `ground_truth.py` (quote lookup in stored chunks), `metrics.py`, `runner.py`, `judge.py` (optional LLM client and judge), `report.py`
- `scripts/`: `find_evidence.py`, `validate_benchmark.py`, `run_eval.py`
- `eval/benchmark/hands_on_llm_v1.jsonl`: the benchmark (35 verified items, page-level ground truth)

## Known limits

- `instance_segmentation.pdf` is not part of the benchmark (mostly images); it only matters here as a possible source of intruding chunks in the same room, which the report shows as "share of top-K from another document".
- Do not tune retrieval or the prompt on these questions and then quote the same numbers as a held-out result.
- Retrieval is run on the room exactly as the product uses it; the answerer is run with its own `k` and `min_score`.

## Hybrid retrieval modes

`run_eval.py --retrieval-mode {dense,hybrid}` (default `dense`) selects the retriever; `--rrf-k` (60) and `--candidate-pool` (50) apply to
hybrid only and are refused with dense. The mode is recorded in `results.json` under `meta.retrieval`. In hybrid runs each retrieved row also stores
`fusion_score`, `dense_rank`, `lexical_rank` and `lexical_score`; `score` stays the true cosine. Each record has `max_score` (best cosine among the
retrieved chunks), which the threshold sweep uses (for old dense runs it equals `top1_score`). Existing result directories are never overwritten.

`scripts/replay_hybrid.py` re-ranks a stored dense run offline (no model). Its candidate pool equals the stored depth (5), it leaves the abstention
gate and the sweep empty, and it labels itself "hybrid REPLAY". It does not replace a real hybrid run at the default pool of 50.
Reports: `eval/analysis/`. Audit of q01/q21: `eval/audit/q01_q21_audit.md`.

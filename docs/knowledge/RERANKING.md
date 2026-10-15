# Optional Cross-Encoder reranking (`BAAI/bge-reranker-v2-m3`)

```
existing retrieval (dense or hybrid) -> candidate pool (default 50) -> Cross-Encoder -> top 6 -> existing Answerer
```

Off by default. Without `--rerank` every code path is the one that existed at commit `a0f9c0b`.

## What changed
| file | change |
|---|---|
| `src/knowledge_base/retrieval/reranker.py` | NEW: `CrossEncoderReranker` (lazy, loaded once, `auto` device = CUDA else CPU, one CPU retry if the GPU load fails) and `RerankingRetriever` (wraps any retriever) |
| `src/knowledge_base/retrieval/factory.py` | `build_retriever(..., rerank=False, reranker=None, rerank_pool=50)`; defaults return the same classes as before |
| `scripts/run_eval.py` | `--rerank`, `--rerank-pool`, `--rerank-model`, `--rerank-device`, `--rerank-batch-size`, `--rerank-max-length` |
| `scripts/query_book.py` | `--rerank`, `--rerank-pool`, `--rerank-device` |
| `src/knowledge_base/evaluation/runner.py` | records `rerank_score` / `pre_rerank_rank` per retrieved chunk when present (nothing for baselines) |
| `scripts/rerank_report.py` | NEW: read-only three-way comparison, h02 / h07 shown explicitly |
| `tests/test_reranker.py`, `tests/test_rerank_report.py` | NEW: mocked, no weights / internet / keys |

Not touched: Answerer, prompts, `min_score`, abstention, citations, benchmarks, existing results, `pyproject.toml`, `uv.lock`.

## Behaviour guarantees
* `score` stays the true cosine (the Answerer's gate and the report's abstention numbers keep their meaning). `rerank_score` is an extra field used only for ordering.
* Chunks come only from the inner retriever: room isolation, chunk ids, file/page/topic metadata and text are untouched.
* Deterministic: ties by `rerank_score`, then candidate rank, then `chunk_id`.
* Pool = top 50 of the inner retriever's output (for hybrid: the fused list), `max(pool, k)`.

## Fair comparison
The store (`data/vector_store`, 625 chunks) and the strict held-out benchmark (`eval/benchmark/hands_on_llm_heldout_v1.jsonl`, sha256 prefix `6b0e57f23ff7`) are tracked in git, so Colab uses exactly the same candidates source as your baseline. The earlier held-out result folders (`eval/results/heldout_*`) are **not** in git; the baselines are therefore re-run (retrieval-only, minutes). Compare against your local numbers in `eval/analysis/heldout_final_comparison.md` as a sanity check (Hybrid Hit@1/3/5 = 64.3 / 90.5 / 97.6 %, MRR@5 0.772). Small differences can appear from GPU vs CPU float arithmetic.

## Commands (new output directories only; `run_eval.py` refuses an existing `--out-dir`)
PowerShell (CPU, slow but works):
```powershell
$B = "eval/benchmark/hands_on_llm_heldout_v1.jsonl"
uv run python scripts/run_eval.py --benchmark $B --retrieval-mode dense  --out-dir eval/results_rerank/heldout_dense
uv run python scripts/run_eval.py --benchmark $B --retrieval-mode hybrid --out-dir eval/results_rerank/heldout_hybrid
uv run python scripts/run_eval.py --benchmark $B --retrieval-mode hybrid --rerank --out-dir eval/results_rerank/heldout_hybrid_rerank
uv run python scripts/rerank_report.py --dense eval/results_rerank/heldout_dense/results.json --hybrid eval/results_rerank/heldout_hybrid/results.json --hybrid-rerank eval/results_rerank/heldout_hybrid_rerank/results.json --out eval/analysis_rerank/rerank_comparison.md
```
Colab/GPU: add `--device cuda` to each `run_eval.py` call (embedder); the reranker uses `--rerank-device auto`. Use `--exclude-non-content` on all three or none; your existing hybrid baseline decides (the held-out comparison reports both).

## Answer-level evaluation (optional, NOT run automatically)
Adds API calls and cost. Only after retrieval-only results are reviewed:
```powershell
$env:GEMINI_API_KEY = "..."
uv run python scripts/run_eval.py --benchmark $B --retrieval-mode hybrid --rerank --llm-provider gemini --llm-model <model> --llm-min-interval 6 --out-dir eval/results_rerank/heldout_hybrid_rerank_answers
```
* API failures are recorded as errors by the existing runner (`n_errors`); they are not abstentions. Check `n_errors` before reading any abstention rate.
* Metrics in these reports (Hit@k, MRR, citation scores, key-fact coverage, the optional LLM judge) are **this project's custom metrics**. They are not official RAGAS metrics. Official RAGAS (`ragas==0.4.3`, `scripts/rescore_run.py`) needs the separate `ragas` dependency group and is unchanged.

## Compatibility
* `pyproject.toml` still says `requires-python >=3.11,<3.12` and `uv.lock` is for 3.11; both untouched. On Colab (3.13) do not `pip install -e .`; use `requirements-colab.txt` + `PYTHONPATH=src` (the notebook does this).
* Wheels for Python 3.13 exist (checked on PyPI metadata): numpy 2.5.x, faiss-cpu 1.15.x (abi3), pymupdf 1.28.x; sentence-transformers 6.1.0 is pure Python and requires `transformers>=5,<6`, `huggingface-hub>=1.3`, `torch>=2.2`. Colab's preinstalled `transformers` may be older; pip will upgrade it, which can need a runtime restart (the notebook says when).
* The legacy ragas group (`ragas==0.4.3`, `langchain-community`) was not checked on 3.13 and is not needed for retrieval-only runs.
* Memory: the reranker is ~568M parameters; fp32 weights ~2.3 GB download. Plus BGE-M3 on the same T4 (14.5 GB) is fine for batch 16 / max length 512.

## Limitations
* The real reranker was NOT run in the build environment (no model download possible). Unit tests mock it. Expected behaviour on the actual model, including h02 and h07, is unmeasured; no improvement is claimed.
* Reranking changes order only, so out-of-scope questions are unaffected by it; the cosine gate still lets all 12 through at 0.35 (existing finding).
* 42 answerable questions: one question is 2.4 points.

## Diagnosing refusals (`scripts/diagnose_answers.py`, `colab/diagnose_answers.ipynb`)
Records the full provider response (content, reasoning, `finish_reason`, usage) for a few questions and classifies why the Answerer refused (`http_error` / `empty` / `sentinel_exact` / `sentinel_substring_only` / `no_valid_marker` / `accepted`). It does not modify the Answerer. `--dry-run` makes no LLM call; `--max-calls` (default 6) refuses to start over budget; the output directory must be new; the API key is never written. HTTP errors are errors, not abstentions.

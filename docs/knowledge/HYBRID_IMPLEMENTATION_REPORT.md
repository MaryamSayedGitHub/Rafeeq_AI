# Hybrid Search: implementation report

Nothing was committed, pushed or merged; the remote was not touched (the work was done on a separate copy with `origin` removed).
Dense retrieval is still the default and its code path is unchanged.

## What was NOT possible here

* BGE-M3 could not be downloaded (no Hugging Face access), so **no real hybrid run at the default candidate pool (50) was made** and no new
  query embeddings exist. The hybrid numbers below come from an offline **replay** (see "Evaluation") and must be confirmed with the
  four `run_eval.py` commands in `README.md`.
* Real `pytest`, real `faiss`, `fitz` (PyMuPDF) and `sentence_transformers` were not installed in this environment. Tests were executed with a small
  stand-in test runner and a numpy-based FAISS stand-in (sandbox only, **not shipped**). Run `uv run pytest` on your machine to confirm.

## Changed files

| file | change |
|---|---|
| `src/knowledge_base/retrieval/bm25.py` | NEW: tokenizer + BM25 Okapi (k1 1.5, b 0.75) |
| `src/knowledge_base/retrieval/hybrid_retriever.py` | NEW: `HybridRetriever`, `reciprocal_rank_fusion` (k 60, equal weights, pool 50) |
| `src/knowledge_base/retrieval/factory.py` | NEW: `build_retriever(..., mode="dense"\|"hybrid")`, default dense |
| `src/knowledge_base/retrieval/__init__.py` | exports added (existing exports kept) |
| `src/knowledge_base/vector_store/faiss_store.py` | additive: `room_chunks()`, `similarities()` (exact cosine of given chunks) |
| `src/knowledge_base/evaluation/runner.py` | audit fields (`fusion_score`, `dense_rank`, `lexical_rank`, `lexical_score`), `max_score`; sweep uses `max_score` (old runs: same as `top1_score`) |
| `scripts/run_eval.py` | `--retrieval-mode`, `--rrf-k`, `--candidate-pool`; refuses to overwrite an existing `results.json` |
| `scripts/query_book.py` | `--retrieval-mode`, `--exclude-non-content` |
| `scripts/replay_hybrid.py` | NEW: offline replay of a stored dense run |
| `tests/test_bm25.py`, `test_hybrid_retriever.py`, `test_run_eval_hybrid.py`, `test_replay_hybrid.py` | NEW |
| `tests/test_faiss_store.py`, `tests/test_eval_runner.py` | extended |
| `eval/results/replay_hybrid_baseline/`, `replay_hybrid_exclude/` | NEW result dirs (all older result dirs untouched) |
| `eval/analysis/*.md`, `eval/audit/q01_q21_audit.md` | NEW reports |
| `README.md` (was empty), `eval/EVALUATION.md` | usage sections |

The embedding model, chunking, `min_score` (0.35) and the vector store were not changed; the benchmark file was not changed.

## Design decisions

* `HybridRetriever` is a separate class with the same `retrieve(room_id, query, k)` signature; `Retriever` is untouched.
* `score` = true cosine (BM25-only candidates get their exact cosine from the store, via the stored vectors); ranking uses `fusion_score`.
  The Answerer's cosine gate therefore keeps its meaning, and the evaluation's gate/sweep use `max_score` because fusion can put a lower-cosine chunk first.
* Index-page exclusion filters both candidate lists before ranks are assigned. Room isolation is enforced for both lists and fails loudly.
* Fixed values (k 60, equal weights, pool 50, BM25 defaults) were not tuned on the benchmark.

## Test results

Executed with stand-ins (not real pytest/FAISS), 23 unit-test files (the integration folder was not run):

| status | files |
|---|---|
| passed (all tests in the file) | test_answerer 39, test_bm25 9, test_chunker 48, test_compare_runs 7, test_content_filter 10, test_eval_dataset_ground_truth 25, test_eval_metrics 29, test_eval_runner 40, test_faiss_store 88, test_hybrid_retriever 18, test_ingest_query_scripts 41, test_replay_hybrid 6, test_retriever 30, test_retriever_exclusion 12, test_run_eval_exclusion 2, test_run_eval_hybrid 9, test_run_eval_script 14, test_scanned_pages 20, test_store_persistence 28, test_topic_manager 82 |
| failed because a dependency is missing (identical on the untouched original code) | test_embedder: 11 failed (`sentence_transformers` missing), 13 passed; test_pipeline: 2 failed (`fitz` missing: `test_ingest_a_real_small_pdf`, `test_the_real_book_can_be_ingested`), 24 passed |
| could not run | test_parser (`fitz` missing); `tests/integration/*` (need BGE-M3) |
| failing because of the new code | none observed |

The 20 passing files were last run before the two final tests were added to test_bm25 / test_hybrid_retriever; those two files were re-run afterwards and pass (counts above).
Mutation check: 15 deliberate bugs (score = fusion, wrong k, no tie-break, exclusion off, room checks off, BM25 formula/parameters, default mode, sweep field, replay tie-break,
overwrite guard) were applied one at a time; every one is caught by at least one test (one mutant, `s > 0` -> `s >= 0` in `BM25Index.top`, is equivalent because zero scores never appear).
All changed files compile under Python 3.11.

## Evaluation (same benchmark `hands_on_llm_v1.jsonl`, v1 labels, 30 answerable questions)

Replay, candidate pool 5 (dense ranks from the stored run, BM25 recomputed from the stored chunk text with the shipped code; equivalence to a real `HybridRetriever(candidate_pool=5)` is tested):

| run | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 |
|---|---|---|---|---|---|
| dense (`baseline_again`) | 0.567 | 0.733 | 0.833 | 0.658 | 0.661 |
| hybrid replay | 0.567 | 0.700 | 0.800 | 0.655 | 0.639 |
| dense + exclusion (`exclude_index`) | 0.567 | 0.800 | 0.833 | 0.669 | 0.661 |
| hybrid replay + exclusion | 0.533 | 0.733 | 0.833 | 0.649 | 0.672 |

Per question (first relevant rank; details in `eval/analysis/`):

* vs dense: improved q06 (2->1), q18 (4->2), q26 (2->1), q28 (4->1); regressed q02 (1->2), q03 (1->2), q05 (3->4), q17 (3->5), q23 (1->2), q25 (3->5), q30 (4->miss).
* vs dense + exclusion: improved q06 (2->1), q18 (3->1), q28 (4->1); regressed q03 (1->2), q04 (1->2), q05 (3->5), q17 (3->5), q23 (1->2), q29 (1->2), q30 (2->4).
* Dense vs dense + exclusion (existing runs): improved q18, q30; no regressions.

Reading: at pool 5 hybrid does **not** beat dense on this benchmark (slightly worse Hit@3/Hit@5/MRR; 4 or 3 questions better, 7 worse). One question moves a Hit rate by 3.3 points, so
these differences are small, but there is no evidence of a gain. The replay cannot show what a pool of 50 would do, and it says nothing about the abstention gate; run the four commands in the README.
A BM25-only evaluation on the stored text gave Hit@1/3/5 = 0.23/0.50/0.53, so lexical matching is weak for this English book with many Arabic questions.

q01 and q21 stay misses in every run (see `eval/audit/q01_q21_audit.md`): the v1 label (p135) is narrow, and hybrid additionally pulls index pages 592/539 into the top 5 unless exclusion is on.

## Known limitations

* No real run at pool 50; no hybrid abstention-gate numbers; RAG answers and judge not run.
* Replay is exact only for pool = stored depth (5).
* The index-page detector misses some index-like pages (e.g. p591), which BM25 can surface.
* BM25 tokenisation is simple (plural-s only); Arabic questions match the English book only through Latin technical terms.
* The lexicon is held in memory per room and rebuilt when the store size changes (not when a chunk is replaced at equal size).
* 30 answerable questions: differences of one question are within noise.

## Addendum: real runs with BGE-M3 (run by the user on 2026-10-10)

Three real runs were added under `eval/results/`: `dense_v2`, `dense_exclude_v2`, `hybrid_exclude` (same benchmark file, hash c872d23dcc16 = CRLF variant of the same content).
The plain `hybrid` run (no exclusion) was supplied afterwards (`eval/results/hybrid`).

| run | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 |
|---|---|---|---|---|---|
| dense_v2 | 0.567 | 0.733 | 0.833 | 0.658 | 0.661 |
| dense_exclude_v2 | 0.567 | 0.800 | 0.833 | 0.669 | 0.661 |
| hybrid (pool 50) | 0.467 | 0.667 | 0.700 | 0.569 | 0.556 |
| hybrid_exclude (pool 50) | 0.533 | 0.700 | 0.767 | 0.626 | 0.633 |

* `dense_v2` retrieves exactly the same chunks, in the same order, as the older `baseline_again`: the dense default is unchanged and reproducible.
* Real hybrid + exclusion vs dense + exclusion (`eval/analysis/dense_exclude_v2_vs_hybrid_exclude.md`): improved q06 (2->1), q18 (3->1), q28 (4->1); regressed q03 (1->2), q04 (1->4), q05 (3->5), q17 (3->miss), q23 (1->2), q29 (1->miss). Net: worse on this benchmark.
* The pool-50 result is worse than the pool-5 replay (Hit@5 0.767 vs 0.833): a larger lexical pool lets more BM25-only pages displace correct dense pages (q17, q29 fall out of the top 5).
* Abstention gate: the best retrieved cosine for out-of-scope o03 dropped from 0.362 to 0.305, so at min_score 0.35 hybrid rejects it (out-of-scope pass 0.8 vs 1.0 for dense) while answerable pass stays 1.0. This is a side effect of which chunks reach the top 5, not a designed improvement, and rests on one question.

* Real hybrid (no exclusion) vs dense (`eval/analysis/dense_v2_vs_hybrid.md`): improved q06 (2->1), q18 (4->2), q28 (4->1); regressed q02, q03, q23 (rank 1->2) and q04, q05, q17, q29 (relevant page leaves the top 5). Net: clearly worse (Hit@5 0.700 vs 0.833).
* Cause, measured on this run: 24 of the 150 top-5 slots for answerable questions (16%) are index-like pages that BM25 rewards for matching many topic words. Exclusion removes most of that damage (Hit@5 0.700 -> 0.767) but hybrid is still below dense.
* Conclusion: on this benchmark hybrid (RRF k=60, equal weights) does not improve retrieval and dense remains the right default. The code is kept as an opt-in mode.

## Addendum 2: index-page detector and labels

* Detector audit (`content_filter.py`, unchanged): of the 62 index pages (537-598 region) it flags 57; the unflagged index-like pages are p565 and p591 (borderline comma / function-word share). In the real runs, unflagged index pages occupy 3 of 150 top-5 slots in hybrid+exclusion and 1 in dense+exclusion, and none of the hybrid regressions (q04, q17, q29) involve them. Loosening the thresholds would therefore not fix hybrid, so it was NOT changed.
* Labels: `hands_on_llm_v1_1.jsonl`, `BENCHMARK_LABELS.md`, `scripts/rescore_run.py` and their tests (`test_benchmark_versions`, `test_rescore_run`: 11 tests pass with the stand-in runner) were added. Re-scoring under v1.1 raises all four runs but keeps the order dense+exclusion >= dense > hybrid+exclusion > hybrid.

## Addendum 3: held-out benchmark (built; NOT yet evaluated)

Requested after the dev-set analysis showed that the v1.1 gain was a label effect and that widening q01/q21 after looking at dense results is biased.

* **New files** (nothing existing was changed): `eval/benchmark/hands_on_llm_heldout_v1.jsonl` (strict, primary), `..._lenient.jsonl` (sensitivity), `...heldout_v1.sha256`, `eval/HELDOUT_PROTOCOL.md`, `eval/audit/heldout_label_audit.md`, `scripts/heldout_report.py`, `.gitattributes` (keeps the benchmark bytes, and so its sha256, from being rewritten by line-ending conversion), `tests/test_heldout_benchmark.py` (7), `tests/test_heldout_report.py` (12).
* **Composition:** 54 questions = 23 single-passage, 19 multi-passage (2-3 pages each), 12 out-of-scope; 42 answerable questions with expected pages, verbatim evidence quotes and expected chunk ids; 16+15+7 English and 7+4+5 Arabic. No question text and no labelled page is shared with v1 / v1.1.
* **Label audit:** labels from the book text (independent `pdftotext`), never from retrieval; every quote re-found by `validate_benchmark.py` in the stored chunk of the same page number; every quote occurs only on its labelled pages; out-of-scope terms absent from the PDF text and the store. Open issues: no printed page numbers, one chunk per page (straddling facts become multi-page labels), three items with a restating page (strict vs lenient labels), no questions from chapters 1, 2, 3 and 5, 42 answerable questions only, labeller not independent of the system under test (disclosed in the protocol).
* **Tests:** with the stand-in runner (not real pytest), every test file that can run here gives the same results as before, plus the 19 new tests passing; `test_embedder` (11 failed), `test_pipeline` (2 failed) and `test_parser` (not importable) fail/skip only for the missing `sentence_transformers` / `fitz`, exactly as on the untouched original.
* **Evaluation status:** no held-out retrieval number exists. BGE-M3 cannot run in the environment where the benchmark was built, so the four runs must be done on your machine (commands in `eval/HELDOUT_PROTOCOL.md`); `scripts/heldout_report.py` then produces the verified comparison and rejects anything that is a replay.

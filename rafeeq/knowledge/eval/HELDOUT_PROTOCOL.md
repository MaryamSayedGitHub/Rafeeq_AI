# Held-out evaluation protocol (hands_on_llm_heldout_v1)

Status: **benchmark frozen, no held-out evaluation has been run.** Everything below was written before any retrieval was run on this set.
The only things that exist for it are the benchmark files, the label audit, and the tools that will score it.

| file | role | sha256 |
|---|---|---|
| `eval/benchmark/hands_on_llm_heldout_v1.jsonl` | **the held-out benchmark (strict labels, primary)** | `6b0e57f23ff7499fd47824ef6bbcd940792ef32090c681db150e250115ed0500` |
| `eval/benchmark/hands_on_llm_heldout_v1_lenient.jsonl` | sensitivity labels (strict + 3 documented pages), rescoring only | `99e4119c0702ed5830ffcf5be47bb8d3aacd0c928d8ec029d001a39ea6a18457` |
| `eval/benchmark/hands_on_llm_heldout_v1.sha256` | the two hashes above in `sha256sum` format | |
| `eval/audit/heldout_label_audit.md` | per-question evidence, pages, remarks | |

`tests/test_heldout_benchmark.py` fails if either file changes by one byte, if the dev benchmark `hands_on_llm_v1.jsonl` (sha256 prefix `c872d23dcc16`) changes, or if the held-out set shares an id, a question text or a labelled page with v1 / v1.1.
`run_eval.py` prints the first 12 hex characters of the file it scored (`6b0e57f23ff7` for the strict file); `scripts/heldout_report.py` refuses runs whose recorded hash is different.
The existing v1 / v1.1 benchmarks and every file under `eval/results/` are untouched.

## 1. What is in the benchmark (to be reported before any evaluation)

**54 questions: 42 answerable (23 single-passage, 19 multi-passage) and 12 out-of-scope.**

| category | n | English | Arabic | notes |
|---|---|---|---|---|
| single-passage | 23 | 16 | 7 | `definition` 5, `explanation` 15, `comparison` 3; exactly one labelled page each |
| multi-passage | 19 | 15 | 4 | 2-3 labelled pages each; the facts needed are stated on different pages (adjacent pages included) |
| out-of-scope | 12 | 7 | 5 | 4 unrelated to the book (capital cities, a banana, the light bulb), 8 on-topic-sounding but absent from the book (Mixtral/MoE, vLLM/Kubernetes, GRPO, HNSW, ColBERT, RLAIF, Kubeflow, Elasticsearch/BM25F) |

* Arabic questions ask about the English book, as in v1. No Arabic question is a translation of an English question in this set.
* 61 distinct physical pages are labelled; none of them is labelled in v1 or v1.1 (no page is shared with the development set), and no question text is shared.
* Items by chapter of the first labelled page: ch 4: 3, ch 6: 5, ch 7: 1, ch 8: 6, ch 9: 1, ch 10: 8, ch 11: 5, ch 12: 13. **Chapters 1, 2, 3 and 5 have no held-out question** (see limitations).
* Each answerable item carries `expected_pages`, 1-3 verbatim `evidence_quotes` (25+ characters each), the derived `expected_chunk_ids`, a reference answer and bilingual `key_facts`. Out-of-scope items carry `absent_terms`.

## 2. Label-audit method

1. **Page mapping check.** Pages are physical PDF pages (page = PDF index + 1, the `pNNNN` in chunk ids). An *independent* extraction of the PDF (`pdftotext`, per page) was compared with the stored chunk text: every evidence quote written from the independent extraction was found by `scripts/validate_benchmark.py` in the stored chunk of exactly the same page number, for all 42 answerable items. That is the page/chunk-mapping evidence.
2. **Questions were written from the book, not from a search.** The book was read chapter by chapter in the independent extraction (outline from font sizes via `pdftohtml -xml`); candidate questions came from reading sections. No retriever, no embedding model, no BM25 and no earlier run result was used to write a question or to pick a page.
3. **Pre-registered labelling rules** (fixed before drafting):
   * **R1** a page is labelled when its body text states a fact the reference answer needs; chapter introductions and chapter summaries are never labelled.
   * **R2** captions, code comments and table cells are used only if no body text states the fact.
   * **R3** a fact restated on another page should be avoided in the question; where that was not possible the restating page is written down (item `notes`) and counted only in the *lenient* sensitivity labels.
4. **Whole-PDF string audit.** Every evidence quote was searched in the whole PDF text: for all 42 items it occurs only on the labelled pages. Additional phrase probes and a rare-term co-occurrence scan over all 598 pages were used only to *flag* pages for manual reading (`eval/audit/heldout_label_audit.md`, remarks column). Result: index pages (back-of-book index, p. 560+), pure cross-references, one bibliography line, and three real restatements (m09 -> p470, m10 -> p431, a06 -> p414). Reading-based, not ranking-based: no page was added or removed because of how a system ranks it.
5. **Out-of-scope check.** Every `absent_terms` entry is absent from the PDF text and from the stored chunks of both documents in the store (`hands_on_llm.pdf`, `instance_segmentation.pdf`); `validate_benchmark.py` confirms the latter.
6. **Frozen** (hashes above) before any held-out retrieval. The strict file is primary. The lenient file differs in exactly three items by one extra page and one extra quote each.

### Independence, stated plainly

* The person who wrote these labels (Claude, in this project) had earlier seen per-question retrieval output for the **35 development questions** (v1/v1.1) and designed the Hybrid retriever. The labeller is therefore not independent of the system under test. What limits the damage: nothing from the development runs was used (no held-out topic, page or wording was taken from them), no held-out question has been run through any retriever, labels come from the PDF text only, and everything is in the audit file for a third party to re-check.
* The question mix was fixed by category (single / multi / out-of-scope, English / Arabic) and by chapter coverage, not by guessing which questions help or hurt Hybrid. Some questions contain rare keywords (names, numbers) and some are paraphrases; the set was not tuned either way.

## 3. Unresolved page-mapping issues and limitations

* **No printed page numbers.** The book has none we can rely on; every number here is the physical PDF page. The independent extraction and the store agree on every page that matters here: the store holds pages 2-598 of `hands_on_llm.pdf` except 12 and 214 (595 pages); pages 1, 12 and 214 carry no extractable text and the extraction tool reports one extra empty trailing element after page 598; pages 4 and 596 contain only the `OceanofPDF.com` watermark. None of these is labelled, and a label can never point at a page that has no stored chunk.
* **One chunk per page.** A fact that straddles a page break (e.g. m02 p479-480, m01 p241-242, m10 p430-435) forces a multi-page label. That is deliberate (it is what chunking does to a user), but it means "multi-passage" here is *evidence on two or more pages*, not "reasoning over distant sections".
* **Restatements.** Three items have a second page that restates one sub-fact (list above). Strict labels do not count them; if strict and lenient results disagree about a conclusion, that conclusion depends on labelling and is reported as such.
* **Back-matter pages (p. 560-598)** contain index entries for many terms. They are never labelled. The `--exclude-non-content` runs remove them; the plain runs do not, which is exactly what the Dense vs Dense + exclusion comparison measures.
* **Coverage.** No question from chapters 1, 2, 3 and 5 (13 of 42 answerable items come from chapter 12). Results say nothing about those chapters.
* **Size.** 42 answerable questions: one question moves a Hit rate by 2.4 points (19 multi-passage: 5.3 points). Differences of a few points are not distinguishable from noise.
* **Book errors left as written.** p466 computes `20 * 32 = 680`; p424 labels 0.72 as the softmax result while p420 says 0.59. Questions only use statements that are correct as written (m08 asks for the number of pairs and the F1, a06 for 0.59 -> 0.72).
* **Arabic.** Arabic questions are answered from English evidence; `key_facts` list Arabic and English spellings. Retrieval is judged by page, so answer-language does not matter for the retrieval metrics.

## 4. Evaluation protocol (fixed now, run later)

Four runs, same benchmark file, same store, embedder (BAAI/bge-m3), room, `--ks 1 3 5`, answer-k and `min_score` defaults (0.35, not tuned). Hybrid runs use RRF k = 60, equal weights, candidate pool 50. Nothing is tuned on this set. Each run needs a new `--out-dir`; `run_eval.py` never overwrites.

```powershell
$B = "eval\benchmark\hands_on_llm_heldout_v1.jsonl"
Get-FileHash $B -Algorithm SHA256        # must print 6B0E57F23FF7499F...D0500

uv run python scripts/run_eval.py --benchmark $B --out-dir eval\results\heldout_dense
uv run python scripts/run_eval.py --benchmark $B --exclude-non-content --out-dir eval\results\heldout_dense_exclude
uv run python scripts/run_eval.py --benchmark $B --retrieval-mode hybrid --candidate-pool 50 --out-dir eval\results\heldout_hybrid
uv run python scripts/run_eval.py --benchmark $B --retrieval-mode hybrid --candidate-pool 50 --exclude-non-content --out-dir eval\results\heldout_hybrid_exclude

uv run python scripts/heldout_report.py `
  --dense eval\results\heldout_dense\results.json `
  --dense-exclude eval\results\heldout_dense_exclude\results.json `
  --hybrid eval\results\heldout_hybrid\results.json `
  --hybrid-exclude eval\results\heldout_hybrid_exclude\results.json `
  --out eval\analysis\heldout_final_comparison.md

# sensitivity to the three restated pages: no re-run, same stored results
uv run python scripts/rescore_run.py eval\results\heldout_dense\results.json `
  --benchmark eval\benchmark\hands_on_llm_heldout_v1.jsonl --benchmark eval\benchmark\hands_on_llm_heldout_v1_lenient.jsonl
```

**Reported:** Hit@1/3/5, MRR@5, Recall@5 for all answerable questions, for single-passage and for multi-passage questions, and Full Recall@5 (every labelled page in the top 5) for multi-passage questions; the 0.35 abstention gate (answerable passing, out-of-scope false accepts, highest out-of-scope best cosine); a per-question list of improvements and regressions for Dense -> Hybrid, Dense + exclusion -> Hybrid + exclusion, Dense -> Dense + exclusion and Hybrid -> Hybrid + exclusion with an exact sign test on the changed questions; and the first relevant rank of every question in every configuration.

**Decision rule, written before the data exists.** "Hybrid improves retrieval" may be claimed only if, within the same exclusion setting, it has more improved than regressed questions with a two-sided exact sign test p < 0.05, a higher MRR@5, and no increase in out-of-scope false accepts at the 0.35 gate. Otherwise the report says "no demonstrated improvement" and gives the numbers. The same rule applies to any claim that exclusion helps. Results on the 35 development questions do not count towards either claim.

**Rules for use.**
* The development benchmark (v1, v1.1) is for development. This set is for the final comparison only: no change to the embedding model, chunking, `min_score`, RRF constant, pool size, tokenizer or exclusion rules is made after looking at it. If something must change, the change is made, justified on the development set, and a *new* held-out set (v2) is built; this one is then no longer held out.
* Any additional run on this set (other parameters, other modes) is labelled exploratory in the report and is not part of the comparison above.
* `scripts/heldout_report.py` rejects results that are not the frozen file, not the stated configuration, not the same setup, or that come from `replay_hybrid.py`.

## 5. Verified versus simulated

* **Verified (real):** the benchmark content (quotes re-found in the stored chunks; absent terms absent from the store), the tests listed in the report, and, once you run the four commands, the retrieval metrics of those four runs.
* **Not yet existing:** any held-out retrieval number. There is none: the embedding model (BGE-M3) is not available in the environment this benchmark was built in, so nothing was run on it.
* **Simulated / offline replay:** `scripts/replay_hybrid.py` can re-rank stored depth-5 dense results but equals the real retriever only at pool 5, cannot compute the abstention gate, and needs stored results of the same benchmark. It is not used here; `heldout_report.py` rejects its output.

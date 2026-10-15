# Benchmark labels: rule, change log, open questions

Page numbers are physical PDF pages of `hands_on_llm.pdf` (598 pages; page = PDF index + 1, which is also the `pNNNN` in the chunk ids).
All quotes below were checked against the text of that PDF.

## Rule for `expected_pages`

A page is labelled when it contains a passage that directly states part of the answer to the question **as worded**.
Chapter introductions and chapter summaries restate what the body pages say; they are not labelled (otherwise every question
would also accept the summary of its chapter). A label is changed only after reading the pages, and only by widening it.
Every change is recorded here and in the item's `notes`; `tests/test_benchmark_versions.py` fails if anything else changes.

Hit@K accepts any labelled page. Recall@K / FullRecall@K require **all** labelled pages. That is right for questions that need
several passages and wrong for a definition that several pages state independently; see "Limitation" below.

## Versions

| file | what it is |
|---|---|
| `hands_on_llm_v1.jsonl` | the original 35 items (keep it unmodified; `git checkout -- eval/benchmark/hands_on_llm_v1.jsonl` restores it) |
| `hands_on_llm_v1_1.jsonl` | v1 + the three corrections below |

Results are only comparable under the same labels. To see what a label change does on its own, re-score a stored run:

```powershell
uv run python scripts/rescore_run.py eval\results\<run>\results.json `
    --benchmark eval\benchmark\hands_on_llm_v1.jsonl --benchmark eval\benchmark\hands_on_llm_v1_1.jsonl
```

For run `20261009T130509Z` (baseline retrieval, nothing re-run) the label change alone gives Hit@3 73.3% -> 76.7%, Hit@5 83.3% -> 90.0%,
MRR@5 0.658 -> 0.678. The same numbers came out of the real re-run on the changed benchmark, so the whole difference between those two
reports is the labels, not the retrieval.

## Change log

### v1 -> v1.1, 2026-10-09

| item | expected_pages | evidence in the book |
|---|---|---|
| q01, q21 (self-attention) | [135] -> [135, 141, 142] | p141 "Self-attention: Relevance scoring: ... multiplying the query vector of the current position with the keys matrix"; p142 "Self-attention: Combining information: ... Summing up those resulting vectors produces the output of this attention step". Both are the book's own sections on self-attention. |
| q08 (QLoRA memory) | [501, 505] -> [501, 502, 505] | The question asks how QLoRA reduces **memory**. p502: "A 4-bit representation significantly reduces the memory requirements of the LLM during training." Neither p501 (blockwise quantization) nor p505 (the bitsandbytes code) says that. |

How these were decided: the q01/q21 widening was **prompted by** the observation that run `20261009T113241Z` returned p141 for them (an earlier
note claiming it was made "not by looking at retrieval scores" overstated this); the pages were then read and the rule above applied.
The q08 page was found by reading pp. 499-506; p502 is not retrieved, so this change moves no metric.

### Examined and deliberately not changed

| item | finding | decision |
|---|---|---|
| q16 (prompt -> token) | Labels [121, 122, 128] are right: they contain the tokenizer -> Transformer blocks -> LM head description. The retrieved pp. 118-120 describe the generation **loop** (one token at a time, appended to the prompt), a different reading of "path from prompt to token". p157 is the chapter summary. | Question is ambiguous, not mislabelled. Left as is; if reworded (e.g. "...through the tokenizer, the Transformer blocks and the LM head") that is a new item in v2, not an edit of v1.x. |
| q19 (vision encoder + LLM) | Labels [379, 381] (BLIP-2 / Q-Former) are right. Retrieved pp. 360-361 are the chapter introduction and pp. 395-396 the chapter summary (p396: "projecting visual features from input images to text embeddings that can be used by LLMs"). | Genuine retrieval miss under the rule above (summary/intro pages are not labelled). Left as is. |
| q08, p506 | States the effect: "~1 GB VRAM compared to the ~4 GB of VRAM it would need without quantization". It qualifies under the rule. | **Not added.** It was noticed because it is retrieved (rank 2); adding it makes q08 a hit and lifts Hit@3 to 80.0%, Hit@5 to 93.3%, MRR@5 to 0.694 on the stored run. If you add it, say so in the change log: it is a judgement call that coincides with the retriever's output. |
| q08, p500 | States the principle (fewer bits -> lower memory). | Not added (same reasoning; also makes the page set larger than a single question should need). |
| q01/q21, p136 | Figure 3-14 caption: "The self-attention layer incorporates relevant information from previous positions that help process the current token." Qualifies. | Not added: with all-pages-required recall a definition would need four pages. |

## Limitation

One list per item cannot say "any of these pages is enough" (a definition stated on several pages) and "all of these are needed"
(a multi-passage question) at once. Widening `expected_pages` for the first case lowers Recall/FullRecall; leaving it narrow turns valid
retrievals into misses. The clean fix is a second, optional list (`acceptable_pages`: counts for Hit/MRR/Precision, not required for
Recall). That is a schema change (dataset, validator, scorer, report, tests) and has not been made.

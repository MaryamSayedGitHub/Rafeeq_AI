# Audit of q01 and q21 (self-attention)

Benchmark file audited: `eval/benchmark/hands_on_llm_v1.jsonl` (NOT modified). Evidence text is quoted from
`data/vector_store/metadata.json` (the stored chunk text, one chunk per physical PDF page).

| id | question | expected_pages (v1) |
|---|---|---|
| q01 | What is self-attention in a Transformer? | [135] |
| q21 | Arabic version of the same question | [135] |

## What the stored pages say

| page | content (stored text) | states a self-attention definition? |
|---|---|---|
| 135 | "Attention is a mechanism that helps the model incorporate context as it's processing a specific token." (general attention, "The attention layer at a glance") | yes, attention in general (the v1 label) |
| 136 | Fig. 3-14 caption: "The self-attention layer incorporates relevant information from previous positions that help process the current token." | yes, the only sentence on these pages that says "self-attention layer" in a definitional way |
| 141 | "Self-attention: Relevance scoring ... multiplying the query vector of the current position with the keys matrix." | yes, first half of the mechanism |
| 142 | "Self-attention: Combining information ... Summing up those resulting vectors produces the output of this attention step." | yes, second half of the mechanism |
| 38, 39 | Chapter 1: the encoder block "consists of two parts, self-attention and a feedforward neural network"; self-attention "can attend to different positions within a single sequence" | yes, an introductory description |
| 138 | multi-head attention | no (a different topic) |
| 144 | "More Efficient Attention" (sparse/sliding window attention) | no (the top dense hit for both questions) |
| 592, 539 | index pages (flagged by the non-content detector) | no (BM25 matches the word "attention" in index entries) |

## What each run retrieved (top 5 pages, cosine in brackets)

| run | q01 | q21 |
|---|---|---|
| dense (`baseline_again`) | 144 (.620), 38 (.618), 141 (.604), 138 (.594), 150 (.581) | 144 (.625), 138 (.624), 38 (.617), 141 (.600), 146 (.592) |
| dense + exclusion (`exclude_index`) | identical to dense | identical to dense |
| hybrid replay, pool 5 | 38, 144, **592, 539** (index pages), 141 | 38, 144, **592**, 138, **539** |
| hybrid replay + exclusion, pool 5 | 38, 141, 150, 144, **39** | 38, 144, 138, **39**, 156 |

Facts that follow from the table:

* Neither question is a hit in any run under the v1 label: the only labelled page (135) is never retrieved. BM25 alone ranks p135 at
  position 24 (q01) / 38 (q21) over the whole room (19 / 29 with exclusion), far outside any candidate pool.
* Under dense retrieval the cosines of the top five pages differ by 0.04 (q01) or 0.03 (q21); the order among them is within noise.
* The dense and hybrid results are mostly valid self-attention sources: of the pages that state self-attention (136, 141, 142, 38, 39) dense retrieval returns 141
  and 38 (and 138 for q21); p141 is at rank 3 (q01) / 4 (q21).
* Hybrid does **not** fix these two questions. It moves p38 to rank 1, and without exclusion it lets index pages 592/539 into the top 5 (they
  contain the query word "attention" in many index entries). Exclusion removes them, as designed. One further index-like page (p591) is
  not flagged by the detector and still reaches BM25 rank 5 for q21 with exclusion; it does not reach the fused top 5 here, but it shows the
  detector is not perfect.
* The Arabic question q21 shares only the Latin tokens "self", "attention" and "Transformer" with the English book, so its BM25 ranking is driven by those three words.

## Decision

* The v1 label for q01 and q21 is too narrow: it accepts one general page while the book defines self-attention on pages 136, 141 and 142 too.
  This is mainly a labelling issue, not a retrieval defect (a judgement from reading the stored pages above, not a measured result).
* Nothing in this delivery changes the labels. Every number in the comparison reports uses v1, so the dense baselines are compared on the same labels.
* The widened labels now ship in this repository as `eval/benchmark/hands_on_llm_v1_1.jsonl` (rule and change log: `eval/BENCHMARK_LABELS.md`; re-scoring tool: `scripts/rescore_run.py`).
  They were not used for the main comparison. Re-scoring the four real runs under v1.1 (no retrieval re-run; `eval/analysis/rescore_*.md`) lifts every run by a similar amount
  (Hit@5: dense 83.3 -> 90.0, dense+exclusion 83.3 -> 90.0, hybrid 70.0 -> 76.7, hybrid+exclusion 76.7 -> 83.3), so the ranking of the four runs does not change.
  Note that the widening of q01/q21 was prompted by seeing which pages dense retrieval returned (see `BENCHMARK_LABELS.md`), so v1.1 numbers must not be presented as an independent held-out result.

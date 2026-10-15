# Re-score of eval/results/hybrid/results.json

Retrieval is taken from the stored run; only the ground truth differs between columns.

**all answerable** (n = 30, 30)

| metric | hands_on_llm_v1.jsonl | hands_on_llm_v1_1.jsonl |
|---|---|---|
| hit@1 | 46.7% | 46.7% |
| hit@3 | 66.7% | 73.3% |
| hit@5 | 70.0% | 76.7% |
| mrr@5 | 0.569 | 0.597 |
| recall@5 | 55.6% | 57.8% |
| full_recall@5 | 43.3% | 43.3% |

**multi_passage** (n = 7, 7)

| metric | hands_on_llm_v1.jsonl | hands_on_llm_v1_1.jsonl |
|---|---|---|
| hit@1 | 14.3% | 14.3% |
| hit@3 | 28.6% | 28.6% |
| hit@5 | 42.9% | 42.9% |
| mrr@5 | 0.250 | 0.250 |
| recall@5 | 19.0% | 19.0% |
| full_recall@5 | 0.0% | 0.0% |

## Questions that score differently under hands_on_llm_v1_1.jsonl than under hands_on_llm_v1.jsonl

| id | expected pages (ref -> this) | first relevant rank | recall@5 | retrieved pages (top 5) |
|---|---|---|---|---|
| q01 | [135] -> [135, 141, 142] | none -> 2 | 0.00 -> 0.33 | 38 141 592 150 144 |
| q08 | [501, 505] -> [501, 502, 505] | none -> none | 0.00 -> 0.00 | 533 509 506 510 503 |
| q21 | [135] -> [135, 141, 142] | none -> 3 | 0.00 -> 0.33 | 38 539 141 144 592 |

Small samples: with ~30 questions one question moves a Hit rate by 3.3 points. A label change that is followed by a retrieval run does not show that the system improved; compare runs only under the same labels.

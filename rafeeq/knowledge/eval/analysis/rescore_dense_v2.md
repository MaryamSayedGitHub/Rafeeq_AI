# Re-score of eval/results/dense_v2/results.json

Retrieval is taken from the stored run; only the ground truth differs between columns.

**all answerable** (n = 30, 30)

| metric | hands_on_llm_v1.jsonl | hands_on_llm_v1_1.jsonl |
|---|---|---|
| hit@1 | 56.7% | 56.7% |
| hit@3 | 73.3% | 76.7% |
| hit@5 | 83.3% | 90.0% |
| mrr@5 | 0.658 | 0.678 |
| recall@5 | 66.1% | 68.3% |
| full_recall@5 | 50.0% | 50.0% |

**multi_passage** (n = 7, 7)

| metric | hands_on_llm_v1.jsonl | hands_on_llm_v1_1.jsonl |
|---|---|---|
| hit@1 | 28.6% | 28.6% |
| hit@3 | 42.9% | 42.9% |
| hit@5 | 71.4% | 71.4% |
| mrr@5 | 0.405 | 0.405 |
| recall@5 | 33.3% | 33.3% |
| full_recall@5 | 0.0% | 0.0% |

## Questions that score differently under hands_on_llm_v1_1.jsonl than under hands_on_llm_v1.jsonl

| id | expected pages (ref -> this) | first relevant rank | recall@5 | retrieved pages (top 5) |
|---|---|---|---|---|
| q01 | [135] -> [135, 141, 142] | none -> 3 | 0.00 -> 0.33 | 144 38 141 138 150 |
| q08 | [501, 505] -> [501, 502, 505] | none -> none | 0.00 -> 0.00 | 533 506 510 499 509 |
| q21 | [135] -> [135, 141, 142] | none -> 4 | 0.00 -> 0.33 | 144 138 38 141 146 |

Small samples: with ~30 questions one question moves a Hit rate by 3.3 points. A label change that is followed by a retrieval run does not show that the system improved; compare runs only under the same labels.

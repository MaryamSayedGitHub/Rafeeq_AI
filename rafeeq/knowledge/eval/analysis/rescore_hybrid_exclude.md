# Re-score of eval/results/hybrid_exclude/results.json

Retrieval is taken from the stored run; only the ground truth differs between columns.

**all answerable** (n = 30, 30)

| metric | hands_on_llm_v1.jsonl | hands_on_llm_v1_1.jsonl |
|---|---|---|
| hit@1 | 53.3% | 53.3% |
| hit@3 | 70.0% | 76.7% |
| hit@5 | 76.7% | 83.3% |
| mrr@5 | 0.626 | 0.659 |
| recall@5 | 63.3% | 65.6% |
| full_recall@5 | 50.0% | 50.0% |

**multi_passage** (n = 7, 7)

| metric | hands_on_llm_v1.jsonl | hands_on_llm_v1_1.jsonl |
|---|---|---|
| hit@1 | 28.6% | 28.6% |
| hit@3 | 42.9% | 42.9% |
| hit@5 | 42.9% | 42.9% |
| mrr@5 | 0.357 | 0.357 |
| recall@5 | 23.8% | 23.8% |
| full_recall@5 | 0.0% | 0.0% |

## Questions that score differently under hands_on_llm_v1_1.jsonl than under hands_on_llm_v1.jsonl

| id | expected pages (ref -> this) | first relevant rank | recall@5 | retrieved pages (top 5) |
|---|---|---|---|---|
| q01 | [135] -> [135, 141, 142] | none -> 2 | 0.00 -> 0.33 | 38 141 150 144 39 |
| q08 | [501, 505] -> [501, 502, 505] | none -> none | 0.00 -> 0.00 | 533 509 506 510 503 |
| q21 | [135] -> [135, 141, 142] | none -> 2 | 0.00 -> 0.33 | 38 141 144 39 40 |

Small samples: with ~30 questions one question moves a Hit rate by 3.3 points. A label change that is followed by a retrieval run does not show that the system improved; compare runs only under the same labels.

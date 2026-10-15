# Retrieval comparison

- before: dense, index-like pages excluded (run 2026-10-09 16:21:18)
- after: hybrid REPLAY (dense ranks from stored run, BM25 recomputed; RRF k=60, equal weights, candidate pool 5), index-like pages excluded (run 2026-10-10 11:49:06)

> WARNING: the two runs report different embedders

## Metrics

Small samples: with ~30 questions one question moves a Hit rate by 3.3 points. Read the lists below, not only the deltas.

**all answerable** (n=30)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 23.3% | 16.7% | -6.7 pp |
| full_recall@3 | 43.3% | 36.7% | -6.7 pp |
| full_recall@5 | 50.0% | 50.0% | 0 |
| hit@1 | 56.7% | 53.3% | -3.3 pp |
| hit@3 | 80.0% | 73.3% | -6.7 pp |
| hit@5 | 83.3% | 83.3% | 0 |
| mrr@5 | 0.669 | 0.649 | -0.020 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 56.7% | 53.3% | -3.3 pp |
| precision@3 | 34.4% | 30.0% | -4.4 pp |
| precision@5 | 23.3% | 24.0% | +0.7 pp |
| recall@1 | 38.3% | 32.8% | -5.6 pp |
| recall@3 | 59.4% | 53.9% | -5.6 pp |
| recall@5 | 66.1% | 67.2% | +1.1 pp |

**type: comparison** (n=6)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 0.0% | 16.7% | +16.7 pp |
| full_recall@3 | 66.7% | 66.7% | 0 |
| full_recall@5 | 83.3% | 83.3% | 0 |
| hit@1 | 83.3% | 100.0% | +16.7 pp |
| hit@3 | 83.3% | 100.0% | +16.7 pp |
| hit@5 | 100.0% | 100.0% | 0 |
| mrr@5 | 0.875 | 1.000 | +0.125 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 83.3% | 100.0% | +16.7 pp |
| precision@3 | 50.0% | 50.0% | 0 |
| precision@5 | 33.3% | 33.3% | 0 |
| recall@1 | 41.7% | 58.3% | +16.7 pp |
| recall@3 | 75.0% | 83.3% | +8.3 pp |
| recall@5 | 91.7% | 91.7% | 0 |

**type: definition** (n=8)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 62.5% | 25.0% | -37.5 pp |
| full_recall@3 | 75.0% | 62.5% | -12.5 pp |
| full_recall@5 | 75.0% | 75.0% | 0 |
| hit@1 | 62.5% | 25.0% | -37.5 pp |
| hit@3 | 75.0% | 62.5% | -12.5 pp |
| hit@5 | 75.0% | 75.0% | 0 |
| mrr@5 | 0.667 | 0.463 | -0.204 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 62.5% | 25.0% | -37.5 pp |
| precision@3 | 25.0% | 20.8% | -4.2 pp |
| precision@5 | 15.0% | 15.0% | 0 |
| recall@1 | 62.5% | 25.0% | -37.5 pp |
| recall@3 | 75.0% | 62.5% | -12.5 pp |
| recall@5 | 75.0% | 75.0% | 0 |

**type: explanation** (n=9)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 22.2% | 22.2% | 0 |
| full_recall@3 | 33.3% | 22.2% | -11.1 pp |
| full_recall@5 | 44.4% | 44.4% | 0 |
| hit@1 | 55.6% | 66.7% | +11.1 pp |
| hit@3 | 88.9% | 88.9% | 0 |
| hit@5 | 88.9% | 88.9% | 0 |
| mrr@5 | 0.704 | 0.759 | +0.056 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 55.6% | 66.7% | +11.1 pp |
| precision@3 | 37.0% | 33.3% | -3.7 pp |
| precision@5 | 26.7% | 26.7% | 0 |
| recall@1 | 37.0% | 40.7% | +3.7 pp |
| recall@3 | 59.3% | 53.7% | -5.6 pp |
| recall@5 | 66.7% | 66.7% | 0 |

**type: multi_passage** (n=7)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 0.0% | 0.0% | 0 |
| full_recall@3 | 0.0% | 0.0% | 0 |
| full_recall@5 | 0.0% | 0.0% | 0 |
| hit@1 | 28.6% | 28.6% | 0 |
| hit@3 | 71.4% | 42.9% | -28.6 pp |
| hit@5 | 71.4% | 71.4% | 0 |
| mrr@5 | 0.452 | 0.421 | -0.031 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 28.6% | 28.6% | 0 |
| precision@3 | 28.6% | 19.0% | -9.5 pp |
| precision@5 | 20.0% | 22.9% | +2.9 pp |
| recall@1 | 9.5% | 9.5% | 0 |
| recall@3 | 28.6% | 19.0% | -9.5 pp |
| recall@5 | 33.3% | 38.1% | +4.8 pp |

**language: ar** (n=10)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 30.0% | 30.0% | 0 |
| full_recall@3 | 50.0% | 50.0% | 0 |
| full_recall@5 | 60.0% | 60.0% | 0 |
| hit@1 | 50.0% | 40.0% | -10.0 pp |
| hit@3 | 80.0% | 80.0% | 0 |
| hit@5 | 90.0% | 90.0% | 0 |
| mrr@5 | 0.658 | 0.608 | -0.050 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 50.0% | 40.0% | -10.0 pp |
| precision@3 | 33.3% | 30.0% | -3.3 pp |
| precision@5 | 22.0% | 24.0% | +2.0 pp |
| recall@1 | 38.3% | 35.0% | -3.3 pp |
| recall@3 | 60.0% | 61.7% | +1.7 pp |
| recall@5 | 70.0% | 73.3% | +3.3 pp |

**language: en** (n=20)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 20.0% | 10.0% | -10.0 pp |
| full_recall@3 | 40.0% | 30.0% | -10.0 pp |
| full_recall@5 | 45.0% | 45.0% | 0 |
| hit@1 | 60.0% | 60.0% | 0 |
| hit@3 | 80.0% | 70.0% | -10.0 pp |
| hit@5 | 80.0% | 80.0% | 0 |
| mrr@5 | 0.675 | 0.670 | -0.005 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 60.0% | 60.0% | 0 |
| precision@3 | 35.0% | 30.0% | -5.0 pp |
| precision@5 | 24.0% | 24.0% | 0 |
| recall@1 | 38.3% | 31.7% | -6.7 pp |
| recall@3 | 59.2% | 50.0% | -9.2 pp |
| recall@5 | 64.2% | 64.2% | 0 |

## Answerable questions

Improved: **3**, regressed: **7**, unchanged rank: 20

### Regressions (rank of first relevant page got worse)

| id | type | lang | first relevant rank (before -> after) | pages before | pages after |
|---|---|---|---|---|---|
| q03 | definition | en | 1 -> 2 | 497 533 499 507 509 | 533 497 492 499 507 |
| q04 | definition | en | 1 -> 2 | 101 400 173 399 366 | 399 101 375 400 173 |
| q05 | definition | en | 3 -> 5 | 212 218 206 210 234 | 212 218 210 229 206 |
| q17 | multi_passage | en | 3 -> 5 | 193 32 320 234 366 | 32 193 13 233 320 |
| q23 | definition | ar | 1 -> 2 | 497 533 509 506 529 | 506 497 533 530 509 |
| q29 | multi_passage | ar | 1 -> 2 | 320 32 31 400 108 | 31 320 375 32 173 |
| q30 | multi_passage | ar | 2 -> 4 | 492 490 499 280 448 | 280 492 14 490 244 |

### Improvements

| id | type | lang | first relevant rank (before -> after) | pages before | pages after |
|---|---|---|---|---|---|
| q06 | explanation | en | 2 -> 1 | 218 206 234 207 228 | 206 234 218 204 207 |
| q18 | multi_passage | en | 3 -> 1 | 533 280 244 490 236 | 490 280 533 488 244 |
| q28 | comparison | ar | 4 -> 1 | 272 269 257 255 264 | 255 272 264 269 279 |

## Out-of-scope questions (top-1 cosine score)

| id | top-1 before | top-1 after | change | top-1 page before -> after |
|---|---|---|---|---|
| o01 | 0.456 | 0.456 | +0.000 | 338 -> 338 |
| o02 | 0.401 | 0.401 | +0.000 | 338 -> 338 |
| o03 | 0.362 | 0.362 | +0.000 | 323 -> 323 |
| o04 | 0.461 | 0.461 | +0.000 | 26 -> 26 |
| o05 | 0.663 | 0.663 | +0.000 | 46 -> 46 |

Descriptive score gate (share of questions whose top-1 score is at least the threshold):

| threshold | answerable before | answerable after | out-of-scope before | out-of-scope after |
|---|---|---|---|---|

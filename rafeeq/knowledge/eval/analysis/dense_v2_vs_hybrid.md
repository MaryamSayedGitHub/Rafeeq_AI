# Retrieval comparison

- before: dense (unchanged baseline behaviour) (run 2026-10-10 19:59:57)
- after: hybrid (dense + BM25, RRF k=60, equal weights, candidate pool 50) (run 2026-10-10 20:04:24)

## Metrics

Small samples: with ~30 questions one question moves a Hit rate by 3.3 points. Read the lists below, not only the deltas.

**all answerable** (n=30)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 23.3% | 13.3% | -10.0 pp |
| full_recall@3 | 43.3% | 36.7% | -6.7 pp |
| full_recall@5 | 50.0% | 43.3% | -6.7 pp |
| hit@1 | 56.7% | 46.7% | -10.0 pp |
| hit@3 | 73.3% | 66.7% | -6.7 pp |
| hit@5 | 83.3% | 70.0% | -13.3 pp |
| mrr@5 | 0.658 | 0.569 | -0.089 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 56.7% | 46.7% | -10.0 pp |
| precision@3 | 32.2% | 28.9% | -3.3 pp |
| precision@5 | 23.3% | 19.3% | -4.0 pp |
| recall@1 | 38.3% | 28.3% | -10.0 pp |
| recall@3 | 57.2% | 51.1% | -6.1 pp |
| recall@5 | 66.1% | 55.6% | -10.6 pp |

**type: comparison** (n=6)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 0.0% | 16.7% | +16.7 pp |
| full_recall@3 | 66.7% | 83.3% | +16.7 pp |
| full_recall@5 | 83.3% | 83.3% | 0 |
| hit@1 | 83.3% | 100.0% | +16.7 pp |
| hit@3 | 83.3% | 100.0% | +16.7 pp |
| hit@5 | 100.0% | 100.0% | 0 |
| mrr@5 | 0.875 | 1.000 | +0.125 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 83.3% | 100.0% | +16.7 pp |
| precision@3 | 50.0% | 55.6% | +5.6 pp |
| precision@5 | 33.3% | 33.3% | 0 |
| recall@1 | 41.7% | 58.3% | +16.7 pp |
| recall@3 | 75.0% | 91.7% | +16.7 pp |
| recall@5 | 91.7% | 91.7% | 0 |

**type: definition** (n=8)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 62.5% | 12.5% | -50.0 pp |
| full_recall@3 | 75.0% | 50.0% | -25.0 pp |
| full_recall@5 | 75.0% | 50.0% | -25.0 pp |
| hit@1 | 62.5% | 12.5% | -50.0 pp |
| hit@3 | 75.0% | 50.0% | -25.0 pp |
| hit@5 | 75.0% | 50.0% | -25.0 pp |
| mrr@5 | 0.667 | 0.312 | -0.354 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 62.5% | 12.5% | -50.0 pp |
| precision@3 | 25.0% | 16.7% | -8.3 pp |
| precision@5 | 15.0% | 10.0% | -5.0 pp |
| recall@1 | 62.5% | 12.5% | -50.0 pp |
| recall@3 | 75.0% | 50.0% | -25.0 pp |
| recall@5 | 75.0% | 50.0% | -25.0 pp |

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
| precision@5 | 26.7% | 24.4% | -2.2 pp |
| recall@1 | 37.0% | 40.7% | +3.7 pp |
| recall@3 | 59.3% | 53.7% | -5.6 pp |
| recall@5 | 66.7% | 64.8% | -1.9 pp |

**type: multi_passage** (n=7)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 0.0% | 0.0% | 0 |
| full_recall@3 | 0.0% | 0.0% | 0 |
| full_recall@5 | 0.0% | 0.0% | 0 |
| hit@1 | 28.6% | 14.3% | -14.3 pp |
| hit@3 | 42.9% | 28.6% | -14.3 pp |
| hit@5 | 71.4% | 42.9% | -28.6 pp |
| mrr@5 | 0.405 | 0.250 | -0.155 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 28.6% | 14.3% | -14.3 pp |
| precision@3 | 19.0% | 14.3% | -4.8 pp |
| precision@5 | 20.0% | 11.4% | -8.6 pp |
| recall@1 | 9.5% | 4.8% | -4.8 pp |
| recall@3 | 19.0% | 14.3% | -4.8 pp |
| recall@5 | 33.3% | 19.0% | -14.3 pp |

**language: ar** (n=10)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 30.0% | 30.0% | 0 |
| full_recall@3 | 50.0% | 50.0% | 0 |
| full_recall@5 | 60.0% | 60.0% | 0 |
| hit@1 | 50.0% | 40.0% | -10.0 pp |
| hit@3 | 70.0% | 70.0% | 0 |
| hit@5 | 90.0% | 80.0% | -10.0 pp |
| mrr@5 | 0.633 | 0.558 | -0.075 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 50.0% | 40.0% | -10.0 pp |
| precision@3 | 30.0% | 26.7% | -3.3 pp |
| precision@5 | 22.0% | 20.0% | -2.0 pp |
| recall@1 | 38.3% | 35.0% | -3.3 pp |
| recall@3 | 56.7% | 58.3% | +1.7 pp |
| recall@5 | 70.0% | 66.7% | -3.3 pp |

**language: en** (n=20)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 20.0% | 5.0% | -15.0 pp |
| full_recall@3 | 40.0% | 30.0% | -10.0 pp |
| full_recall@5 | 45.0% | 35.0% | -10.0 pp |
| hit@1 | 60.0% | 50.0% | -10.0 pp |
| hit@3 | 75.0% | 65.0% | -10.0 pp |
| hit@5 | 80.0% | 65.0% | -15.0 pp |
| mrr@5 | 0.671 | 0.575 | -0.096 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 60.0% | 50.0% | -10.0 pp |
| precision@3 | 33.3% | 30.0% | -3.3 pp |
| precision@5 | 24.0% | 19.0% | -5.0 pp |
| recall@1 | 38.3% | 25.0% | -13.3 pp |
| recall@3 | 57.5% | 47.5% | -10.0 pp |
| recall@5 | 64.2% | 50.0% | -14.2 pp |

## Answerable questions

Improved: **3**, regressed: **7**, unchanged rank: 20

### Regressions (rank of first relevant page got worse)

| id | type | lang | first relevant rank (before -> after) | pages before | pages after |
|---|---|---|---|---|---|
| q02 | definition | en | 1 -> 2 | 349 318 317 319 576 | 576 349 563 318 95 |
| q03 | definition | en | 1 -> 2 | 497 533 499 507 509 | 533 497 572 499 582 |
| q04 | definition | en | 1 -> none | 101 400 173 399 366 | 399 586 400 547 96 |
| q05 | definition | en | 3 -> none | 212 218 206 210 234 | 212 218 207 210 540 |
| q17 | multi_passage | en | 3 -> none | 193 32 320 548 234 | 32 547 195 399 193 |
| q23 | definition | ar | 1 -> 2 | 497 533 509 506 529 | 506 497 530 566 507 |
| q29 | multi_passage | ar | 1 -> none | 320 32 31 400 108 | 547 31 548 195 400 |

### Improvements

| id | type | lang | first relevant rank (before -> after) | pages before | pages after |
|---|---|---|---|---|---|
| q06 | explanation | en | 2 -> 1 | 218 206 234 207 228 | 206 234 218 228 210 |
| q18 | multi_passage | en | 4 -> 2 | 533 280 573 244 490 | 280 490 573 488 487 |
| q28 | comparison | ar | 4 -> 1 | 272 269 257 255 264 | 255 272 264 468 550 |

## Out-of-scope questions (top-1 cosine score)

| id | top-1 before | top-1 after | change | top-1 page before -> after |
|---|---|---|---|---|
| o01 | 0.456 | 0.365 | -0.091 | 338 -> 20 |
| o02 | 0.401 | 0.401 | +0.000 | 338 -> 338 |
| o03 | 0.362 | 0.305 | -0.058 | 323 -> 328 |
| o04 | 0.461 | 0.461 | +0.000 | 26 -> 26 |
| o05 | 0.663 | 0.663 | +0.000 | 46 -> 46 |

Descriptive score gate (share of questions whose top-1 score is at least the threshold):

| threshold | answerable before | answerable after | out-of-scope before | out-of-scope after |
|---|---|---|---|---|
| 0.00 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.05 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.10 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.15 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.20 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.25 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.30 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.35 | 100.0% | 100.0% | 100.0% | 80.0% |
| 0.40 | 100.0% | 100.0% | 80.0% | 80.0% |
| 0.45 | 100.0% | 100.0% | 60.0% | 40.0% |
| 0.50 | 100.0% | 100.0% | 20.0% | 20.0% |
| 0.55 | 96.7% | 96.7% | 20.0% | 20.0% |
| 0.60 | 83.3% | 83.3% | 20.0% | 20.0% |
| 0.65 | 50.0% | 46.7% | 20.0% | 20.0% |
| 0.70 | 20.0% | 20.0% | 0.0% | 0.0% |
| 0.75 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.80 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.85 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.90 | 0.0% | 0.0% | 0.0% | 0.0% |

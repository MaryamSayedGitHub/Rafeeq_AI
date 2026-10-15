# Retrieval comparison

- before: dense (unchanged baseline behaviour) (run 2026-10-09 16:20:08)
- after: dense, index-like pages excluded (run 2026-10-09 16:21:18)

## Metrics

Small samples: with ~30 questions one question moves a Hit rate by 3.3 points. Read the lists below, not only the deltas.

**all answerable** (n=30)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 23.3% | 23.3% | 0 |
| full_recall@3 | 43.3% | 43.3% | 0 |
| full_recall@5 | 50.0% | 50.0% | 0 |
| hit@1 | 56.7% | 56.7% | 0 |
| hit@3 | 73.3% | 80.0% | +6.7 pp |
| hit@5 | 83.3% | 83.3% | 0 |
| mrr@5 | 0.658 | 0.669 | +0.011 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 56.7% | 56.7% | 0 |
| precision@3 | 32.2% | 34.4% | +2.2 pp |
| precision@5 | 23.3% | 23.3% | 0 |
| recall@1 | 38.3% | 38.3% | 0 |
| recall@3 | 57.2% | 59.4% | +2.2 pp |
| recall@5 | 66.1% | 66.1% | 0 |

**type: comparison** (n=6)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 0.0% | 0.0% | 0 |
| full_recall@3 | 66.7% | 66.7% | 0 |
| full_recall@5 | 83.3% | 83.3% | 0 |
| hit@1 | 83.3% | 83.3% | 0 |
| hit@3 | 83.3% | 83.3% | 0 |
| hit@5 | 100.0% | 100.0% | 0 |
| mrr@5 | 0.875 | 0.875 | 0 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 83.3% | 83.3% | 0 |
| precision@3 | 50.0% | 50.0% | 0 |
| precision@5 | 33.3% | 33.3% | 0 |
| recall@1 | 41.7% | 41.7% | 0 |
| recall@3 | 75.0% | 75.0% | 0 |
| recall@5 | 91.7% | 91.7% | 0 |

**type: definition** (n=8)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 62.5% | 62.5% | 0 |
| full_recall@3 | 75.0% | 75.0% | 0 |
| full_recall@5 | 75.0% | 75.0% | 0 |
| hit@1 | 62.5% | 62.5% | 0 |
| hit@3 | 75.0% | 75.0% | 0 |
| hit@5 | 75.0% | 75.0% | 0 |
| mrr@5 | 0.667 | 0.667 | 0 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 62.5% | 62.5% | 0 |
| precision@3 | 25.0% | 25.0% | 0 |
| precision@5 | 15.0% | 15.0% | 0 |
| recall@1 | 62.5% | 62.5% | 0 |
| recall@3 | 75.0% | 75.0% | 0 |
| recall@5 | 75.0% | 75.0% | 0 |

**type: explanation** (n=9)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 22.2% | 22.2% | 0 |
| full_recall@3 | 33.3% | 33.3% | 0 |
| full_recall@5 | 44.4% | 44.4% | 0 |
| hit@1 | 55.6% | 55.6% | 0 |
| hit@3 | 88.9% | 88.9% | 0 |
| hit@5 | 88.9% | 88.9% | 0 |
| mrr@5 | 0.704 | 0.704 | 0 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 55.6% | 55.6% | 0 |
| precision@3 | 37.0% | 37.0% | 0 |
| precision@5 | 26.7% | 26.7% | 0 |
| recall@1 | 37.0% | 37.0% | 0 |
| recall@3 | 59.3% | 59.3% | 0 |
| recall@5 | 66.7% | 66.7% | 0 |

**type: multi_passage** (n=7)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 0.0% | 0.0% | 0 |
| full_recall@3 | 0.0% | 0.0% | 0 |
| full_recall@5 | 0.0% | 0.0% | 0 |
| hit@1 | 28.6% | 28.6% | 0 |
| hit@3 | 42.9% | 71.4% | +28.6 pp |
| hit@5 | 71.4% | 71.4% | 0 |
| mrr@5 | 0.405 | 0.452 | +0.048 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 28.6% | 28.6% | 0 |
| precision@3 | 19.0% | 28.6% | +9.5 pp |
| precision@5 | 20.0% | 20.0% | 0 |
| recall@1 | 9.5% | 9.5% | 0 |
| recall@3 | 19.0% | 28.6% | +9.5 pp |
| recall@5 | 33.3% | 33.3% | 0 |

**language: ar** (n=10)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 30.0% | 30.0% | 0 |
| full_recall@3 | 50.0% | 50.0% | 0 |
| full_recall@5 | 60.0% | 60.0% | 0 |
| hit@1 | 50.0% | 50.0% | 0 |
| hit@3 | 70.0% | 80.0% | +10.0 pp |
| hit@5 | 90.0% | 90.0% | 0 |
| mrr@5 | 0.633 | 0.658 | +0.025 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 50.0% | 50.0% | 0 |
| precision@3 | 30.0% | 33.3% | +3.3 pp |
| precision@5 | 22.0% | 22.0% | 0 |
| recall@1 | 38.3% | 38.3% | 0 |
| recall@3 | 56.7% | 60.0% | +3.3 pp |
| recall@5 | 70.0% | 70.0% | 0 |

**language: en** (n=20)

| metric | before | after | change |
|---|---|---|---|
| full_recall@1 | 20.0% | 20.0% | 0 |
| full_recall@3 | 40.0% | 40.0% | 0 |
| full_recall@5 | 45.0% | 45.0% | 0 |
| hit@1 | 60.0% | 60.0% | 0 |
| hit@3 | 75.0% | 80.0% | +5.0 pp |
| hit@5 | 80.0% | 80.0% | 0 |
| mrr@5 | 0.671 | 0.675 | +0.004 |
| other_document_share@1 | 0.0% | 0.0% | 0 |
| other_document_share@3 | 0.0% | 0.0% | 0 |
| other_document_share@5 | 0.0% | 0.0% | 0 |
| precision@1 | 60.0% | 60.0% | 0 |
| precision@3 | 33.3% | 35.0% | +1.7 pp |
| precision@5 | 24.0% | 24.0% | 0 |
| recall@1 | 38.3% | 38.3% | 0 |
| recall@3 | 57.5% | 59.2% | +1.7 pp |
| recall@5 | 64.2% | 64.2% | 0 |

## Answerable questions

Improved: **2**, regressed: **0**, unchanged rank: 28

### Regressions (rank of first relevant page got worse)

none

### Improvements

| id | type | lang | first relevant rank (before -> after) | pages before | pages after |
|---|---|---|---|---|---|
| q18 | multi_passage | en | 4 -> 3 | 533 280 573 244 490 | 533 280 244 490 236 |
| q30 | multi_passage | ar | 4 -> 2 | 573 492 576 490 577 | 492 490 499 280 448 |

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
| 0.00 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.05 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.10 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.15 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.20 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.25 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.30 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.35 | 100.0% | 100.0% | 100.0% | 100.0% |
| 0.40 | 100.0% | 100.0% | 80.0% | 80.0% |
| 0.45 | 100.0% | 100.0% | 60.0% | 60.0% |
| 0.50 | 100.0% | 100.0% | 20.0% | 20.0% |
| 0.55 | 96.7% | 96.7% | 20.0% | 20.0% |
| 0.60 | 83.3% | 83.3% | 20.0% | 20.0% |
| 0.65 | 50.0% | 50.0% | 20.0% | 20.0% |
| 0.70 | 20.0% | 20.0% | 0.0% | 0.0% |
| 0.75 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.80 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.85 | 0.0% | 0.0% | 0.0% | 0.0% |
| 0.90 | 0.0% | 0.0% | 0.0% | 0.0% |

# Held-out comparison: Dense, Dense + exclusion, Hybrid, Hybrid + exclusion

Benchmark: `hands_on_llm_heldout_v1` (frozen; see eval/HELDOUT_PROTOCOL.md). Same embedder, store, room, cut-offs and candidate pool 50 for the hybrid runs.

**Status of every number below: VERIFIED. They are computed from real `run_eval.py` results.json files; nothing here is a replay or a simulation.** Inputs: Dense = `eval\results\heldout_dense\results.json`, Dense + exclusion = `eval\results\heldout_dense_exclude\results.json`, Hybrid = `eval\results\heldout_hybrid\results.json`, Hybrid + exclusion = `eval\results\heldout_hybrid_exclude\results.json`.

With 42 answerable questions one question moves a Hit rate by 2.4 points, and with 19 multi-passage questions by 5.3 points. Read the per-question lists, not only the percentages; the sign test below has little power at this size.

## Metrics

### All answerable questions (n=42)

| configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Full Recall@5 |
|---|---|---|---|---|---|---|
| Dense | 66.7% | 90.5% | 95.2% | 0.773 | 86.5% | 76.2% |
| Dense + exclusion | 66.7% | 92.9% | 95.2% | 0.775 | 87.3% | 78.6% |
| Hybrid | 64.3% | 90.5% | 97.6% | 0.772 | 90.5% | 83.3% |
| Hybrid + exclusion | 66.7% | 90.5% | 100.0% | 0.788 | 92.5% | 83.3% |

### Single-passage questions (n=23)

| configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Full Recall@5 |
|---|---|---|---|---|---|---|
| Dense | 56.5% | 87.0% | 91.3% | 0.699 | 91.3% | 91.3% |
| Dense + exclusion | 56.5% | 91.3% | 91.3% | 0.703 | 91.3% | 91.3% |
| Hybrid | 52.2% | 91.3% | 100.0% | 0.710 | 100.0% | 100.0% |
| Hybrid + exclusion | 56.5% | 91.3% | 100.0% | 0.732 | 100.0% | 100.0% |

### Multi-passage questions (Full Recall@5 = every labelled page in the top 5) (n=19)

| configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Recall@5 | Full Recall@5 |
|---|---|---|---|---|---|---|
| Dense | 78.9% | 94.7% | 100.0% | 0.861 | 80.7% | 57.9% |
| Dense + exclusion | 78.9% | 94.7% | 100.0% | 0.861 | 82.5% | 63.2% |
| Hybrid | 78.9% | 89.5% | 94.7% | 0.846 | 78.9% | 63.2% |
| Hybrid + exclusion | 78.9% | 89.5% | 100.0% | 0.857 | 83.3% | 63.2% |

## Abstention gate (descriptive; the 0.35 threshold is the project default and was not tuned)

| configuration | answerable questions passing the 0.35 gate | out-of-scope questions passing the gate (false accepts) | highest out-of-scope best-cosine |
|---|---|---|---|
| Dense | 42/42 | 12/12 | 0.613 |
| Dense + exclusion | 42/42 | 12/12 | 0.613 |
| Hybrid | 42/42 | 12/12 | 0.613 |
| Hybrid + exclusion | 42/42 | 12/12 | 0.613 |

## Paired comparisons, per question

### Dense  ->  Hybrid

Questions whose first relevant rank improved: **7**, regressed: **5**, unchanged: 30. Exact two-sided sign test on the 12 changed questions: p = 0.774.

**Regressions**

| id | type | lang | first relevant rank | pages before | pages after |
|---|---|---|---|---|---|
| a01 | explanation | ar | 2 -> 4 | 402 404 595 106 544 | 402 595 544 404 549 |
| a03 | definition | ar | 2 -> 3 | 427 409 422 514 554 | 427 422 409 514 418 |
| a06 | comparison | ar | 1 -> 3 | 420 416 414 417 435 | 416 414 420 434 417 |
| a09 | multi_passage | ar | 3 -> none | 361 362 363 364 381 | 361 362 379 594 380 |
| m02 | multi_passage | en | 1 -> 3 | 480 475 476 81 478 | 475 476 479 480 477 |

**Improvements**

| id | type | lang | first relevant rank | pages before | pages after |
|---|---|---|---|---|---|
| a04 | explanation | ar | 3 -> 2 | 435 441 439 593 440 | 435 439 593 440 406 |
| a05 | explanation | ar | 4 -> 2 | 554 17 135 517 171 | 554 517 403 113 17 |
| h02 | explanation | en | none -> 4 | 289 257 288 290 278 | 289 288 240 259 587 |
| h13 | explanation | en | 3 -> 2 | 176 177 179 468 459 | 176 179 177 468 595 |
| h14 | explanation | en | none -> 2 | 321 334 359 317 322 | 317 329 321 334 338 |
| m05 | multi_passage | en | 3 -> 1 | 269 266 268 267 262 | 268 266 269 577 267 |
| m08 | multi_passage | en | 5 -> 4 | 461 464 468 460 467 | 461 460 464 466 468 |

Out-of-scope best top-1 cosine (descriptive):

| id | before | after |
|---|---|---|
| x01 | 0.370 | 0.370 |
| x02 | 0.423 | 0.369 |
| x03 | 0.531 | 0.531 |
| x04 | 0.611 | 0.606 |
| x05 | 0.613 | 0.613 |
| x06 | 0.360 | 0.360 |
| x07 | 0.407 | 0.407 |
| x08 | 0.506 | 0.506 |
| x09 | 0.489 | 0.489 |
| x10 | 0.492 | 0.464 |
| x11 | 0.495 | 0.495 |
| x12 | 0.526 | 0.526 |

### Dense + exclusion  ->  Hybrid + exclusion

Questions whose first relevant rank improved: **7**, regressed: **5**, unchanged: 30. Exact two-sided sign test on the 12 changed questions: p = 0.774.

**Regressions**

| id | type | lang | first relevant rank | pages before | pages after |
|---|---|---|---|---|---|
| a02 | definition | ar | 3 -> 4 | 412 415 411 414 427 | 412 427 415 411 414 |
| a03 | definition | ar | 2 -> 3 | 427 409 422 514 418 | 427 422 409 418 438 |
| a06 | comparison | ar | 1 -> 3 | 420 416 414 417 435 | 416 414 420 434 417 |
| a09 | multi_passage | ar | 3 -> 5 | 361 362 363 364 381 | 361 362 379 380 363 |
| m02 | multi_passage | en | 1 -> 3 | 480 475 476 81 478 | 475 476 479 477 480 |

**Improvements**

| id | type | lang | first relevant rank | pages before | pages after |
|---|---|---|---|---|---|
| a04 | explanation | ar | 3 -> 2 | 435 441 439 440 175 | 435 439 440 406 43 |
| a05 | explanation | ar | 3 -> 1 | 17 135 517 171 105 | 517 403 113 17 135 |
| h02 | explanation | en | none -> 4 | 289 257 288 290 278 | 289 288 240 259 285 |
| h10 | definition | en | 3 -> 2 | 535 526 532 527 525 | 535 532 525 526 531 |
| h14 | explanation | en | none -> 3 | 321 334 359 317 322 | 321 317 329 334 322 |
| m05 | multi_passage | en | 3 -> 1 | 269 266 268 267 262 | 268 266 269 267 264 |
| m08 | multi_passage | en | 5 -> 4 | 461 464 468 460 467 | 461 460 464 466 468 |

Out-of-scope best top-1 cosine (descriptive):

| id | before | after |
|---|---|---|
| x01 | 0.370 | 0.370 |
| x02 | 0.423 | 0.369 |
| x03 | 0.531 | 0.531 |
| x04 | 0.611 | 0.606 |
| x05 | 0.613 | 0.613 |
| x06 | 0.360 | 0.360 |
| x07 | 0.407 | 0.407 |
| x08 | 0.506 | 0.506 |
| x09 | 0.489 | 0.489 |
| x10 | 0.492 | 0.464 |
| x11 | 0.495 | 0.495 |
| x12 | 0.526 | 0.526 |

### Dense  ->  Dense + exclusion

Questions whose first relevant rank improved: **1**, regressed: **0**, unchanged: 41. Exact two-sided sign test on the 1 changed questions: p = 1.000.

**Regressions**

none

**Improvements**

| id | type | lang | first relevant rank | pages before | pages after |
|---|---|---|---|---|---|
| a05 | explanation | ar | 4 -> 3 | 554 17 135 517 171 | 17 135 517 171 105 |

Out-of-scope best top-1 cosine (descriptive):

| id | before | after |
|---|---|---|
| x01 | 0.370 | 0.370 |
| x02 | 0.423 | 0.423 |
| x03 | 0.531 | 0.531 |
| x04 | 0.611 | 0.611 |
| x05 | 0.613 | 0.613 |
| x06 | 0.360 | 0.360 |
| x07 | 0.407 | 0.407 |
| x08 | 0.506 | 0.506 |
| x09 | 0.489 | 0.489 |
| x10 | 0.492 | 0.492 |
| x11 | 0.495 | 0.495 |
| x12 | 0.526 | 0.526 |

### Hybrid  ->  Hybrid + exclusion

Questions whose first relevant rank improved: **4**, regressed: **3**, unchanged: 35. Exact two-sided sign test on the 7 changed questions: p = 1.000.

**Regressions**

| id | type | lang | first relevant rank | pages before | pages after |
|---|---|---|---|---|---|
| a02 | definition | ar | 3 -> 4 | 412 427 411 415 433 | 412 427 415 411 414 |
| h13 | explanation | en | 2 -> 3 | 176 179 177 468 595 | 176 177 179 468 459 |
| h14 | explanation | en | 2 -> 3 | 317 329 321 334 338 | 321 317 329 334 322 |

**Improvements**

| id | type | lang | first relevant rank | pages before | pages after |
|---|---|---|---|---|---|
| a01 | explanation | ar | 4 -> 2 | 402 595 544 404 549 | 402 404 106 462 103 |
| a05 | explanation | ar | 2 -> 1 | 554 517 403 113 17 | 517 403 113 17 135 |
| a09 | multi_passage | ar | none -> 5 | 361 362 379 594 380 | 361 362 379 380 363 |
| h10 | definition | en | 3 -> 2 | 535 572 532 571 525 | 535 532 525 526 531 |

Out-of-scope best top-1 cosine (descriptive):

| id | before | after |
|---|---|---|
| x01 | 0.370 | 0.370 |
| x02 | 0.369 | 0.369 |
| x03 | 0.531 | 0.531 |
| x04 | 0.606 | 0.606 |
| x05 | 0.613 | 0.613 |
| x06 | 0.360 | 0.360 |
| x07 | 0.407 | 0.407 |
| x08 | 0.506 | 0.506 |
| x09 | 0.489 | 0.489 |
| x10 | 0.464 | 0.464 |
| x11 | 0.495 | 0.495 |
| x12 | 0.526 | 0.526 |

## First relevant rank of every answerable question (`-` = not in the top 5)

| id | type | lang | expected pages | Dense | Dense + exclusion | Hybrid | Hybrid + exclusion |
|---|---|---|---|---|---|---|---|
| h01 | definition | en | 250 | 1 | 1 | 1 | 1 |
| h02 | explanation | en | 259 | - | - | 4 | 4 |
| h03 | explanation | en | 461 | 1 | 1 | 1 | 1 |
| h04 | comparison | en | 471 | 1 | 1 | 1 | 1 |
| h05 | explanation | en | 405 | 1 | 1 | 1 | 1 |
| h06 | definition | en | 512 | 1 | 1 | 1 | 1 |
| h07 | explanation | en | 516 | 1 | 1 | 1 | 1 |
| h08 | explanation | en | 520 | 1 | 1 | 1 | 1 |
| h09 | explanation | en | 522 | 2 | 2 | 2 | 2 |
| h10 | definition | en | 532 | 3 | 3 | 3 | 2 |
| h11 | explanation | en | 507 | 1 | 1 | 1 | 1 |
| h12 | explanation | en | 506 | 1 | 1 | 1 | 1 |
| a01 | explanation | ar | 404 | 2 | 2 | 4 | 2 |
| a02 | definition | ar | 411 | 3 | 3 | 3 | 4 |
| a03 | definition | ar | 409 | 2 | 2 | 3 | 3 |
| a04 | explanation | ar | 439 | 3 | 3 | 2 | 2 |
| a05 | explanation | ar | 517 | 4 | 3 | 2 | 1 |
| a06 | comparison | ar | 420 | 1 | 1 | 3 | 3 |
| m01 | multi_passage | en | 241 242 | 1 | 1 | 1 | 1 |
| m02 | multi_passage | en | 479 480 | 1 | 1 | 3 | 3 |
| m03 | multi_passage | en | 424 425 | 1 | 1 | 1 | 1 |
| m04 | multi_passage | en | 526 527 | 1 | 1 | 1 | 1 |
| m05 | multi_passage | en | 267 268 | 3 | 3 | 1 | 1 |
| m06 | multi_passage | en | 251 252 | 1 | 1 | 1 | 1 |
| m07 | multi_passage | en | 297 300 304 | 1 | 1 | 1 | 1 |
| m08 | multi_passage | en | 466 467 | 5 | 5 | 4 | 4 |
| m09 | multi_passage | en | 469 474 | 1 | 1 | 1 | 1 |
| m10 | multi_passage | en | 430 435 | 1 | 1 | 1 | 1 |
| m11 | multi_passage | en | 516 517 | 1 | 1 | 1 | 1 |
| m12 | multi_passage | en | 498 499 | 1 | 1 | 1 | 1 |
| a07 | multi_passage | ar | 524 525 | 1 | 1 | 1 | 1 |
| a08 | multi_passage | ar | 488 489 | 1 | 1 | 1 | 1 |
| a09 | multi_passage | ar | 363 365 | 3 | 3 | - | 5 |
| h13 | explanation | en | 179 | 3 | 3 | 2 | 3 |
| h14 | explanation | en | 329 | - | - | 2 | 3 |
| a10 | explanation | ar | 331 | 1 | 1 | 1 | 1 |
| h16 | comparison | en | 348 | 1 | 1 | 1 | 1 |
| h17 | explanation | en | 355 | 1 | 1 | 1 | 1 |
| m13 | multi_passage | en | 326 328 | 1 | 1 | 1 | 1 |
| m14 | multi_passage | en | 333 334 | 1 | 1 | 1 | 1 |
| m15 | multi_passage | en | 182 183 185 | 2 | 2 | 2 | 2 |
| a11 | multi_passage | ar | 186 187 | 1 | 1 | 1 | 1 |


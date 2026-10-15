# Member 3 evaluation report
Generated 2026-10-10T16:36:36+00:00 on Windows-10-10.0.26200-SP0. Every number below comes from running the code in this repository; nothing is estimated.

## 1. CER with and without preprocessing, VLM against OCR baseline
Data: **DEMO images (synthetic, NOT real handwriting)**, 20 images. Lower is better.

| reader / preset | ar | code | en | math | ALL |
|---|---|---|---|---|---|
| easyocr / pre | 0.037 | 0.152 | 0.39 | 0.157 | 0.184 |
| easyocr / raw | 0.019 | 0.052 | 0.367 | 0.14 | 0.145 |
| ollama / pre | 0.219 | 0.091 | 0.143 | 0.063 | 0.129 |
| ollama / raw | 0.879 | 0.341 | 0.265 | 0.651 | 0.534 |

- easyocr: raw 0.145 -> pre 0.184 (pre - raw = 0.04, negative = preprocessing helps) on 20 images
- ollama: raw 0.534 -> pre 0.129 (pre - raw = -0.405, negative = preprocessing helps) on 20 images
- baseline (EasyOCR/PaddleOCR) next to a vision model: yes

## 2-4. Step status, first error, hints, uncertainty
Data: developer-written cases in work_cases.json (text, not photos), 14 cases. LLM available: **True** (qwen3:8b).
- step_status_accuracy: 0.947 (36/38)
- first_error_localization: 0.929 (13/14)
- hint_relevance_keyword_proxy: 0.714 (5/7)
- hint_format: 1.0 (38/38), rule: one sentence <= 28 words for a step that is not ok, empty for ok
- uncertainty_handling: 1.0 (4/4)
- incomplete_work_detected: 1.0 (1/1)
- hint_leaks: {'hints_containing_a_solution_string': 0, 'hints_containing_a_line_of_working': 0}  (must both be 0)
- hint relevance is a keyword proxy against a hand-written list, not a human judgement.

## 5. Citation validity
Material source: **sample**. material_source 'sample' means M2's real retrieval was not reachable: these are the agreed sample passages, not your database
- citation_validity: 1.0 (3/3)
- support of answers by their cited passages: {'supported': 2, 'unchecked': 1, 'none': 2}
- answerable questions answered: 1.0 (3/3)
- unanswerable questions refused with no citation: 1.0 (2/2)

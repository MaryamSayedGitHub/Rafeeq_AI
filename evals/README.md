# Evaluation pipeline

One command runs every feature's suite and writes one report:

```powershell
uv run python evals/run_all.py                  # everything, offline (about 10 seconds)
uv run python evals/run_all.py --only montage   # one suite
uv run python evals/run_all.py --live           # also the checks that call the models (Groq generator, Gemini judge)
uv run python evals/run_all.py --no-upload      # keep the results local (no LangSmith / Langfuse)
uv run python evals/run_all.py --no-fail        # always exit 0
```

It writes `evals/results/report.md` (read this one, ordered like the M5 presentation), `deck.json` (the headline numbers per
feature, for the slides), `latest.json` and one `run-<time>.json` per run. The report has a
column **since last run**, so a change to a prompt, a model or the code shows up as a number that moved. The exit
status is 1 when a gate fails, so CI can refuse a merge that makes a feature worse.

## How it is organised

- Every feature has its own suite: `evals/suites/<feature>.py`. Each metric has a **gate** (the minimum we accept).
- `evals/core.py` has what suites share: `Metric`, WER, range overlap, run-to-run comparison.
- Suites already here, for the **teacher** side (M5): **montage**, **class_report**, **parent_report**, **inbox**, **parent_batch**.
  They run on simulated or developer-written data with a fixed seed, so the same code gives the same numbers.
  Also **m3_handwriting** (reads M3's own `results/report.json`; it does not re-run OCR).
- The data in those suites is written by us (a simulated class, hand-written question pairs). It shows whether the code does what
  it should and finds real bugs, but it is not a measurement on real students. Each metric's note says what it is built on.
- A suite never invents a number: when a model, key or data file is missing the metric says **SKIPPED** and why.

## Adding your feature (5 minutes)

1. Copy `evals/suites/_template.py` to `evals/suites/<your_feature>.py`.
2. Return a list of `Metric(name, value, unit, gate=..., higher_is_better=..., note=...)`.
3. If you already have an eval script (M2's `rafeeq/knowledge/scripts/run_eval.py`, the quiz judge), do what
   `m3_handwriting.py` does: read the JSON it writes and expose its headline metrics. Do not copy its logic.
4. `uv run python evals/run_all.py --only <your_feature>`, then commit the suite (not `evals/results/`).

## What the montage suite measures

| metric | meaning |
|---|---|
| speech_kept / silence_removed / wrongly_cut_seconds | does the pause cutter keep every spoken second and remove the pauses? |
| length_error / sound_survives / failed_steps | does the finished video have the length and sound it should, with no step failing? |
| caption_time_error | a caption placed on the ORIGINAL timeline lands at the right time in the CUT video |
| gif_visible_in_window / gif_absent_outside_window | the GIF is on screen when placed, and only then |
| seconds_per_video_minute | processing speed on this machine |
| caption_wer, chapter_position_error | need your labelled recordings and `--live` (see `evals/data/montage/README.md`) |

## The other teacher suites

| suite | what it checks |
|---|---|
| class_report | a simulated class (40 students x 6 topics, noisy quizzes): are the right (student, topic) pairs flagged, is the topic order right, how many quizzes until improvement shows, how well does the no-LLM topic grouping work |
| parent_report | counts and average recomputed from the raw records; every number and topic in the Arabic summary is real; the signed link rejects edits, expiry, another secret |
| inbox | the same question always gets the teacher's answer; rewordings are reused; DIFFERENT questions on the same topic must not get a wrong answer (the threshold sweep is in the JSON) |
| parent_batch | 24 sheet layouts parse; Egyptian phone numbers (including the Excel version that lost its 0) become the wa.me number; a contact never goes to the wrong student |

## LangSmith and Langfuse

Both are optional and both get the same numbers as `report.md`. Set them in `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rafeeq
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com      # or https://us.cloud.langfuse.com
```

and install the Langfuse SDK once: `uv add langfuse`.

What you get per run:

| | LangSmith | Langfuse |
|---|---|---|
| every live case | a trace `eval · <suite> · <case>` with every LangGraph node / LLM / judge call nested | the same trace, with its scores attached |
| every suite | dataset `rafeeq-m5-<suite>` + an experiment `<suite>-...` (per-case scores, `suite.<metric>` summary scores) | dataset `rafeeq-m5-<suite>` + a dataset run `<suite> <time>` (item scores + run scores) |
| the app itself | LangGraph runs (lecturer, montage team) + `traced()` direct calls | the same through the LangChain callback + `traced()` |

How it works (`evals/observe.py`): each case runs ONCE on your machine inside both traces; the scorers (and the judge)
run inside the same trace. Then the saved outputs and scores are replayed into each platform's experiment API, so no
model is called twice and both dashboards agree with the report. A missing key or a network error never breaks the
local report; the report says where each suite was published.

## Live suites (one per feature, `--live`)

| feature (slide) | suite | cases | what is measured |
|---|---|---|---|
| Lesson studio + Presentations | `lesson_studio` | 6 briefs (en/ar, with and without code) | parts complete, quiz shape, quiz answered blind by the judge vs the key, code verified by running it, language, bullet rule, judge grade, .pptx export + versioning |
| Video discovery | `video_discovery` | 8 questions (en/ar) | answered, top-1 relevant and precision@3 (judge), valid timestamps, first vs repeat search time |
| Animated video | `story_video` | 5 topics (en/ar) | scene count, narration 18-32 words, distinct scenes, language, judge grade of the teaching arc (+ offline planner guards) |
| Voice to Visual (future) | `voice_visual` | 3 hand-labelled lectures | moments on teaching lines, coverage, rules, label language, judge grade (+ offline plan clean-up) |
| My library & learning platform | `learning_platform` | scripted class (offline) | library versions, publish snapshot, access rules, progress, class summary |

The judge is Gemini (`EVAL_JUDGE_MODEL`, else `GEMINI_MODEL`), a different model family from the Groq generator, at
temperature 0. A judge that cannot answer gives no score, never a guessed one.

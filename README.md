# Rafeeq · رفيق

A multi-agent learning copilot for students and teachers, Arabic-first with an English toggle. Built as an NTI AI capstone project.

Rafeeq combines a RAG tutor grounded in the course material, a student-modeling layer that predicts weak topics, agents for planning, quizzing and reporting, handwriting/photo recognition with hints (Snap & Ask), and a teacher dashboard.

## Features

- **Ask** — a tutor that answers only from the room's own material, with page citations, in Arabic or English.
- **Snap & Ask** — photograph a question or your own worked solution; a vision-language model reads it and the tutor answers or checks it.
- **Quiz** — generated per topic, graded automatically, feeding back into the student's profile.
- **My plan** — a "what to do next" suggestion based on weak topics.
- **Rooms** — a course room per teacher, with student uploads and (proposed) a personal study room with no teacher.
- **Teacher dashboard** — class-wide insights, a per-topic report, and an assistant that drafts replies for approval.

## Stack

FastAPI + Jinja2 + HTMX (one server-rendered app, no separate frontend build), Postgres with pgvector, Docker for local development, deployed on Render. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

## Run it locally

```bash
uv sync
uv run uvicorn rafeeq.main:app --reload
```

Open http://127.0.0.1:8000 — you should see the Arabic home page.

**With Docker** (also starts Postgres):

```bash
docker compose up --build
```

## The student pages

The site opens on `/` with every page in the top bar, in Arabic or English (the **EN / عربي** switch, kept in a cookie).

| Page | What it does |
|---|---|
| **Ask** `/chat` | A question about the room's material: grounded answer + clickable sources (the exact page) + 3 MCQ + "explain simpler / give an example / compare with ..." + read aloud. A math or code problem gets a step check, then **3 hints one at a time, then the full solution with steps**. Web research writes a study file of 1..15 pages. Topic chips filter the search. |
| **Snap** `/snap` | Photo (file or device camera) of a question or handwritten work: the text, then every step checked, the wrong line boxed on the page; wrong or unfinished work gets the 3 hints and the full solution. |
| **Study files** `/study` | Read, download Word / PDF / Markdown, quiz yourself, short summary, flashcards, delete. |
| **Quiz** `/quiz` | Pick a topic (M2) and take M4's quiz; scored by M4 (here until it exists); mistakes go to Review. |
| **Review** `/review` | Mistakes come back after 1, 7 and 30 days until they stick. |
| **My plan, Rooms** | M4's next step and weak topics; the rooms. |

M2 and M4 are called exactly as `contracts/openapi.yaml` says (`rafeeq/tutor/m2.py`, `rafeeq/tutor/m4.py`): M2 `/search`, `/rooms/{id}/topics`, `/sources/page`; M4 `/quiz`, `/quiz/{id}/submit`, `/next-step`, `/profile/summary`, `/events` (`question` and `understood`). Until their routes exist the pages use the agreed sample data and say so on screen. `M2_BASE_URL`, `M4_BASE_URL`, `TUTOR_SAMPLE_FALLBACK=0` in `.env`. Settings: `LLM_PROVIDER=ollama`, `OLLAMA_MODEL=qwen3:8b`, `TAVILY_API_KEY`, `SNAP_VISION_MODEL=qwen2.5vl:7b`. Level 2, Snap and Review data are kept in the git-ignored `data/` folder.

## M3 tutor API and the handwriting measurements

Room routes (agreed in `contracts/openapi.yaml`, tag M3): `POST /rooms/{id}/ask` (Level 1 or 2), `POST /rooms/{id}/vision/check-work` (photo), `GET /rooms/{id}/questions` (the student's own log), `GET /rooms/{id}/instructor/questions` (anonymous, `?format=csv`, `INSTRUCTOR_PIN` header when set) and `POST /rooms/{id}/instructor/publish` (to `M5_PUBLISH_URL`). Ask runs the LangGraph agent in `rafeeq/tutor/agent.py`: tag_topic, M2 answer, log, M4 event; Level 2: search (cached), study file, saved.

Handwriting: `rafeeq/snap/` is OpenCV (perspective, deskew, denoise, adaptive threshold), line boxes, a vision model per line (`SNAP_READER=ollama|gemini`, baselines `easyocr|paddleocr`). To measure it on your 20 REAL photos:

```bash
# photos in data/images/ named ar_01.jpg en_02.jpg math_03.jpg code_04.jpg
uv run python -m rafeeq.snap.init_gt                 # creates data/ground_truth.csv: fill the text column by hand
uv run python -m rafeeq.snap.evaluate --readers easyocr ollama --presets raw pre --debug   # CER with and without preprocessing, vision model vs baseline
```

`python -m rafeeq.snap.make_demo_set` makes 20 DEMO images (drawn text, not real handwriting) in `data/demo_images/` to check that the measuring works; do not report their numbers as results. Output goes to `results/` (`summary_cer.csv`, `results.csv`, `debug/`).

### How M3 is connected to the other members

- **M1 (Copilot, /snap, /app UI)**: `rafeeq/tutor/connect.py` (`connect_m3()`, called in `rafeeq/main.py`) plugs M3 into `rafeeq/copilot/clients.py`: `tutor_answer` (M2 retrieve, verified citations, `topic_id`), `check_work` (every step ok/check/error, the real box of the wrong line), `read_image` (OpenCV + vision model for `POST /snap`) and `record_event` (delivers the Copilot's QuestionEvent to M4). M1's UI at `/app` therefore shows live M3 answers instead of samples. `RAFEEQ_CONNECT_M3=0` switches it off (tests do).
- **M2**: `rafeeq/tutor/m2.py` calls `POST /search`, `GET /rooms/{id}/topics`, `GET /sources/page` exactly as in the contract; until they exist the agreed samples are used and labelled `sample`.
- **M4**: `rafeeq/tutor/events.py` posts the QuestionEvent (`question` / `understood`) to `EVENTS_URL`.
- **M5**: `GET /rooms/{id}/instructor/questions` (anonymous, no names or ids; `?format=csv`) and `POST .../instructor/publish` to `M5_PUBLISH_URL`.
- **The tutor agent in the green app** (`rafeeq/tutor/tutor_agent.py`, a LangGraph agent under M1's Copilot): Ask Rafeeq (`/app/rooms/{id}/study`) now offers, under every answer, the 3-question quiz, simpler / example / compare, and "go deeper" (Level 2: web search, a study file of N pages, saved). A study file has quiz, summary, flashcards and Word / PDF / Markdown downloads, and the room's files are listed in the **Study files** tab. A math or code problem gets 3 hints one click at a time, then the full solution; a photo gets every step coloured ok / not sure / wrong with one hint each. Everything goes through one contract route, `POST /rooms/{id}/tutor/act`. The old blue site (/chat, /snap, /study, /quiz, /review, /plan, /lab) was removed.
- **Class insights**: what students ask again and again (same topic, nearly the same words, no names) is shown to the teacher in Class insights (`GET /rooms/{id}/instructor/frequent`).

### M3 reliability: uncertainty, citations, routing, evaluation

- **Uncertainty** (`rafeeq/tutor/uncertainty.py`): a handwriting line that may be misread (the reader wrote `[?]`, crossed-out text, an unclear sign like `x ? 5`, noise, letters of two scripts inside one word) is never judged: it is `check` with one question for the student, and when the reader's confidence is below 0.6 or most lines are bad nothing is judged (`needs_clarification`, no model call). The reader marks what it cannot see through `SNAP_MARK_UNCERTAIN` (default 1; the CER evaluation turns it off).
- **Citations** (`rafeeq/tutor/citations.py`): every citation must be a passage M2 returned for this question, and the answer must share real words with its cited text (same script only, otherwise `unchecked`). An unsupported answer is withheld (`answer_flagged`) and Level 2 is offered.
- **Routing** (`rafeeq/tutor/agent.py`): handwriting -> check work (never the web); Level 1 -> M2 only; Level 2 -> Tavily -> study file. One retry at most for a transient failure (M2 down, network); a missing key or quota is never retried. A failed web search or file falls back to the M2 answer instead of an error.
- **Report**: `python -m rafeeq.snap.report --cer-from results/results.csv --cer-label "..."` (or `--run-cer` on your photos) writes `results/report.md` and `report.json`: CER with/without preprocessing, VLM vs OCR baseline, step-status accuracy, first-error localisation, hint relevance (keyword proxy), hint leaks, uncertainty handling, citation validity. Your own labelled cases go in `data/eval/work_cases.json` and `data/eval/questions.json`; a metric with no data says NOT RUN.
- **Adversarial tests**: `tests/test_adversarial.py` (blurry, dark, faded, rotated, upside down, cropped, tiny, blank, crossed out, ambiguous sign, mixed Arabic/English, incomplete, bad uploads, solution leaks, invented citations, retries and fallbacks).

## Run the tests

```bash
uv run pytest
```

This also checks that every file in `contracts/samples/` matches the API contract in `contracts/openapi.yaml`.

## Project structure

```
rafeeq/
├── main.py          # starts the app, plugs in every router
├── samples.py        # load_sample(name) — read an agreed example JSON
├── web/               # page frame, nav, the one app.css
├── platform/          # login, rooms, file uploads, database
├── copilot/           # routes a message to ask / quiz / check_work / plan
├── snap/               # photo → text (OpenCV + a vision-language model)
├── knowledge/          # indexing and search (hybrid retrieval, grounded answers, evaluation) -> knowledge/README.md
├── tutor/              # grounded answers, handwriting check
├── quiz/                # quiz generation, grading, student profile
└── teacher/            # teacher dashboard and reports
contracts/
├── openapi.yaml        # every route, and the JSON shape it sends/receives
└── samples/             # one real example per JSON file
tests/                   # automated checks
docs/knowledge/          # M2 evaluation reports (historical, written in the standalone project)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow and how the team is organized.

## Status

In active development for the NTI AI capstone.

## Team

A five-member team built this for the NTI AI course capstone, Education domain.
# Rafeeq · رفيق

<img width="2001" height="900" alt="rafeeq-banner" src="https://github.com/user-attachments/assets/a6eb651b-ff5f-4b3e-9d1c-e8763d80fe3f" />


> **Every teacher sees. Every student understands.**

An Arabic-first AI learning companion that connects teachers, students and parents around the teacher's own material. Arabic is the default, with an English toggle. Built as an NTI AI capstone project (Education domain, team of 5).

*Rafeeq* means "companion". Each class gets one shared room. AI agents answer only from the teacher's own slides and books, and the teacher stays in charge of what students receive.

## The problem

One teacher cannot see thirty minds at once. Egypt has about 25.8M students in school and 1.26M teachers. Teachers spend around 12 hours a week on planning and marking, and up to 75% of final-year secondary students pay for private tutoring. Struggling students stay invisible until the exam, and families pay to fill the gap. *(Sources: UNESCO GEM / World Education Blog 2025; OECD TALIS; World Bank via Egyptian Streets 2026.)*

## What Rafeeq does

Rafeeq combines a RAG tutor grounded in the course material, a student-modeling layer that predicts weak topics, agents for planning, quizzing and reporting, handwriting/photo recognition with hints (Snap & Ask), and a teacher dashboard.

| For | Features |
|---|---|
| **Students** | Cited answers · handwriting check with one hint at a time · quizzes that come back after 1, 7 and 30 days · a "what to do next" plan |
| **Teachers** | See who is struggling and on which topic · one-click lesson studio (slides, code examples, quiz; the teacher approves before anything is built) · AI-drafted replies for approval |
| **Parents** | A weekly Arabic report by private link, written from fixed templates so nothing is invented |

### Weekly loop

1. **Upload**: the teacher adds slides and books to the class room.
2. **Understand**: Rafeeq reads and indexes every page, in Arabic and English.
3. **Learn**: students ask, snap their work and take quizzes.
4. **Detect**: Rafeeq tracks each student's weak topics.
5. **Act**: the teacher drafts a review lesson in one click.
6. **Inform**: parents receive a clear weekly report.

Steps 3 to 6 repeat every week.

### Feature list

- **Ask**: a tutor that answers only from the room's own material, with page citations, in Arabic or English. If the answer is not in the material it says so.
- **Snap & Ask**: photograph a question or your own worked solution; a vision-language model reads it and the tutor answers or checks it.
- **Quiz**: generated per topic, graded automatically, feeding back into the student's profile.
- **My plan**: a "what to do next" suggestion based on weak topics.
- **Rooms**: a course room per teacher, with student uploads and (proposed) a personal study room with no teacher.
- **Teacher dashboard**: class-wide insights, a per-topic report, and an assistant that drafts replies for approval.
- **Lecturer studio and parent reports**: slides, code examples and a quiz drafted for teacher approval; a weekly parent summary.
- **Video, voice and images** (in testing): story videos, narration and image editing for lessons.

## Stack

FastAPI + Jinja2 + HTMX (one server-rendered app, no separate frontend build), LangGraph agents, Postgres with pgvector, Docker for local development, deployed on Render. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

**Models.** Every model is called through a configurable provider, so any of them can be swapped.

| Role | Local development default | Hosted configuration |
|---|---|---|
| Tutor answers | Ollama `qwen3:8b` | Gemini 3.5 Flash-Lite |
| Router, Snap, quiz, lessons | Ollama `qwen3:8b` | Qwen 27B (Groq) |
| Handwriting reader | `qwen2.5vl:7b` (or Gemini) | vision model via `SNAP_READER` |
| Search and reranking | bge-m3 (local) + FAISS + BM25 | same |
| Web search (Level 2) | Tavily | Tavily |
| Story video / voice | n/a | Wan 2.6 (720p) / edge-tts |

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

## How it is built

One app, one router, specialised agents:
<img width="2001" height="990" alt="rafeeq-architecture" src="https://github.com/user-attachments/assets/e92beba0-c3d4-454b-af20-389cf6dc65e7" />


The full set of diagrams (every feature workflow plus architecture and deployment) is in **[docs/architecture.md](docs/architecture.md)**.

```mermaid
flowchart TB
    subgraph CLIENT["Browser"]
        ST[Student / Teacher / Parent]
        UI["Jinja2 + HTMX UI<br/>Arabic-first, EN toggle"]
    end
    subgraph APP["FastAPI app (rafeeq.main)"]
        COP["M1 Copilot router<br/>ask · quiz · check work · plan"]
        M3["M3 Tutor & Snap<br/>LangGraph agent, citations, uncertainty"]
        KNOW["M2 Knowledge<br/>hybrid retrieval"]
        QUIZ["M4 Quiz & Profile"]
        TEACH["M5 Teacher tools"]
    end
    subgraph AI["AI services"]
        LLM[LLM]
        VLM[Vision model]
        TAV[Tavily]
    end
    subgraph DATA["Persistence"]
        PG[("Postgres + pgvector")]
        FS[("data/ folder")]
    end
    ST --> UI --> COP
    COP --> M3
    M3 --> KNOW
    M3 --> QUIZ
    M3 --> TAV
    M3 --> VLM
    M3 --> LLM
    KNOW --> PG
    QUIZ --> PG
    TEACH --> PG
    M3 --> FS
    UI --> TEACH
```

### Modules

| Module | Responsibility |
|---|---|
| **M1 — Copilot & UI** | User-facing interface; routes each message (keywords, then LLM, then rules) to ask / check work / quiz / plan / clarify |
| **M2 — Knowledge & RAG** | Content indexing, hybrid retrieval (bge-m3 + FAISS + BM25), grounded answers |
| **M3 — Tutor & Snap** | Guided tutoring, citations, uncertainty checks, handwritten-work analysis |
| **M4 — Quiz & Student Profile** | Quiz generation and grading, learning events, student profile, review data |
| **M5 — Teacher Tools** | Dashboard, class insights, reports, teacher-approved replies |

`contracts/openapi.yaml` and `contracts/samples/` define every route and JSON shape between modules.

```mermaid
flowchart LR
    M1["M1 Copilot & UI"] -->|"tutor/act, ask, snap"| M3["M3 Tutor & Snap"]
    M3 -->|"POST /search; GET topics; GET sources/page"| M2["M2 Knowledge"]
    M3 -->|"POST /events: question, understood"| M4["M4 Quiz & Profile"]
    M1 -->|"POST /quiz; /quiz/id/submit; GET /next-step; /profile/summary"| M4
    M3 -->|"GET instructor/questions; /frequent; POST publish"| M5["M5 Teacher"]
    M4 -->|"profile and learning events"| M5
    M2 -.->|"fallback when a route is missing"| SMP["Contract sample data"]
    M4 -.-> SMP
```

### Feature workflows (see [docs/architecture.md](docs/architecture.md) for the diagrams)

1. Ask, Level 1: grounded tutoring with citation validation
2. Ask, Level 2: deeper research into a 1–15 page study file
3. Guided hints for math and code (check, 3 hints, full solution)
4. Snap & Ask: photo and handwriting analysis
5. Quiz
6. Review: spaced repetition (day 1, 7, 30)
7. My plan
8. Copilot message routing
9. Rooms and course materials
10. Study files
11. Teacher dashboard

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

M2 and M4 are called exactly as `contracts/openapi.yaml` says (`rafeeq/tutor/m2.py`, `rafeeq/tutor/m4.py`): M2 `/search`, `/rooms/{id}/topics`, `/sources/page`; M4 `/quiz`, `/quiz/{id}/submit`, `/next-step`, `/profile/summary`, `/events` (`question` and `understood`). Until their routes exist the pages use the agreed sample data and say so on screen.

## Configuration

Set these in `.env` (locally) or in the Render environment:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER`, `OLLAMA_MODEL` | Which LLM provider and model to use (default `ollama`, `qwen3:8b`) |
| `TAVILY_API_KEY` | Web search for Level 2 |
| `SNAP_VISION_MODEL`, `SNAP_READER` | Handwriting reader (`qwen2.5vl:7b`; `ollama` or `gemini`; baselines `easyocr`, `paddleocr`) |
| `M2_BASE_URL`, `M4_BASE_URL` | Where to reach M2 and M4 |
| `EVENTS_URL` | Where QuestionEvents are posted (M4) |
| `M5_PUBLISH_URL` | Where approved teacher replies are published |
| `INSTRUCTOR_PIN` | When set, required as a header on the instructor routes |
| `TUTOR_SAMPLE_FALLBACK=0` | Turn off the sample-data fallback |
| `RAFEEQ_CONNECT_M3=0` | Switch off the M3 connection (tests do) |
| `SNAP_MARK_UNCERTAIN` | Reader marks what it cannot see (default 1) |

Level 2, Snap and Review data are kept in the git-ignored `data/` folder.

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

## Evaluation results

Every AI part is scored by a script against a frozen test set (locked by a SHA-256 hash), so later results can be compared on the same baseline. Reports (`report.md`, `report.json`) are committed next to the code.

| Part | Result | Test set |
|---|---|---|
| Hybrid search | **100% Hit@5**, MRR 0.788 | 42 held-out questions |
| Handwriting reading (CER) | **0.53 → 0.13** after OpenCV preprocessing | vision model, raw vs preprocessed, 20 images |
| Step checking | **36 / 38** steps judged correctly; no hint leaked a solution | 14 test cases |
| Automated tests | **274** pass, plus every sample checked against the API contract | `uv run pytest` |

> **Caveat.** The test sets are small and the handwriting images are synthetic, so these numbers show the pipeline works, not final accuracy. Re-measure on real photos (see above) before quoting them as results.

Planned: a results page for teachers that reads the same report files, and an automatic run on every code change.

## Cost

AI costs about **$0.44 per student per month** (a 30-student class with its teacher is about $13/month; 1,000 students about $436/month including story video). A student uses about 447k tokens a month, roughly $0.36 of LLM cost.

This is an estimate from prompt sizes and assumed usage; the app does not count tokens yet. Prices are from public pages, Oct 2026.

## Business model and market (proposed, to validate in pilots)

| Plan | Price | Includes |
|---|---|---|
| Teacher Free | EGP 0 | 1 class room, up to 30 students, AI tutor and quizzes, basic struggling-student view |
| Teacher Pro | EGP 199 / month | Unlimited class rooms, one-click lesson studio, video tools and exports |
| School & Centre | EGP 25 / student / month | Dashboards across every class, weekly parent reports, LMS and WhatsApp integration (main revenue) |

A month of Rafeeq per student costs less than one private group session (EGP 150–300). The beachhead is private learning centres and schools in Greater Cairo (about 2.2M of Egypt's 25.8M students are in private K-12 schools), then universities, training centres, online academies and corporate training. Growth path: teachers first through the free plan, then pilots with centres, then LMS/WhatsApp integrations and white-label versions.

**On-premise option** (for privacy and contract needs, not savings):

| Tier | Size | Estimated cost |
|---|---|---|
| Pilot | One school, up to ~300 students (RTX 5090, 32 GB) | $8–9k |
| Mid | 1,000–3,000 students (RTX PRO 6000, 96 GB) | $22–28k |
| Large | A whole school network | By quote |

Hosted AI costs about $436/month for 1,000 students, so a $22–28k server takes roughly 50–65 months to pay back. Hardware figures are estimates from public GPU prices and were not load-tested; Arabic answer quality with open models must be re-measured before promising it.

## Run the tests

```bash
uv run pytest
```

This also checks that every file in `contracts/samples/` matches the API contract in `contracts/openapi.yaml`.

## Deployment

Local development uses `uv sync` + `uvicorn`, or `docker compose` (app + Postgres). The same Docker image is deployed to Render with PostgreSQL + pgvector and the environment variables listed under [Configuration](#configuration).

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
docs/
├── architecture.md      # feature workflows, architecture and deployment diagrams
├── images/              # slides from the capstone presentation used in this README
└── knowledge/           # M2 evaluation reports (historical, written in the standalone project)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow and how the team is organized.

## Status

In active development for the NTI AI capstone.

## Team

A five-member team built this for the NTI AI course capstone, Education domain.

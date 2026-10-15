# Rafeeq web UI: handoff (branch `m1/web-ui`)

The web UI is served by the same FastAPI app as the API, at `/app`. It uses **only** the
HTTP routes in `contracts/openapi.yaml` (called in-process), so each teammate's real
implementation appears in the UI as soon as their route returns it. Nothing is faked: if a
route is missing, that section says "not connected yet"; if M3/M4 are not connected, the
Copilot labels its answer as a sample.

Stack: FastAPI + Starlette sub-app, Jinja2, HTMX, plain CSS, ~250 lines of vanilla JS.
Arabic first (RTL), English toggle (LTR), desktop first, works down to phone width.

## 1. Merge it

```
git switch main && git pull && git switch -c m1/web-ui
# copy the branch files in (same paths), then:
uv add jinja2 itsdangerous httpx python-multipart
uv add langgraph opencv-python-headless numpy        # optional: Copilot graph + photo clean-up
uv run python docs/frontend/apply_contract_edits.py   # M1 contract edits (stops safely if your file differs)
```

In the file that creates the FastAPI app, **after** all `include_router(...)` calls:

```python
from rafeeq.web import install
install(app)          # UI at /app; "/" redirects there if "/" is free
```

Add `storage/` to `.gitignore` (accounts, rooms and uploads live there in the demo).

Run: `RAFEEQ_WEB_SECRET=<long random string> uv run uvicorn <your.module>:app --reload`, open `http://127.0.0.1:8000/`.

Tests: `uv run pytest`. Expected here: 91 passed, 1 skipped (the skipped one runs the
journey through the real FastAPI router; it runs in the repo because FastAPI is installed).

### Old tests that must change (intended behaviour changes)

| Test | Old assumption | Now |
|---|---|---|
| login with any email | any credentials return `DEMO_USER` | real accounts. Demo: `sara@rafeeq.demo` / `mona@rafeeq.demo`, password `rafeeq-demo` |
| `test_snap_route` valid photo | returns the sample | needs a real image **and** a reader (M3 OCR or `LLM_API_KEY`), else 503. Non-image: 422 |
| `test_copilot_route` valid message | equals `copilot_reply.json` | a real routed reply. Assert `route` is in the enum and validate against `CopilotReply` (see `tests/test_copilot_supervisor.py`) |

Calls **without** an `Authorization` header still return the contract samples (rooms, files,
students, join, create), so stubs and sample-based tests keep working.

### Environment

| Variable | Default | Purpose |
|---|---|---|
| `RAFEEQ_WEB_SECRET` | random per start | session signing key (set it, or sessions end on restart) |
| `RAFEEQ_DATA_DIR` | `storage` | M1 data, uploads, snaps |
| `RAFEEQ_DEMO_ACCOUNTS` | `1` | seed and show the two demo accounts (they start with no rooms) |
| `RAFEEQ_ALLOW_SAMPLES` | `1` | `0` = Copilot says "not available" instead of labelled samples |
| `RAFEEQ_HTTPS_ONLY` | `0` | `1` behind HTTPS (secure cookie) |
| `RAFEEQ_API_BASE` | unset | call a separately deployed API instead of in-process |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | unset | OpenAI-compatible model for routing and photo reading |

## 2. Page map

| Student | Teacher |
|---|---|
| Log in / Create account (role, optional join code) | same |
| **My rooms**: empty state explains the join code, join form | **My rooms**: empty state, create room |
| Room > **Ask Rafeeq**: one Copilot box + attach photo | Room > **Overview**: what's happening / needs attention / next step, join code |
| Room > **Materials** (read only) + source page viewer | Room > **Materials**: upload, status, extracted topics |
| Room > **Practice**: pick topic, quiz, result | Room > **Students**: list, at-risk, draft a message |
| Room > **My progress**: weak topics, level, next step | Room > **Class insights**: questions, mistakes, level per topic, "prepare a review lesson" |
| | Room > **Lesson studio**: request, draft, revise / approve, download, past lessons |
| | **Profile & defaults** |

Removed from the old design: separate Planner, Inbox, Parent reports, Research, Evaluation,
Code-story and Presentation pages. Each is either merged above or needs a backend that does
not exist yet (section 6).

## 3. Journeys

- **New student**: register (no rooms) → empty state: "ask your teacher for the code" → join → Ask Rafeeq.
- **Ask**: question → Copilot routes it (ask / quiz / check_work / plan / clarify) → answer with source chips → chip opens the cited page.
- **Photo**: attach → `/snap` (OpenCV + M3 OCR or vision model) → student can **fix the text**, then chooses *Explain this* (`mode=ask`) or *Check my solution* (`mode=check_work`) → steps marked, box drawn on their own photo, a hint (never the full answer).
- **Closed loop (teacher)**: students ask → M3 answer carries `topic_id` → M1 records a QuestionEvent → M4/M5 aggregate → Class insights shows the topic and real questions → *Prepare a review lesson* opens the studio pre-filled with the topic and those questions → teacher reviews, revises, approves → files → students practise → M4 mastery change → Report shows the topic level.

## 4. Integration map and status

Connected = real today. Partial = the UI calls the contract route; real data arrives when the owner implements it (until then: sample or "not connected"). Unavailable = no contract route; not shown in the UI.

| UI action | Route | Owner | Status |
|---|---|---|---|
| Register / log in / log out | `POST /auth/register`, `POST /auth/login` | M1 | Connected |
| My rooms, create, join, join code | `GET/POST /rooms`, `POST /rooms/join`, `GET /rooms/{id}/join-code` | M1 | Connected |
| Upload, file list | `POST/GET /rooms/{id}/files` | M1 (+M2 status) | Partial: stored and listed; stays "Uploaded" until M2 registers `service.on_upload` |
| Students list | `GET /rooms/{id}/students` | M1 (+M4) | Partial: members and questions are real; mastery/status `null` until M4 |
| Ask Rafeeq | `POST /copilot/message` | M1 → M3/M4 | Partial: routing real; answers labelled sample until `clients.connect(...)` |
| Read photo | `POST /snap` | M1 + M3 | Partial: OpenCV real; reading needs M3 OCR or `LLM_API_KEY`, else honest 503 |
| Check my solution | `/copilot/message` `mode=check_work` | M3 | Partial |
| Topics, source page | `GET /rooms/{id}/topics`, `GET /sources/page` | M2 | Partial |
| Practice | `POST /quiz`, `POST /quiz/{id}/submit` | M4 | Partial |
| My progress | `GET /profile/summary`, `GET /next-step` | M4 | Partial |
| Overview, Insights | `GET /teacher/rooms/{id}/insights`, `/report` | M5 | Partial (M1 parts connected) |
| Draft a message | `POST /teacher/assistant` | M5 | Partial |
| Lesson studio | `POST/GET /teacher/sessions`, `/{sid}`, `/{sid}/decision`, `/{sid}/files/{name}` | M5 | Partial (contract as M5 defined it) |
| Profile & defaults | `GET/PUT /teacher/profile` | M5 | Partial |
| Go deeper (web, YouTube minutes, generated video), teacher inbox / verified answers, parent report, research, evaluation, notifications | none | see 6 | Unavailable |

### How each teammate plugs in

- **M2**: register the indexer: `from rafeeq.platform import service; service.on_upload(fn)` where `fn(room_id, file_id, path)`; report progress with `service.get_store().set_file_status(room_id, file_id, "indexing"|"ready"|"failed")`. Serve `/rooms/{room_id}/topics` and `/sources/page` (room_id as a **query** parameter, see 5).
- **M3**: `from rafeeq.copilot import clients`, then `clients.connect("tutor_answer", fn)`, `clients.connect("check_work", fn)`, `clients.connect("read_image", fn)` (signatures at the top of `rafeeq/copilot/clients.py`). Snapped photos are at `storage/snaps/{file_id}.jpg`.
- **M4**: `clients.connect("make_quiz", fn)`, `clients.connect("next_step", fn)`, `clients.connect("record_event", fn)`; serve `/quiz`, `/quiz/{id}/submit`, `/profile/summary`, `/next-step`.
- **M5**: nothing to change for the UI. `teacher_id` is M1's `user_id` (also sent as the token); `room_id` comes from M1 rooms.

## 5. Contract changes in this branch (M1, applied by the script; post in the group before merging)

E1 `/auth/login` 401 · E2 `/auth/register` 400/409 · E3 **new** `GET /rooms/{room_id}/join-code` → `JoinCode {join_code}` (sample `join_code.json`, registered in `tests/test_contracts.py`) · E4 `/snap` 422/503 · E5 `CopilotRequest.mode` optional `auto | ask | check_work` (default `auto`) · E6 `CopilotReply.payload` flags documented (`sample`, `unavailable`, `needs_photo`, `escalated`, `question_id`, `verified_by_teacher`) · E7 `RoomStudent.mastery` and `status` may be `null` · E8 schema `JoinCode`.
All are additive or widen a type; existing clients keep working.

### Issues found in the current contract (not changed here)

1. `RoomId` is declared `in: path`, but `/sources/page`, `/next-step` and `/profile/summary` have no `{room_id}` in the path, which is invalid OpenAPI. The UI sends `room_id` as a query parameter. Fix (M2, M4): add a `RoomIdQuery` parameter (`in: query`) and use it on those three paths.
2. The Copilot's quiz route returns `{quiz_id}` but there is no way to fetch a quiz by id, so the UI links to Practice instead (proposal P5).
3. `TutorRequest` has no `user_id`, and `TutorAnswer` has no `topic_id`, so nobody can create the QuestionEvent that insights need (P3).
4. M5 routes still take `teacher_id` "until M1's login exists". Login exists now; M5 can read the token.
5. `GET /teacher/sessions` lists only approved sessions, so an unapproved draft cannot be reopened later (P8).

## 6. Proposed contract extensions (not applied; owners decide)

```yaml
# P1 (M1 route, M3/M5 services): "Still confused? Go deeper"
/copilot/deeper:
  post:
    requestBody: { content: { application/json: { schema:
      { type: object, additionalProperties: false, required: [room_id, question, level],
        properties: { room_id: {type: string}, question: {type: string},
                      level: {type: string, enum: [web, video, generated_video]} } } } } }
    responses: { "200": { content: { application/json: { schema:
      { type: object, additionalProperties: false, required: [level, items],
        properties: { level: {type: string},
          items: { type: array, items: { type: object, additionalProperties: false, required: [title, url],
            properties: { title: {type: string}, url: {type: string}, excerpt: {type: string},
                          start_seconds: {type: integer}, end_seconds: {type: integer} } } } } } } } },
                 "503": { description: level not available } }

# P2 (M5 + M2): teacher inbox for unanswered questions; approved answers feed find_verified
/teacher/rooms/{room_id}/questions:
  get: { responses: { "200": { description: "{questions: [{question_id, text, asked_at, status: waiting|answered}]}" } } }
/teacher/rooms/{room_id}/questions/{question_id}/answer:
  post: { requestBody: "{answer: string}", responses: { "200": { description: "{question_id, status: answered}" } } }

# P3 (M3): TutorAnswer gets optional topic_id (lets M1 record QuestionEvent -> insights)
TutorAnswer.properties.topic_id: { type: string }

# P4 (M4): read access to events for M5's insights
/rooms/{room_id}/events: { get: { responses: { "200": { description: "{events: [QuestionEvent]}" } } } }

# P5 (M4): fetch a quiz by id (Copilot quiz route)
/quiz/{quiz_id}: { get: { responses: { "200": { content: { application/json: { schema: { $ref: Quiz } } } } } } }

# P6 (M5): parent report, teacher-controlled (copy / export; no parent login)
/teacher/rooms/{room_id}/students/{user_id}/parent-report:
  post: { responses: { "200": { description: "{text: string, language: ar|en}" } } }

# P7 (M5): research and YouTube moments for lesson prep (same item shape as P1)
/teacher/research: { post: { requestBody: "{query, kind: web|video}", responses: { "200": { description: "{items: [...]}" } } } }

# P8 (M5): list drafts waiting for review
GET /teacher/sessions?status=awaiting_approval

# P9 (tech team): evaluation runs
/eval/runs: { post: {}, get: {} }   # {run_id, grounded_rate, citation_correct, latency_ms_by_route, cases: [...]}

# P10 (M1): POST /auth/logout to revoke the token server-side
```

## 7. What was verified, and what remains

Verified here: 91 tests (contract samples; M1 service; Copilot routing and honesty rules;
photo reading rules; 30 web tests that drive real forms and fail if the UI sends a body
the contract rejects); every template compiles; a Playwright walkthrough of both journeys
in Arabic and English with no JS or server errors; pyflakes clean.

Not verified here: `rafeeq/platform/router.py` under real FastAPI (no FastAPI in the
sandbox). `tests/test_web.py::test_journey_through_real_fastapi_router` covers it and runs
in the repo. It was also reviewed line by line.

Remaining: proposals P1 to P10; chat history is in memory (lost on restart); demo storage
is a JSON file (swap `PlatformStore` for a database later; same methods); HTMX loads from
unpkg (vendor it into `rafeeq/web/static/` before deployment); logout does not revoke the
token yet (P10).


## 8. Teacher features merged from the `maryam-sayed` branch (M5)

Merged with a real `git merge` (her commits keep their history). Only `pyproject.toml` and `uv.lock` conflicted.

### Real class reports (no more sample data)

`GET /teacher/rooms/{room_id}/report` and `/insights` (already shown on `/app` Overview, Students and Class insights) are now computed by `rafeeq/teacher/class_report.py`:

| Input | Recorded by | Used for |
|---|---|---|
| Every student copilot message (text, time) | `rafeeq/platform/router.py` `copilot_message` (one added line) | Insights: questions grouped by topic, with an example |
| Quiz results (`question`, `chosen`, `correct` per answer) | **M4:** call `class_report.record_quiz_result(room_id, user_id, topic_id, answers)` inside `POST /quiz/{quiz_id}/submit` | Report: mastery per topic, students at risk; Insights: repeated mistakes; Students: each student's mastery and status |

- Stored in the teacher service's LangGraph store (Postgres when `DATABASE_URL` works) under `("class", room_id, ...)`. Last 30 days count.
- Topics: M2's `/rooms/{room_id}/topics` is not implemented, so a topic id is the topic name. Questions without a topic are grouped once (LLM, with the lecturer's session topics for that room as candidates; word overlap without an LLM).
- Mastery is a stand-in until M4's learner profile exists (recent quizzes weigh more; ≥ 0.7 on track, ≥ 0.5 watch, else struggling). When M4 lands, read mastery from M4 and keep the rest.
- `GET /rooms/{room_id}/students` now fills `mastery`/`status` from those results (still `null` without data).
- Demo data: `uv run python -m rafeeq.teacher.seed_class_demo <room_id> [--add-students 4] [--clear]` (needs Postgres).

### Video tools inside the `/app` Lesson studio

| Feature | `/app` page | M5 API |
|---|---|---|
| YouTube videos that open at the second answering a question, Previous / Next | page `/app/video` (Video Discovery Studio) | `GET /teacher/videos/search` (no key) |
| Animated video from a description or image (paid, QwenCloud Wan) | page `/app/media` (Visual & Media Studio; polls every 5 s) | `POST /teacher/videos/generate`, `GET /teacher/videos/generate/{task_id}` (`DASHSCOPE_API_KEY`) |
| Presenter image used as the first frame when no image is uploaded | Profile page | `POST /teacher/avatar`, `GET /teacher/avatar/{teacher_id}` |

All are in `contracts/openapi.yaml` with samples. `TeacherProfile` gained the optional `avatar_filename`.

### Also from her branch

Video discovery over approved lessons' slide notes (Lab 4, `lecturer/video_discovery.py`), text to speech lab (`labs/lab1_tts/`), fail-fast Postgres pool (`lecturer/memory.py`). The older `/teacher` dashboard still works.

New environment variables: see `.env.example` (`DASHSCOPE_API_KEY`, `QWEN_BASE_URL`, `WAN_T2V_MODEL`, `WAN_I2V_MODEL`).
Also fixed: `.gitignore` had been saved as UTF-16 so `storage/` was never ignored, and `storage/platform/platform.json` (demo accounts and tokens) had been committed. It is no longer tracked.

### Teacher library (assets) — phases 1-3

One record per saved or generated item, owned by one teacher. Records are in the same LangGraph store as the profile (`("lecturer", <id>, "assets")`), files stay on disk (`outputs/teacher/<session>/…` for Lesson Studio files, `outputs/assets/<teacher>/<asset_id>/…` for videos; `TEACHER_ASSET_DIR` changes the second). Code: `rafeeq/teacher/assets.py`. No database migration.

| What | How it reaches the library |
|---|---|
| `.pptx`, code, quiz files of an approved session | `final_agent` registers them (`origin: ai_generated`). Sessions finished earlier are added the first time `GET /teacher/assets` is called |
| Animated video | `POST /teacher/videos/generate` (with the teacher token) creates a `pending` asset. A repeat of the same request returns the same job (no second payment). When the job succeeds, the clip is downloaded and kept (`file_url`); the provider link would expire after 24 h |
| YouTube moment | "احفظ هذه اللحظة في مكتبتي" -> `POST /teacher/assets` (`origin: external`) |

API (Bearer token of a teacher is always required): `GET /teacher/assets`, `POST /teacher/assets`, `GET /teacher/assets/{id}`, `GET /teacher/assets/{id}/file`. `/app` plays saved videos from `GET /app/assets/{id}/file`.

Authorization: `/teacher/*` routes now check the Bearer token when one is sent (another teacher's `teacher_id` -> 403). Without the header they behave as before (old `/teacher` pages). Still open: `GET /teacher/sessions/{id}` and its decision route do not check who owns the session.

The studios, Library page and publishing are described in the next section.

### Separate studios (one page each, connected by ids in the URL)

Lesson Studio no longer holds the other tools. Each studio is its own route in the same layout (sidebar group "Studios"), and work moves between them through the library and the URL (`?room=&session=&asset=&theme=`).

| Page | Route | Purpose | API it calls |
|---|---|---|---|
| Lesson Studio | `/app/studio` (redirects to the last room) and `/app/rooms/{id}/studio` | Write a lesson with the agents; when it is approved it offers "next steps" (slides, video, media, course) carrying the session id | `/teacher/sessions…` (unchanged) |
| Video Discovery | `/app/video` | Search YouTube moments; save a moment or the cover picture to the library | `/teacher/videos/search`, `/teacher/assets`, `/teacher/media/thumbnail` |
| Presentation Studio | `/app/presentations`, `/app/presentations/{deck_id}` | Edit slides, choose a theme (built-in or extracted from an uploaded .pptx), add library pictures, draw a picture (LLM writes the prompt, diffusion draws it), export a new .pptx version | `/teacher/decks…`, `/teacher/themes…`, `/teacher/visuals/{id}` |
| Visual & Media Studio | `/app/media` | Upload and edit images (crop, rotate, flip, resize, brightness, contrast, grayscale), trim/mute/volume/extract audio from video and audio (ffmpeg), text to speech, generate animated video. Edits are new versions; originals stay | `/teacher/media…`, `/teacher/videos/generate…` |
| Story Video Studio | `/app/story` | A topic becomes one connected ~30 s video: LLM plans scenes, each scene continues from the last frame of the previous one (Wan), ffmpeg joins them (+ optional narration). Saved as a library video. See `docs/media-pipeline.md` | `/teacher/stories`, `/teacher/stories/{id}` |
| My Learning Platform | `/app/learning`, `/app/learning/{course_id}` | Courses and lessons built from library items; publish a snapshot to a room | `/teacher/courses…` |
| Student view | `/app/rooms/{id}/learn` | Published courses only | `/rooms/{id}/courses`, `…/assets/{id}/file` |
| My Library | `/app/library` | Hub: lessons, decks, courses, themes and every asset, with "use in…" actions | `/teacher/assets`, `/teacher/decks`, `/teacher/courses` |

**Data model (no new database, no migration).** Same LangGraph store as before: `("lecturer", <id>, "assets"|"decks"|"courses")`; published courses are copied to `("class", <room_id>, "courses")` so students see a snapshot, and the teacher's later edits show as `unpublished_changes` until they publish again. Asset kinds now include `video` (uploaded) and `theme`. A deck is created automatically when a lesson is approved (`final_agent`), or on first listing for older lessons. Code: `rafeeq/teacher/{themes,decks,courses,media_tools,visuals,studio_routes}.py`, `rafeeq/web/views_studios.py`.

**Safety.** Every route needs a teacher token (students 403); a teacher only sees their own decks, courses and assets; pictures in a deck must be the teacher's own images; students are served only files in the published snapshot of a room they belong to. Repeating an edit, upload, speech or picture request returns the existing result instead of doing the work (and paying) twice.

**Limits to know.**
- Needs `DASHSCOPE_API_KEY` for pictures/video (503 otherwise). The image model name and endpoint are unverified against the live service.
- Video and audio editing need the `ffmpeg` binary on the server (503 with a clear message otherwise). Image editing uses Pillow only.
- Text to speech needs the `edge-tts` package and internet.
- No exact-frame screenshot of a YouTube video: the cover picture is saved instead (or upload your own screenshot).
- Saved videos are served without range requests (seeking depends on the browser).
- `GET /teacher/sessions/{id}` and its decision route still lack ownership checks. The old `/teacher` pages remain.

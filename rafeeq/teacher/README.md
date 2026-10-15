# Teacher (M5)

The teacher page at `/teacher` and the lecturer assistant behind it.

## What a teacher does

1. Saves a profile once (name, title, subjects, default level/language). The name + title is printed on every generated slide.
2. Types a topic, ticks slides / code / quiz, optionally links a room → **جهّز المسودة**.
3. Reviews the draft on the page, then **approves** (files are made: `.pptx`, code file, `quiz.json`, `quiz.md`) or **asks for changes** (up to `MAX_REVISIONS` times).
4. Finished sessions stay under «جلساتي» with their download links. The assistant remembers the teacher's choices and change requests and shows that list on the profile card.

## Layout

| Path | What |
|---|---|
| `routes.py` | JSON API (M5 section of `contracts/openapi.yaml`) + the HTMX pages |
| `service.py` | Starts the assistant inside the Rafeeq app (called from `main.py`'s lifespan) |
| `profile.py` | Teacher profile, stored next to the assistant's memory |
| `lecturer/` | The LangGraph pipeline, ported from `Lecturer_Multi_Agentic_Assistant` |
| `lecturer/nodes/` | orchestrator, research, slides, code, quiz, draft, human_approval, final |
| `lecturer/mcp_servers/` | `code_runner` (runs generated code), `pptx` (builds the deck), Tavily client |
| `../web/templates/teacher/` | `dashboard.html`, `profile_form.html`, `_review.html`, `session.html` |

## Run it

```bash
uv lock && uv sync            # once, after pulling this change (new dependencies)
docker compose up -d db       # Rafeeq's Postgres; also stores the assistant's checkpoints + memory
cp .env.example .env          # then fill LLM_PROVIDER + that provider's key and model
uv run uvicorn rafeeq.main:app --reload
```

Open http://127.0.0.1:8000/teacher.

- No LLM key: Rafeeq still starts; only the generate button returns an error message.
- No Postgres: Rafeeq still starts with in-memory storage and the page shows a warning.

## Not done yet

- Login: until M1's auth exists the teacher is remembered by a cookie (`rafeeq_teacher`) and the API takes `teacher_id`.
- Insights / class report show **sample data** until M4 stores question events and student profiles.
- `POST /teacher/assistant` (draft a reply to students) is still only in the contract.
- Generation runs inside the request (1–2 minutes). Fine for the demo; a background job queue is the next step for many teachers at once.

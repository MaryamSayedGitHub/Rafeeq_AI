# Rafeeq: team guide

How to run Rafeeq on your laptop, use the demo data, and what changed in the latest update.

---

## 1. Setup (once)

You need Python, `uv` and git. Then, in PowerShell:

```powershell
git clone https://github.com/MaryamSayedGitHub/Rafeeq.git
cd Rafeeq
uv sync --extra knowledge
Copy-Item .env.example .env
```

Open `.env` and fill in these lines. **Each line must appear only once.** If a name appears twice, the
last one wins, and that is exactly what broke uploads before.

```
LLM_API_KEY=<your own Groq key from console.groq.com>
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=qwen/qwen3.8-27b
RAFEEQ_KNOWLEDGE=1
RAFEEQ_KNOWLEDGE_TUTOR=1
GEMINI_API_KEY=<your own key from aistudio.google.com>
GEMINI_MODEL=gemini-3.5-flash-lite
RAFEEQ_WEB_SECRET=<any long random text>
```

- **Use your own Groq key.** The free limits are per key, so sharing one key slows everyone down.
- **The tutor (answers from the PDF) uses Gemini**, not Groq. Without `GEMINI_API_KEY` and
  `GEMINI_MODEL`, Rafeeq says the feature isn't available. Everything else uses the `LLM_*` lines.
- **`RAFEEQ_WEB_SECRET`** keeps you signed in and keeps parent-report links working after a restart.
- **`.env` is never pushed to GitHub** because it holds your key. Only `.env.example` is in the repo.

Check that the model works:

```powershell
uv run python -m rafeeq.copilot.check_llm
```

You should see **8/8**. If you see `?` or errors, the key or the model name is wrong, or you hit the
rate limit. Wait a minute and try again.

Run the tests (optional, takes a few minutes):

```powershell
uv run pytest
```

## 2. Run the app

```powershell
uv run uvicorn rafeeq.main:app --reload
```

Open **http://localhost:8000/app**. This is the main app: use it for everything, including the demo.

**The first PDF you upload is slow.** The search model (`BAAI/bge-m3`, about 2 GB) downloads
automatically from Hugging Face the first time. You need internet for that, and only once.

## 3. Your first room

Nothing is shared between laptops. Accounts, rooms and uploaded files live in your own `storage/` and
`data/` folders, so everyone starts with an empty app.

1. **Sign up as a teacher** in `/app` and create a room.
2. **Upload a PDF** from the room's Files page.
3. **Wait until the file shows "جاهز".** Refresh the page to check. Topics are taken from the PDF itself.
4. **Share the room's join code** with a student account so it can join.

## 4. Demo data

This fills one room with 5 students, two weeks of quiz history, questions to Rafeeq and 2 questions
waiting in the teacher's inbox, so the teacher pages are not empty in a demo.

1. **Do section 3 first.** The room needs a PDF that is "جاهز".
2. **Copy the room id** from the browser address bar: it is the part after `/rooms/` when the room is open.
3. **Add it to `.env`:** `RAFEEQ_DEMO_ROOM=<room id>`.
4. **Restart the server.** Stop it with Ctrl+C and start it again. The data is only added at startup.
5. **Check the terminal** for `demo seed: room ... filled with 5 demo students`.

If you see a different message:

| Message | Meaning |
|---|---|
| `room ... does not exist` | The room id is wrong. |
| `room ... has no topics yet` | The PDF isn't ready yet. Wait for "جاهز", then restart. |
| `demo seed: failed ...` | Something went wrong. Send the error to M1. |

The seed runs **once per room**. To seed again, create a new room and put its id in `.env`.

### Demo accounts

**Your own teacher account** (the one that created the room) is the one that sees the whole class:
the overview, the Students page with parent reports, and the inbox.

The 5 demo students all use the password **`rafeeq-demo`**. Each student sees only their own data.

| Student | Email | Profile |
|---|---|---|
| عمر خالد | omar@rafeeq.demo | weak on purpose, shows as at risk |
| سارة محمود | sara.m@rafeeq.demo | strongest |
| يوسف أحمد | youssef@rafeeq.demo | weak in the second topic |
| مريم علي | mariam@rafeeq.demo | strong |
| كريم حسن | karim@rafeeq.demo | average |

### Suggested demo path

1. **Teacher:** open the room's overview and show the class level per topic and students at risk.
2. **Teacher → Students:** open **Omar's** "تقرير وليّ الأمر", then copy the parent link.
3. **Open the parent link in a private window.** It opens with no account.
4. **Teacher → أسئلة الطلاب:** answer a question, using "اقترح ردًا" to get an AI draft first.
5. **Sign in as Omar** and show:
   - "ماذا أذاكر الآن؟"
   - "اشرح لي فكرة الدرس الأخير ببساطة"
   - a practice quiz
   - the teacher's reply on his study page
6. **Back as the teacher:** the quiz Omar just took now shows in the class data.

"Repeated mistakes" only shows real wrong answers, so take a quiz live during the demo.

## 5. What changed (simple words)

**Fixes**
- **Uploading PDFs works**, and teachers can delete a file.
- **Room topics are real.** Topics come from the uploaded PDF, not a fixed list about the LLM course.
- **Quizzes are made from the room's PDF.** No more "حدث خطأ".
- **"ماذا أذاكر الآن؟" uses the student's real weak topics**, not sample text.
- **"Explain the last lesson" works.** It explains the newest uploaded file.
- **Rafeeq understands what you're asking more reliably** (explain / quiz / what next), and retries
  when the model service is busy.
- **Tests no longer read your `.env`**, so your settings can't make them fail.

**New**
- **Teacher analysis on real data.** Quizzes and questions reach the teacher's class view, grouped by
  the room's topics. The `/teacher` dashboard shows real rooms, with no "sample data" label.
- **Parent report.** A printable weekly Arabic report per student, with a private link for the parent
  that works without an account for 14 days. The summary comes from fixed templates, so the AI can't
  invent anything.
- **Teacher inbox (أسئلة الطلاب).**
  - When Rafeeq can't answer, the question goes to the teacher.
  - The teacher can ask for an AI draft, edit it and send it.
  - The student sees the reply on their study page.
  - If anyone asks the same question later, Rafeeq answers with the teacher's approved reply.
- **Teacher assistant (`/teacher/assistant`) works now**, using the class's real data.
- **Snap is better.** It fixes phone rotation, finds and straightens the page, and asks for a retake
  when the photo isn't a study page.
- **Logins expire after 30 days.**
- **Demo data** (section 4).

New API routes were added to `contracts/openapi.yaml`. The full list, with which module owns each
change, is in `docs/INTEGRATION-UPDATE.md`. **M2, M4 and M5: please review the parts that touch your
module.**

## 6. Still missing

- **M3:** checking a student's work and reading handwriting. M3 is delivering this.
- **One login for `/teacher` and `/app`.** They still sign in separately. Use `/app` in the demo; it
  links to the teacher studios.
- **Not planned for this version:** WhatsApp and LMS integration, deployment and fine-tuning.

## 7. Notes and known limits

- **Postgres (`DATABASE_URL`).** The teacher's class data and the inbox use Postgres if it is running.
  If not, they are kept **in memory and disappear when the server restarts**. Accounts, rooms, files and
  quiz history are kept on disk either way. Without Postgres:
  - The demo room's class data and inbox are empty after a restart, and the seed won't run again for
    that room. **Create a new room and update `RAFEEQ_DEMO_ROOM` before the demo.**
  - **Or just don't restart the server between seeding and presenting.**
  - **Easiest real fix:** if you have Docker, run `docker compose up -d db` in the project folder. It
    starts the Postgres the repo already describes, matching the default `DATABASE_URL`.
- **Two mastery numbers.** The student's practice page (M4) and the teacher's view (M5) calculate
  mastery differently, so they can show slightly different percentages for the same topic.
- **Groq free limits.** Many messages in a row can hit the rate limit. Rafeeq retries, but if it keeps
  failing, wait a minute.
- **Don't commit local files.** That means `.env`, `storage/`, `data/` and `m4_dev.db`. They're in
  `.gitignore`; check `git status` before you commit anyway.
- **Work on a branch and open a pull request.** Don't push straight to `main`.

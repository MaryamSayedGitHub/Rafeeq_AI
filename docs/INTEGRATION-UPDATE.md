# Integration update (M1, October 2026)

What changed across modules so the student loop and the teacher views run on real data.
Every change has tests; owners please review the parts that touch your module.

## Contract changes (need the group's OK)

| Route | Owner | Why |
|---|---|---|
| `DELETE /rooms/{room_id}/files/{file_id}` | M1 | teachers can remove material; M2 drops it from the index |
| `GET /teacher/rooms/{room_id}/students/{user_id}/parent-report` | M5 | weekly parent report + share token |
| `GET /parent-report/{token}` | M5 | the shared link, no account |
| `GET /rooms/{room_id}/inbox`, `POST .../{question_id}/answer`, `POST .../{question_id}/draft` | M5 | questions the tutor could not answer |
| `POST /teacher/assistant` | M5 | was in the contract, now implemented |

New schemas: `ParentReport`, `InboxQuestion`, `InboxQuestions`, `InboxAnswerRequest`.

## By module

**M1 (platform, Copilot, Snap)**
- M4's routes are mounted in `main.py`; the Copilot connects to M4 (`next_step`, `record_event`) and M2 (`overview`, `topic_of`) and M5 (`escalate`, `find_verified`).
- Router: keywords first, then the LLM, then rules; retries on 429/503; `uv run python -m rafeeq.copilot.check_llm`.
- "Explain the lesson" questions get an explanation of the newest file; tutor answers count towards the student's profile (topic from the cited pages).
- Snap: EXIF rotation, page detection + perspective warp; `other` photos ask for a retake.
- Files can be deleted; login tokens expire after 30 days; tests ignore the developer's `.env` (`tests/conftest.py`).
- Demo data: `RAFEEQ_DEMO_ROOM=<room_id>` seeds one room once (`rafeeq/demo_seed.py`).

**M2 (knowledge)**
- Topics are learned per room from the uploaded material (`topics/room_topics.py`), on by default (`RAFEEQ_KNOWLEDGE_ROOM_TOPICS`).
- New service methods: `remove_file`, `room_chunks`, `overview`, `topic_for_citations`; `FAISSVectorStore.remove_document`.

**M4 (quiz, profile)**
- Quizzes use the room's own material from M2 (`deps.room_chunks`); the dev book is only a fallback.
- `compatible` LLM provider (uses `LLM_*`), chosen automatically when `M4_LLM_PROVIDER` is not set; a missing key is a 503.
- Submitted quizzes feed M5's class report; `copilot_hooks.py` serves the Copilot.

**M5 (teacher)**
- Class report/insights use the room's M2 topic ids; questions are recorded with the answer's topic.
- `/teacher` dashboard shows real rooms and real class data.
- Parent report (`parent_report.py`), inbox and assistant (`inbox.py`).

## Not done (by agreement)
M3's tutor, check-work and handwriting (M3 delivers); LMS and WhatsApp; deployment; fine-tuning;
one login for `/teacher` and `/app` (use `/app`, which links to the teacher studios).

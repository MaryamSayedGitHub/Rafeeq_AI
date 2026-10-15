# Contributing

This is a student team's internal workflow. If you're not on the team, the [README](README.md) has what you need instead.

## Who owns what

| Folder | Owner | What's inside |
|---|---|---|
| `rafeeq/platform`, `copilot`, `snap`, `web`, `main.py`, root config | Member 1 | Login, rooms, files, database, the Copilot router, Snap & Ask |
| `rafeeq/knowledge` | Member 2 | Indexing course material, search with citations |
| `rafeeq/tutor` | Member 3 | Grounded answers, step-by-step mode, handwriting check |
| `rafeeq/quiz` | Member 4 | Quiz generation and grading, the student profile |
| `rafeeq/teacher` | Member 5 | Teacher dashboard, reports, LMS/WhatsApp integrations |
| `contracts/`, `tests/` | Shared | The API agreement between all five parts, and its checks |

## Workflow

1. `git switch main && git pull`
2. `git switch -c m<your-number>/short-name`
3. Work only inside your own folder and your own test file.
4. `uv run pytest` must pass before you push.
5. Open a pull request and merge with **Squash and merge**.
6. Changing a field in `contracts/` needs a pull request every affected member approves.

## Rules

- Never commit `.env`.
- Don't wait on another member's part — use `load_sample()` from `rafeeq/samples.py` against `contracts/samples/`.
- Merge only when `uv run pytest` passes.
- Reply in the group chat within 2 hours, within 15 minutes for a message starting `BLOCKED:`.

Full guide: `Rafeeq_Team_Guide.pdf` in the team chat.
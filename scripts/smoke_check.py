"""End-to-end check of a RUNNING Rafeeq server: every part, the way a teacher and a student use it.

    # terminal 1
    uv run uvicorn rafeeq.main:app
    # terminal 2
    uv run python scripts/smoke_check.py                      # makes its own small lesson PDF
    uv run python scripts/smoke_check.py --pdf data/lesson.pdf --question "What is photosynthesis?"

It creates a NEW teacher, a NEW student and a NEW room each run (nothing existing is touched), then:
platform -> upload + indexing (M2) -> topics/search (M2) -> Copilot ask/quiz/plan (M1 + M3) -> quiz (M4)
-> Snap photo + check work (M1 + M3) -> teacher report, inbox, insights, parent report (M5).

Every step prints PASS / WARN / FAIL / SKIP and the reason. The full result is saved to results/smoke_check.json
(no keys or passwords in it), so it can be shared as is.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[dict] = []
COLORS = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m", "SKIP": "\033[90m", "INFO": "\033[36m"}

LESSON = [
    "Photosynthesis",
    "Photosynthesis is the process plants use to turn light energy into chemical energy.",
    "It happens in the chloroplasts, which contain the green pigment chlorophyll.",
    "The plant takes in carbon dioxide and water, and with sunlight it makes glucose and releases oxygen.",
    "The light-dependent reactions happen in the thylakoid membranes and produce ATP and NADPH.",
    "The Calvin cycle happens in the stroma and uses ATP and NADPH to fix carbon dioxide into sugar.",
    "Cellular respiration",
    "Cellular respiration releases the energy stored in glucose. It happens in the mitochondria.",
    "Glucose and oxygen are used, and carbon dioxide, water and ATP are produced.",
    "Glycolysis happens in the cytoplasm and splits glucose into two pyruvate molecules.",
]


def report(part: str, step: str, status: str, detail: str = "") -> None:
    RESULTS.append({"part": part, "step": step, "status": status, "detail": detail[:400]})
    color = COLORS.get(status, "") if sys.stdout.isatty() else ""
    reset = "\033[0m" if color else ""
    print(f"{color}{status:<4}{reset}  [{part}] {step}" + (f"  -- {detail[:200]}" if detail else ""))


class Api:
    def __init__(self, base: str, timeout: float):
        self.base, self.timeout = base.rstrip("/"), timeout

    def call(self, method: str, path: str, token: str | None = None, **kw) -> tuple[int, object]:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.request(method, self.base + path, headers=headers, timeout=self.timeout, **kw)
        try:
            body = r.json()
        except ValueError:
            body = r.text[:300] if "text" in r.headers.get("content-type", "") else f"<{len(r.content)} bytes>"
        return r.status_code, body


def env_names() -> dict:
    """Which keys are filled in .env (names only, never the values)."""
    path = ROOT / ".env"
    if not path.is_file():
        return {}
    seen: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        seen[k.strip()] = bool(v) and not v.lower().startswith(("put-your", "aiza..."))   # the last line wins, like python-dotenv
    return seen


def make_pdf() -> bytes | None:
    try:
        import pymupdf as fitz  # PyMuPDF, already a project dependency
    except ImportError:
        return None
    doc = fitz.open()
    for start in (0, 6):
        page = doc.new_page()
        y = 72
        for i, line in enumerate(LESSON[start:start + 6]):
            page.insert_text((60, y), line, fontsize=16 if i == 0 else 11)
            y += 30
    data = doc.tobytes()
    doc.close()
    return data


def make_photo() -> bytes | None:
    """A white page with three lines of 'handwritten' algebra (a big font), slightly rotated, on a grey table."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    page = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(page)
    font = None
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            font = ImageFont.truetype(name, 64)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    for i, line in enumerate(["2x + 6 = 14", "2x = 20", "x = 10"]):
        draw.text((80, 80 + i * 150), line, fill=(20, 20, 60), font=font)
    table = Image.new("RGB", (1200, 900), (110, 110, 110))
    table.paste(page.rotate(4, expand=True, fillcolor=(110, 110, 110)), (120, 100))
    buf = io.BytesIO()
    table.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def run(a) -> None:
    api = Api(a.base, a.timeout)
    tag = uuid.uuid4().hex[:6]

    # ---------------------------------------------------------------- 0. server and settings
    try:
        code, body = api.call("GET", "/health")
        report("server", "GET /health", "PASS" if code == 200 else "FAIL", f"{code} {body}")
    except requests.RequestException as e:
        report("server", "GET /health", "FAIL", f"server not reachable at {a.base}: {type(e).__name__}. Start it with: uv run uvicorn rafeeq.main:app")
        return
    env = env_names()
    if not env:
        report("settings", ".env", "WARN", "no .env file next to the project: copy .env.example to .env and fill the keys")
    else:
        for name, why in [("LLM_API_KEY", "Copilot router, M3 tutor, quiz generation, Snap reader"),
                          ("RAFEEQ_KNOWLEDGE", "M2 knowledge base (must be 1)"),
                          ("GROQ_API_KEY", "M5 teacher assistant drafts"),
                          ("TAVILY_API_KEY", "Level-2 web research (optional)"),
                          ("GOOGLE_API_KEY", "voice / video discovery (optional)"),
                          ("DATABASE_URL", "M5 teacher data kept in Postgres (optional)")]:
            ok = env.get(name, False)
            optional = "optional" in why
            report("settings", name, "PASS" if ok else ("SKIP" if optional else "WARN"), ("set" if ok else "empty") + f": {why}")

    # ---------------------------------------------------------------- 1. accounts and room (M1)
    pw = "smoke-" + tag
    code, t = api.call("POST", "/auth/register", json={"name": f"Smoke Teacher {tag}", "email": f"t_{tag}@smoke.test", "password": pw, "role": "teacher"})
    if code != 200:
        report("platform", "register teacher", "FAIL", f"{code} {t}")
        return
    report("platform", "register teacher", "PASS", t["user_id"])
    teacher = t["token"]
    code, body = api.call("POST", "/auth/login", json={"email": f"t_{tag}@smoke.test", "password": pw})
    report("platform", "login", "PASS" if code == 200 and body.get("token") else "FAIL", str(code))
    code, room = api.call("POST", "/rooms", teacher, json={"name": f"Smoke room {tag}"})
    if code != 200:
        report("platform", "create room", "FAIL", f"{code} {room}")
        return
    room_id, join = room["room"]["room_id"], room["join_code"]
    report("platform", "create room", "PASS", f"{room_id} join code {join}")
    code, s = api.call("POST", "/auth/register", json={"name": f"Smoke Student {tag}", "email": f"s_{tag}@smoke.test", "password": pw,
                                                        "role": "student", "join_code": join})
    if code != 200:
        report("platform", "register student with join code", "FAIL", f"{code} {s}")
        return
    student = s["token"]
    report("platform", "register student with join code", "PASS", s["user_id"])
    code, rooms = api.call("GET", "/rooms", student)
    report("platform", "student sees the room", "PASS" if code == 200 and any(r["room_id"] == room_id for r in rooms.get("rooms", [])) else "FAIL", str(code))

    # ---------------------------------------------------------------- 2. material (M1 upload -> M2 indexing)
    pdf = Path(a.pdf).read_bytes() if a.pdf else make_pdf()
    question = a.question or ("What happens in the Calvin cycle?" if not a.pdf else "")
    indexed = False
    if pdf is None:
        report("knowledge", "upload PDF", "SKIP", "no --pdf given and PyMuPDF is missing to make one")
    else:
        name = Path(a.pdf).name if a.pdf else f"smoke_lesson_{tag}.pdf"
        code, up = api.call("POST", f"/rooms/{room_id}/files", teacher, files={"file": (name, pdf, "application/pdf")})
        if code not in (200, 202):
            report("knowledge", "upload PDF", "FAIL", f"{code} {up}")
        else:
            report("knowledge", "upload PDF", "PASS", f"{up.get('file_id')} status {up.get('status')}")
            status, start = up.get("status"), time.time()
            while status in ("uploaded", "indexing") and time.time() - start < a.index_wait:
                time.sleep(3)
                print(f"      ... {status} ({int(time.time() - start)}s)", end="\r", flush=True)
                _, files = api.call("GET", f"/rooms/{room_id}/files", teacher)
                status = next((f["status"] for f in files.get("files", []) if f["file_id"] == up.get("file_id")), status)
            took = int(time.time() - start)
            if status == "ready":
                indexed = True
                report("knowledge", "indexing (parse, chunk, embed, FAISS + BM25)", "PASS", f"ready after {took}s")
            elif status == "uploaded":
                report("knowledge", "indexing", "FAIL", "the file stays 'uploaded': the knowledge base is off. Set RAFEEQ_KNOWLEDGE=1 in .env "
                       "(only once in the file), run `uv sync --extra knowledge`, restart the server")
            elif status == "failed":
                report("knowledge", "indexing", "FAIL", "status 'failed': see the server log (embedder download? scanned PDF with no text?)")
            else:
                report("knowledge", "indexing", "WARN", f"still '{status}' after {took}s (the first run downloads bge-m3, ~2.3 GB). Re-run with --index-wait 900")

    if indexed:
        code, tp = api.call("GET", f"/rooms/{room_id}/topics", student)
        topics = tp.get("topics", []) if isinstance(tp, dict) else []
        report("knowledge", "room topics", "PASS" if code == 200 and topics else "WARN",
               f"{code} {len(topics)} topics: " + ", ".join(t.get("name", "") for t in topics[:5]))
        if question:
            code, sr = api.call("POST", "/search", student, json={"room_id": room_id, "query": question, "top_k": 3})
            chunks = sr.get("chunks", []) if isinstance(sr, dict) else []
            report("knowledge", "hybrid search", "PASS" if code == 200 and chunks else "FAIL",
                   f"{code} {len(chunks)} passages, first: {chunks[0]['file_name']} p.{chunks[0]['page']}" if chunks else f"{code} {sr}")

    # ---------------------------------------------------------------- 3. Copilot (M1 router + M3 tutor + M4 + M5 hooks)
    def copilot(text: str, mode: str = "auto", file_id: str | None = None):
        body = {"room_id": room_id, "text": text, "mode": mode, **({"file_id": file_id} if file_id else {})}
        return api.call("POST", "/copilot/message", student, json=body)

    if question:
        t0 = time.time()
        code, out = copilot(question)
        took = time.time() - t0
        if code != 200:
            report("copilot", "ask about the material", "FAIL", f"{code} {out}")
        else:
            p = out.get("payload", {})
            if out.get("route") != "ask":
                report("copilot", "ask about the material", "WARN", f"routed to '{out.get('route')}' instead of 'ask'")
            elif p.get("unavailable"):
                report("copilot", "ask about the material", "FAIL", "the tutor is not connected (payload.unavailable)")
            elif p.get("sample"):
                report("copilot", "ask about the material", "FAIL", "answered from the contract SAMPLE, not your material (LLM key / knowledge base?)")
            elif p.get("citations"):
                report("copilot", "ask about the material", "PASS", f"{took:.1f}s, {len(p['citations'])} citations, grounded={p.get('grounded')}: {out.get('reply', '')[:120]}")
            else:
                report("copilot", "ask about the material", "WARN", f"{took:.1f}s, no citations (grounded={p.get('grounded')}, notice={p.get('notice', '')}): {out.get('reply', '')[:120]}")
    for text, want in [("اختبرني", "quiz"), ("أذاكر إيه النهارده؟", "plan")]:
        code, out = copilot(text)
        route = out.get("route") if isinstance(out, dict) else None
        report("copilot", f"route '{text}'", "PASS" if code == 200 and route == want else "FAIL", f"{code} route={route} reply={str(out.get('reply', ''))[:80] if isinstance(out, dict) else out}")
    code, out = copilot("ما هي عاصمة اليابان؟")
    escalated_id = None
    if code == 200:
        p = out.get("payload", {})
        escalated_id = p.get("question_id") if p.get("escalated") else None
        from_web = bool(p.get("web_sources"))
        ok = p.get("grounded") is False or from_web
        how = "answered from the web (M3, Tavily key set)" if from_web else ("sent to the teacher" if p.get("escalated") else "")
        report("copilot", "question outside the material", "PASS" if ok else "WARN",
               f"{how} grounded={p.get('grounded')} escalated={p.get('escalated')} reply={out.get('reply', '')[:100]}")

    # ---------------------------------------------------------------- 4. quiz and profile (M4)
    if indexed or a.pdf is None:
        t0 = time.time()
        code, quiz = api.call("POST", "/quiz", student, json={"room_id": room_id, "count": 3})
        if code != 200:
            report("quiz", "generate quiz from the room's material", "FAIL" if indexed else "WARN", f"{code} {quiz}")
        else:
            qs = quiz.get("questions", [])
            report("quiz", "generate quiz from the room's material", "PASS" if len(qs) == 3 else ("WARN" if qs else "FAIL"),
                   f"{time.time() - t0:.1f}s, {len(qs)} of 3 questions: {qs[0]['text'][:80] if qs else ''}")
            if qs:
                answers = [q["correct_answer"] for q in qs[:-1]] + [next(c for c in qs[-1]["choices"] if c != qs[-1]["correct_answer"])]
                code, res = api.call("POST", f"/quiz/{quiz['quiz_id']}/submit", student, json={"answers": answers})
                ok = code == 200 and res.get("score") == len(qs) - 1
                report("quiz", "submit (all right but the last)", "PASS" if ok else "FAIL", f"{code} score={res.get('score') if isinstance(res, dict) else res}/{len(qs)}")
        code, summ = api.call("GET", f"/profile/summary?room_id={room_id}", student)
        report("quiz", "student profile summary", "PASS" if code == 200 else "FAIL", f"{code} {str(summ)[:150]}")
        code, nxt = api.call("GET", f"/next-step?room_id={room_id}", student)
        report("quiz", "next step", "PASS" if code == 200 else "WARN", f"{code} {str(nxt)[:150]}")

    # ---------------------------------------------------------------- 5. Snap (M1 clean -> M3 lines + vision model) and check work
    photo = Path(a.photo).read_bytes() if a.photo else make_photo()
    if photo is None:
        report("snap", "read a photo", "SKIP", "no --photo and Pillow is missing to draw one")
    else:
        t0 = time.time()
        code, snap = api.call("POST", "/snap", student, files={"photo": ("work.jpg", photo, "image/jpeg")}, data={"room_id": room_id})
        took = time.time() - t0
        if code == 503:
            report("snap", "read a photo", "FAIL", f"no image reader: {snap}. LLM_MODEL must accept images (or SNAP_READER=ollama)")
        elif code != 200:
            report("snap", "read a photo", "FAIL", f"{code} {snap}")
        else:
            text = snap.get("text", "")
            report("snap", "read a photo", "PASS" if text.strip() else "WARN",
                   f"{took:.1f}s type={snap.get('type')} confidence={snap.get('confidence')} text={text[:80]!r}")
            if text.strip():
                code, out = copilot(text, mode="check_work", file_id=snap["file_id"])
                p = out.get("payload", {}) if isinstance(out, dict) else {}
                ok = code == 200 and out.get("route") == "check_work" and not p.get("unavailable") and not p.get("sample")
                report("snap", "check the worked solution (step status + hint)", "PASS" if ok else "FAIL",
                       f"{code} steps={len(p.get('steps', []))} hint={str(p.get('hint', ''))[:100]}" + (" (sample!)" if p.get("sample") else ""))

    # ---------------------------------------------------------------- 6. teacher side (M5)
    for step, path in [("class report", f"/teacher/rooms/{room_id}/report"),
                       ("class insights (questions by topic)", f"/teacher/rooms/{room_id}/insights"),
                       ("students list with levels", f"/rooms/{room_id}/students"),
                       ("inbox (escalated questions)", f"/rooms/{room_id}/inbox")]:
        code, body = api.call("GET", path, teacher)
        if path.endswith("/inbox") and code == 200 and escalated_id:
            ids = [q.get("question_id") for q in body.get("questions", [])]
            report("teacher", step, "PASS" if escalated_id in ids else "FAIL",
                   f"the escalated question {escalated_id} is {'in' if escalated_id in ids else 'NOT in'} the inbox ({len(ids)} questions)")
            continue
        report("teacher", step, "PASS" if code == 200 else "FAIL", f"{code} {str(body)[:150]}")
    code, body = api.call("GET", f"/teacher/rooms/{room_id}/students/{s['user_id']}/parent-report", teacher)
    report("teacher", "parent report link", "PASS" if code == 200 else "FAIL", f"{code} {str(body)[:150]}")
    if a.full:
        t0 = time.time()
        code, body = api.call("POST", "/teacher/sessions", teacher, json={"lecturer_id": t["user_id"], "lecturer_name": "Smoke Teacher", "room_id": room_id,
                                                                           "topic": "Photosynthesis", "needs": ["slides", "quiz"], "num_quiz_questions": 3})
        report("teacher", "lecturer assistant draft (LangGraph + MCP)", "PASS" if code in (200, 201, 202) else "FAIL",
               f"{code} {time.time() - t0:.0f}s {str(body)[:150]}")
    else:
        report("teacher", "lecturer assistant draft", "SKIP", "slow and uses many tokens: run with --full to include it")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=os.getenv("RAFEEQ_URL", "http://127.0.0.1:8000"))
    p.add_argument("--pdf", help="a lesson PDF to upload (default: a small generated English lesson)")
    p.add_argument("--question", help="a question answered in that PDF")
    p.add_argument("--photo", help="a photo of handwritten work (default: a generated one)")
    p.add_argument("--index-wait", type=int, default=420, help="seconds to wait for indexing (first run downloads the embedder)")
    p.add_argument("--timeout", type=float, default=180, help="seconds per request")
    p.add_argument("--full", action="store_true", help="also run the slow teacher assistant draft")
    a = p.parse_args()
    for flag, path in (("--pdf", a.pdf), ("--photo", a.photo)):
        if path and not Path(path).is_file():
            p.error(f"{flag}: file not found: {path}")
    try:
        run(a)
    except Exception as e:                                         # noqa: BLE001  (report it instead of losing it)
        import traceback
        report("script", "unexpected error", "FAIL", f"{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        counts = {k: sum(r["status"] == k for r in RESULTS) for k in ("PASS", "WARN", "FAIL", "SKIP", "INFO")}
        print("\n" + "  ".join(f"{k} {v}" for k, v in counts.items()))
        out = ROOT / "results" / "smoke_check.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "base": a.base, "counts": counts, "results": RESULTS},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved {out.relative_to(ROOT)}")
        sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()

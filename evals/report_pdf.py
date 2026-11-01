"""The evaluation report as a PDF: every instructor feature, in the order of the presentation.

    uv run python evals/report_pdf.py                 # -> evals/results/report.pdf

Per feature: how the feature works (its diagram from docs/diagrams), how it is evaluated (a pipeline diagram drawn
here), the metrics, every case with its input, the output it produced and its scores, and the insights the numbers
show. Nothing is re-run and nothing is re-scored: the cases, outputs and feedback are read from the feature's
newest LangSmith experiment (the one recorded in evals/results/run-*.json), the Langfuse judge and the monitoring
from the newest run that read them from Langfuse.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

from evals import core, flow                                                    # noqa: E402

OUT = core.RESULTS / "report.pdf"
DIAGRAMS = ROOT / "docs" / "diagrams"
FEATURE_PNG = {"lesson_studio": "lesson.png", "video_discovery": "discovery.png", "story_video": "story.png",
               "montage": "montage.png", "voice_visual": "voice-to-visual.png", "parent_batch": "reports.png",
               "parent_report": "reports.png"}
JUDGE_KEYS = {"quiz_key_agreement", "judge_grade", "top1_relevant", "precision_at_3", "judge_arc", "judge_prompts"}
TARGET = {
    "lesson_studio": "LangGraph lecturer graph run to the approval step, then the deck exported to .pptx twice",
    "video_discovery": "YouTube search, transcripts, Gemini embeddings in pgvector; the question asked twice",
    "story_video": "Story planner: facts blueprint, then the scene plan (LLM_PROVIDER)",
    "voice_visual": "Voice-to-Visual planner on hand-labelled caption lines",
    "learning_platform": "Library, courses, publishing and progress code on an in-memory store",
    "montage": "Pause cutter and render on a lecture built with ffmpeg (known speech and pauses)",
    "class_report": "Class report logic on a simulated class",
    "inbox": "Inbox answer reuse on hand-written question pairs",
    "parent_batch": "Parent report batch sending on simulated data",
    "parent_report": "Parent report content on simulated data",
}

TEAL, TEAL_LIGHT, INK, QUIET, LINE = (14, 124, 123), (224, 242, 241), (30, 41, 59), (100, 116, 139), (148, 163, 184)
GOOD, BAD, GREY = (22, 128, 61), (185, 28, 28), (120, 120, 120)
W = 180                                                                          # usable width (A4, 15 mm margins)


# ------------------------------------------------------------------ data
def saved_entries() -> tuple[dict, dict | None]:
    """Newest LangSmith-backed result of each suite + the newest Langfuse monitoring, from evals/results/run-*.json."""
    out, monitoring = {}, None
    for f in sorted(core.RESULTS.glob("run-*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if monitoring is None and (data.get("monitoring") or {}).get("features"):
            monitoring = data["monitoring"]
        for s in data.get("suites", []):
            if s.get("name") and s.get("experiment") and s["name"] not in out:
                out[s["name"]] = {**s, "run": s.get("carried_from") or data.get("started", "")}
    return out, monitoring


def langsmith_cases(experiment: str) -> tuple[list[dict], int]:
    """Every root run of the experiment: input, output, error, latency, feedback (score + comment). From LangSmith."""
    from langsmith import Client

    client = Client()
    runs = list(client.list_runs(project_name=experiment, is_root=True))
    fb = list(client.list_feedback(run_ids=[r.id for r in runs])) if runs else []
    by_run: dict = {}
    for f in fb:
        by_run.setdefault(f.run_id, {})[f.key] = (f.score, f.comment or "")
    project = client.read_project(project_name=experiment)
    n_examples = 0
    if project.reference_dataset_id:
        n_examples = sum(1 for _ in client.list_examples(dataset_id=project.reference_dataset_id))
    cases = []
    for r in sorted(runs, key=lambda r: (r.inputs or {}).get("case_id", "")):
        cases.append({"id": (r.inputs or {}).get("case_id", ""), "checks": bool((r.inputs or {}).get("checks")),
                      "input": (r.inputs or {}).get("input"), "output": r.outputs or {}, "error": r.error or "",
                      "seconds": (r.end_time - r.start_time).total_seconds() if r.end_time else None,
                      "finished": r.end_time is not None, "feedback": by_run.get(r.id, {})})
    return cases, n_examples


# ------------------------------------------------------------------ short text for inputs and outputs
def _cut(text, n: int = 110) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def input_lines(suite: str, inp) -> list[str]:
    if not isinstance(inp, dict):
        return [_cut(inp)] if inp else []
    if suite == "lesson_studio":
        return [f"Topic: {inp.get('topic')}", f"{inp.get('language', 'English')} · {inp.get('student_level')} · needs: "
                f"{', '.join(inp.get('needs', []))} · {inp.get('num_quiz_questions')} questions",
                *(["Objectives: " + "; ".join(inp.get("learning_objectives", []))] if inp.get("learning_objectives") else [])]
    if suite == "video_discovery":
        return [f"Question: {inp.get('question')}"]
    if suite == "story_video":
        return [f"Topic: {inp.get('topic')}", f"{inp.get('language')} · {inp.get('scenes')} scenes · {inp.get('style')}"]
    if suite == "voice_visual":
        segs = inp.get("segments", [])
        return [f"{len(segs)} caption lines of 30 s ({inp.get('language')}), {int(inp.get('total', 0))} s lecture",
                *[f"{int(s['start'])} s: {_cut(s['text'], 90)}" for s in segs[:3]], "…" if len(segs) > 3 else ""]
    return [_cut(json.dumps(inp, ensure_ascii=False), 200)]


def output_lines(suite: str, out: dict) -> list[str]:
    if not out:
        return ["(no output)"]
    if "metrics" in out:
        return [f"{m['name']}: {m['value']}{(' ' + m['unit']) if m.get('unit') else ''} ({m.get('status', '')})"
                + (" · " + ", ".join(f"{k}={v}" for k, v in (m.get("detail") or {}).items()) if m.get("detail") else "")
                for m in out["metrics"]]
    if suite == "lesson_studio":
        lines = [f"Status: {out.get('status')} · agents: {', '.join(out.get('agents', []))}",
                 f"{len(out.get('slides', []))} slides: " + _cut(" | ".join(s.get('title', '') for s in out.get('slides', [])), 300)]
        for q in out.get("quiz", [])[:6]:
            opts = q.get("options", [])
            key = q.get("answer_index")
            lines.append(f"Q: {_cut(q.get('question'), 120)}  key: {opts[key] if isinstance(key, int) and key < len(opts) else key}")
        for c in out.get("code", []):
            lines.append(f"Code: {_cut(c.get('title'), 80)} · verified={c.get('verified')} · fix attempts={c.get('fix_attempts')}")
        p = out.get("pptx") or {}
        if p:
            lines.append(f"PPTX: opens={p.get('opens')} · {p.get('pptx_slides')} pages for {p.get('draft_slides')} slides · "
                         f"2nd export = v{p.get('second_version')}" + (f" · {p['error']}" if p.get("error") else ""))
        return lines
    if suite == "video_discovery":
        res = out.get("results", [])
        lines = [f"{len(res)} moments · first search {out.get('first_seconds')} s · repeat {out.get('repeat_seconds')} s"]
        for r in res[:3]:
            lines.append(f"{_cut(r.get('title'), 70)} @ {int(r.get('start_sec') or 0)} s · similarity {r.get('score')}")
        return lines
    if suite == "story_video":
        lines = [f"Character: {_cut(out.get('character'), 100)}"]
        for i, s in enumerate(out.get("scenes", []), 1):
            lines.append(f"{i}. {_cut(s.get('narration'), 130)}")
        return lines
    if suite == "voice_visual":
        plan = out.get("plan", [])
        return [f"Planner: {out.get('planner')} · {len(plan)} moments"] + \
               [f"{int(m['start'])}-{int(m['end'])} s · {m['kind']} · {m['label']}" for m in plan]
    return [_cut(json.dumps(out, ensure_ascii=False, default=str), 300)]


# ------------------------------------------------------------------ insights (from the numbers above, nothing new)
def insights(suite: str, entry: dict, cases: list[dict], n_examples: int) -> list[str]:
    tips = []
    metrics = entry.get("metrics", [])
    gated = [m for m in metrics if m.get("gate") is not None and m.get("value") is not None]
    fails = [m for m in gated if m["status"] == "FAIL"]
    if gated:
        tips.append(f"{len(gated) - len(fails)} of {len(gated)} gated metrics pass" +
                    (": " + "; ".join(f"{m['name']} {m['value']:g} vs gate {m['gate']:g}" for m in fails) if fails else "."))
    real = [c for c in cases if not c["checks"]]
    if n_examples:
        expected = n_examples - (1 if any(c["checks"] for c in cases) else 0)
        done = [c for c in real if c["finished"]]
        if real and len(done) < expected:
            missing = expected - len(done)
            tips.append(f"Only {len(done)} of {expected} dataset cases have a finished run in this experiment "
                        f"({missing} missing or unfinished); the averages cover those {len(done)} only. Re-run the feature to cover all.")
    for c in real:
        if c["error"]:
            tips.append(f"Case '{c['id']}' ended with an error: {_cut(c['error'], 160)}")
    keys = sorted({k for c in real for k in c["feedback"]})
    for k in keys:
        vals = [(c["feedback"][k][0], c) for c in real if k in c["feedback"] and c["feedback"][k][0] is not None]
        if not vals or k in ("first_seconds", "repeat_seconds"):
            continue
        low = min(vals, key=lambda v: v[0])
        if low[0] < 1 and (k in JUDGE_KEYS or low[0] < 0.999):
            kind = "judge" if k in JUDGE_KEYS else "check"
            comment = f" Judge note: {_cut(low[1]['feedback'][k][1], 160)}" if low[1]["feedback"][k][1] else ""
            tips.append(f"Lowest {kind} score on '{k}': {low[0]:.2f} in case '{low[1]['id']}' "
                        f"(mean {statistics.mean(v for v, _ in vals):.2f} over {len(vals)} cases).{comment}")
    secs = [(c["seconds"], c["id"]) for c in real if c["seconds"]]
    if len(secs) > 1:
        slow = max(secs)
        tips.append(f"Latency: median {statistics.median(s for s, _ in secs):.1f} s; slowest '{slow[1]}' {slow[0]:.1f} s.")
    lf = next((m for m in metrics if m["name"] == "langfuse_judge"), None)
    if lf is not None and lf.get("value") is None:
        tips.append("Langfuse's own judge has no score for this run's traces yet: check the evaluator status in Langfuse "
                    "(Evaluators) before using that number.")
    if real and len(real) < 10:
        tips.append(f"Small sample ({len(real)} cases): read the numbers as a smoke test, not as a statistic.")
    if not tips:
        tips.append("No issue in the numbers of this run.")
    return tips


# ------------------------------------------------------------------ PDF drawing
def _font_files() -> dict[str, Path]:
    dirs = [Path(os.environ["PDF_FONT_DIR"])] if os.environ.get("PDF_FONT_DIR") else []
    dirs += [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/msttcorefonts"),
             Path("/Library/Fonts")]
    for d in dirs:
        if (d / "arial.ttf").exists():
            return {"": d / "arial.ttf", "B": (d / "arialbd.ttf") if (d / "arialbd.ttf").exists() else d / "arial.ttf"}
    dv = Path("/usr/share/fonts/truetype/dejavu")
    if (dv / "DejaVuSans.ttf").exists():
        return {"": dv / "DejaVuSans.ttf", "B": dv / "DejaVuSans-Bold.ttf"}
    raise RuntimeError("No Unicode font found. Set PDF_FONT_DIR to a folder with arial.ttf and arialbd.ttf.")


class Report:
    def __init__(self):
        from fpdf import FPDF

        self.pdf = pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_margins(15, 15, 15)
        pdf.set_auto_page_break(True, 15)
        for style, path in _font_files().items():
            pdf.add_font("U", style, str(path))
        try:
            pdf.set_text_shaping(True)                                           # Arabic: shaping + right-to-left runs
        except Exception:                                                        # noqa: BLE001 - uharfbuzz missing
            pass

    # text
    def text(self, s: str, size: float = 9.5, bold: bool = False, color=INK, h: float | None = None, indent: float = 0):
        from fpdf.enums import XPos, YPos

        p = self.pdf
        p.set_font("U", "B" if bold else "", size)
        p.set_text_color(*color)
        p.set_x(15 + indent)
        p.multi_cell(W - indent, h or size * 0.5, s, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(self, s: str, size: float = 15):
        self.pdf.ln(2)
        self.text(s, size, True, TEAL)
        self.pdf.set_draw_color(*TEAL)
        self.pdf.set_line_width(0.5)
        y = self.pdf.get_y() + 0.5
        self.pdf.line(15, y, 195, y)
        self.pdf.ln(2.5)

    def sub(self, s: str):
        self.need(14)
        self.pdf.ln(2)
        self.text(s, 11, True, INK)
        self.pdf.ln(0.5)

    def bullets(self, items: list[str], color=INK):
        for it in items:
            if it:
                self.need(6)
                self.text("•  " + it, 9, False, color, 4.6, indent=2)

    def need(self, mm: float):
        if self.pdf.get_y() + mm > 297 - 15:
            self.pdf.add_page()

    # boxes and arrows (the diagrams)
    def box(self, x, y, w, title, lines=(), accent=False, h=None) -> float:
        p = self.pdf
        p.set_font("U", "", 7.6)
        body = [ln for ln in lines if ln]
        wrapped = []
        for ln in body:
            wrapped += p.multi_cell(w - 4, 3.4, ln, dry_run=True, output="LINES")
        p.set_font("U", "B", 8.4)
        tl = len(p.multi_cell(w - 4, 3.8, title, dry_run=True, output="LINES"))
        hh = h or (3.7 + 3.8 * tl + 3.4 * len(wrapped) + 2)
        p.set_line_width(0.5 if accent else 0.3)
        p.set_draw_color(*(TEAL if accent else LINE))
        p.set_fill_color(*(TEAL_LIGHT if accent else (255, 255, 255)))
        p.rect(x, y, w, hh, style="DF", round_corners=True, corner_radius=2)
        p.set_xy(x + 2, y + 1.6)
        p.set_font("U", "B", 8.4)
        p.set_text_color(*INK)
        p.multi_cell(w - 4, 3.8, title, align="C")
        p.set_font("U", "", 7.6)
        p.set_text_color(*QUIET)
        p.set_xy(x + 2, y + 2.4 + 3.8 * tl)
        for ln in wrapped:
            p.set_x(x + 2)
            p.multi_cell(w - 4, 3.4, ln, align="C")
        return hh

    def arrow(self, x1, y1, x2, y2, dashed=False):
        p = self.pdf
        p.set_draw_color(*LINE)
        p.set_fill_color(*LINE)
        p.set_line_width(0.35)
        if dashed:
            p.set_dash_pattern(dash=1.2, gap=1)
        p.line(x1, y1, x2, y2)
        p.set_dash_pattern()
        import math

        a = math.atan2(y2 - y1, x2 - x1)
        s = 1.8
        p.polygon([(x2, y2), (x2 - s * math.cos(a - 0.45), y2 - s * math.sin(a - 0.45)),
                   (x2 - s * math.cos(a + 0.45), y2 - s * math.sin(a + 0.45))], style="F")

    def label(self, x, y, s, size=7):
        p = self.pdf
        p.set_font("U", "", size)
        p.set_text_color(*QUIET)
        p.set_xy(x, y)
        p.cell(0, 3, s)

    def image(self, path: Path, max_h: float = 120):
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        width = min(W, max_h * w / h)
        self.need(width * h / w + 4)
        self.pdf.image(str(path), x=15 + (W - width) / 2, w=width)
        self.pdf.ln(2)


def eval_diagram(r: Report, suite: str, mod, entry: dict, n_cases: int, judged: bool):
    """Dataset -> the feature -> evaluators -> LangSmith feedback -> report (+ Langfuse's judge on the traces)."""
    spec = getattr(mod, "SPEC", None)
    if spec:
        keys = [s.key for s in spec if not s.key.startswith("@")]
    else:
        keys = [m["name"] for m in entry.get("metrics", []) if m["name"] != "langfuse_judge"]
    checks = [k for k in keys if k not in JUDGE_KEYS]
    judges = [k for k in keys if k in JUDGE_KEYS]
    extra = [m["name"] for m in entry.get("metrics", []) if spec and m["name"] not in {s.name for s in spec}
             and m["name"] != "langfuse_judge"]
    p = r.pdf
    r.need(95)
    y0 = p.get_y() + 7
    x1, w1, x2, w2, x3, w3, x4, w4 = 15, 34, 56, 38, 101, 46, 154, 41
    h1 = r.box(x1, y0, w1, "Dataset", [f"rafeeq-m5-{suite}", f"{n_cases} cases" + (" + checks" if extra else "")])
    h2 = r.box(x2, y0, w2, "Feature under test", [TARGET.get(suite, suite)], accent=True)
    hc = r.box(x3, y0, w3, "Code checks", [", ".join(checks + extra) or "-"])
    yj = y0 + hc + 6
    hj = r.box(x3, yj, w3, "LLM-as-judge (Gemini)", [", ".join(judges) or "none: no model output to grade"])
    h4 = r.box(x4, y0, w4, "LangSmith", ["feedback stored per run", "report = mean per key"])
    yr = y0 + h4 + 6
    hr = r.box(x4, yr, w4, "Report", ["report.md · report.pdf", "deck.json"], accent=True)
    mid = lambda y, h: y + min(h, 14) / 2                                        # noqa: E731
    r.arrow(x1 + w1, mid(y0, h1), x2, mid(y0, h2))
    r.arrow(x2 + w2, mid(y0, h2), x3, mid(y0, hc))
    r.arrow(x2 + w2, mid(y0, h2) + 2, x3, mid(yj, hj))
    r.arrow(x3 + w3, mid(y0, hc), x4, mid(y0, h4))
    r.arrow(x3 + w3, mid(yj, hj), x4, mid(y0, h4) + 3)
    r.arrow(x4 + w4 / 2, y0 + h4, x4 + w4 / 2, yr)
    r.label(x1, y0 - 4.2, "LangSmith experiment (evaluate)", 7.2)
    bottom = max(yj + hj, yr + hr)
    if judged:
        yl = bottom + 9
        r.label(x1, yl - 4.2, "Langfuse (its own evaluator, on the same run's traces)", 7.2)
        hl1 = r.box(x2, yl, w2, "Langfuse traces", ["environment: evaluation"])
        hl2 = r.box(x3, yl, w3, "Langfuse LLM-as-a-judge", ["rule: this feature's root traces", "score 0-1 + reason"])
        r.arrow(x2 + w2 / 2, y0 + h2, x2 + w2 / 2, yl, dashed=True)
        r.arrow(x2 + w2, mid(yl, hl1), x3, mid(yl, hl2))
        r.arrow(x3 + w3, mid(yl, hl2), x4 + w4 / 2, yr + hr)
        bottom = max(bottom, yl + max(hl1, hl2))
    p.set_y(bottom + 4)


def pipeline_diagram(r: Report):
    p = r.pdf
    r.need(70)
    y0 = p.get_y() + 5
    r.label(15, y0 - 4.2, "LangSmith: offline evaluation (before release)", 7.4)
    xs, w = [15, 52, 89, 126, 163], 32
    titles = [("Suite cases", "cases() in code"), ("Dataset", "rafeeq-m5-*"), ("evaluate()", "runs the feature"),
              ("Evaluators", "checks + judge"), ("Feedback", "stored per run")]
    hs = [r.box(x, y0, w, t, [s]) for x, (t, s) in zip(xs, titles)]
    for i in range(4):
        r.arrow(xs[i] + w, y0 + 7, xs[i + 1], y0 + 7)
    yb = y0 + max(hs) + 12
    r.label(15, yb - 4.2, "Langfuse: online evaluation + monitoring (every trace)", 7.4)
    hb = [r.box(15, yb, w, "Real app use", ["env: default"]), r.box(89, yb, w, "Langfuse", ["traces + judge"]),
          r.box(126, yb, w, "Metrics API", ["scores, latency"])]
    r.arrow(15 + w, yb + 7, 89, yb + 7)
    r.arrow(89 + w, yb + 7, 126, yb + 7)
    r.arrow(89 + w / 2, y0 + hs[2], 89 + w / 2, yb, dashed=True)
    yc = yb + max(hb) + 10
    hc = r.box(126, yc, 69, "Report (md, pdf, deck.json)", ["numbers read from both tools"], accent=True)
    r.arrow(126 + w / 2, yb + max(hb), 126 + w / 2, yc)
    r.arrow(163 + w / 2, y0 + hs[4], 163 + w / 2, yc)
    p.set_y(yc + hc + 4)


def metrics_table(r: Report, metrics: list[dict]):
    from fpdf.fonts import FontFace

    p = r.pdf
    p.set_font("U", "", 8)
    p.set_text_color(*INK)
    p.set_draw_color(*LINE)
    p.set_line_width(0.2)
    p.set_fill_color(240, 248, 247)
    head = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=TEAL)
    with p.table(col_widths=(34, 18, 16, 18, 94), line_height=4.2, text_align=("LEFT", "RIGHT", "CENTER", "CENTER", "LEFT"),
                 headings_style=head, width=W, cell_fill_color=(240, 248, 247), cell_fill_mode="EVEN_ROWS") as t:
        row = t.row()
        for h in ("metric", "value", "gate", "status", "what it means"):
            row.cell(h)
        for m in metrics:
            v = m["value"]
            val = "-" if v is None else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.3g}") + (f" {m['unit']}" if m["unit"] else "")
            gate = "" if m["gate"] is None else f"{'≥' if m['higher_is_better'] else '≤'} {m['gate']:g}"
            color = {"PASS": GOOD, "FAIL": BAD}.get(m["status"], GREY)
            row = t.row()
            row.cell(m["name"])
            row.cell(val)
            row.cell(gate)
            row.cell(m["status"], style=FontFace(emphasis="BOLD", color=color))
            row.cell(_cut(m["note"], 170))


def case_block(r: Report, suite: str, c: dict, n: int):
    r.need(24)
    status = "ERROR" if c["error"] else ("not finished" if not c["finished"] else "ok")
    secs = f"{c['seconds']:.1f} s" if c["seconds"] else "-"
    r.text(f"{n}. {c['id']}   ·   {status}   ·   {secs}", 9.5, True, TEAL, 4.8)
    if not c["checks"]:
        r.text("Input", 8, True, QUIET, 4)
        for ln in input_lines(suite, c["input"]):
            if ln:
                r.text(ln, 8.3, False, INK, 4, indent=3)
    r.text("Output", 8, True, QUIET, 4)
    for ln in output_lines(suite, c["output"]) if not c["error"] else [_cut(c["error"], 300)]:
        r.text(ln, 8.3, False, BAD if c["error"] else INK, 4, indent=3)
    if c["feedback"]:
        r.text("Scores (LangSmith feedback)", 8, True, QUIET, 4)
        parts = []
        for k, (v, comment) in sorted(c["feedback"].items()):
            parts.append(f"{k} = {v:g}" if isinstance(v, (int, float)) else f"{k} = {v}")
        r.text(" · ".join(parts), 8.3, False, INK, 4, indent=3)
        for k, (v, comment) in sorted(c["feedback"].items()):
            if comment and k in JUDGE_KEYS:
                r.text(f"{k}: {_cut(comment, 220)}", 8, False, QUIET, 4, indent=3)
    r.pdf.ln(2)


# ------------------------------------------------------------------ the report
def build(path: Path = OUT) -> Path:
    entries, monitoring = saved_entries()
    r = Report()
    p = r.pdf
    mods = {}
    for name in [n for n, _, _ in flow.FLOW] + list(entries):
        if name not in mods:
            try:
                mods[name] = importlib.import_module(f"evals.suites.{name if name != 'm3_handwriting' else 'm3_report'}")
            except Exception:                                                    # noqa: BLE001
                mods[name] = None
    order = [n for n, sec, _ in flow.FLOW if sec != "Z"]

    # cover + summary
    p.add_page()
    r.text("Rafeeq · Instructor (M5)", 11, True, QUIET)
    r.text("Evaluation report", 24, True, TEAL, 11)
    r.text(f"Built {dt.datetime.now().astimezone():%Y-%m-%d %H:%M} from LangSmith experiments and Langfuse. "
           "Every number, case, output and score in this report was read from those two tools.", 9.5, False, QUIET)
    r.pdf.ln(3)
    r.heading("Summary", 13)
    from fpdf.fonts import FontFace

    head = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=TEAL)
    p.set_font("U", "", 8.5)
    p.set_text_color(*INK)
    p.set_draw_color(*LINE)
    p.set_line_width(0.2)
    p.set_fill_color(240, 248, 247)
    with p.table(col_widths=(52, 20, 20, 50, 38), line_height=4.6, headings_style=head, width=W,
                 cell_fill_color=(240, 248, 247), cell_fill_mode="EVEN_ROWS",
                 text_align=("LEFT", "CENTER", "CENTER", "LEFT", "LEFT")) as t:
        row = t.row()
        for h in ("feature", "cases", "gates", "headline", "run"):
            row.cell(h)
        for name in order:
            _, feature = flow.place(name)
            e = entries.get(name)
            row = t.row()
            row.cell(feature)
            if not e:
                for v in ("-", "-", "not evaluated yet", "-"):
                    row.cell(v)
                continue
            gated = [m for m in e["metrics"] if m["gate"] is not None and m["value"] is not None]
            passed = sum(1 for m in gated if m["status"] == "PASS")
            judge = next((m for m in e["metrics"] if m["name"] in JUDGE_KEYS and m["value"] is not None), None)
            row.cell(str(len([c for c in e.get("cases", [])])))
            row.cell(f"{passed}/{len(gated)}", style=FontFace(emphasis="BOLD", color=GOOD if passed == len(gated) else BAD))
            row.cell(f"{judge['name']} {judge['value']:.2f}" if judge else "-")
            row.cell(str(e["run"])[:16].replace("T", " "))
    r.heading("How the evaluation works", 13)
    r.text("Each feature has a LangSmith dataset (one example per case). langsmith.evaluate() runs the real feature on "
           "every example and runs the evaluators on the result: code checks (exact rules) and an LLM judge (Gemini, a "
           "different model family from the Groq generator). LangSmith stores every score as feedback on the run; the "
           "report reads it back. Langfuse receives the same traces and scores them with its own LLM-as-a-judge "
           "evaluator, and measures latency, errors, tokens and cost of real use.", 9.5)
    pipeline_diagram(r)

    # one chapter per feature
    section = None
    for name in order:
        e = entries.get(name)
        sec, feature = flow.place(name)
        p.add_page()
        if sec != section:
            section = sec
            r.text(sec, 10, True, QUIET)
        r.heading(feature, 16)
        mod = mods.get(name)
        if mod is not None:
            r.text(mod.DESCRIPTION, 9.5, False, QUIET)
        if not e:
            r.text("Not evaluated yet: run  uv run python evals/run_all.py --live --only " + name, 10, True, BAD)
            continue
        cases, n_examples = langsmith_cases(e["experiment"])
        r.text(f"LangSmith experiment {e['experiment']} · dataset {e['dataset']} · run {str(e['run'])[:16].replace('T', ' ')}",
               8.5, False, QUIET)
        png = DIAGRAMS / FEATURE_PNG.get(name, "")
        if FEATURE_PNG.get(name) and png.exists():
            r.sub("How the feature works")
            r.image(png, 110)
        r.sub("How it is evaluated")
        eval_diagram(r, name, mod, e, len([c for c in cases if not c["checks"]]),
                     any(m["name"] == "langfuse_judge" for m in e["metrics"]))
        r.sub("Results")
        metrics_table(r, e["metrics"])
        r.sub("Insights")
        r.bullets(insights(name, e, cases, n_examples))
        r.sub(f"Cases ({len(cases)})")
        for i, c in enumerate(cases, 1):
            case_block(r, name, c, i)

    # monitoring
    p.add_page()
    r.heading("Monitoring (Langfuse)", 16)
    if not monitoring:
        r.text("No Langfuse monitoring saved yet.", 10)
    else:
        r.text(f"Last {monitoring['days']} days, read from the Langfuse Metrics API. 'default' = the running app, "
               "'evaluation' = traces of evaluation runs.", 9, False, QUIET)
        p.set_font("U", "", 8)
        p.set_text_color(*INK)
        p.set_draw_color(*LINE)
        p.set_line_width(0.2)
        p.set_fill_color(240, 248, 247)
        with p.table(col_widths=(48, 22, 14, 16, 16, 16, 48), line_height=4.4, headings_style=head, width=W,
                     cell_fill_color=(240, 248, 247), cell_fill_mode="EVEN_ROWS") as t:
            row = t.row()
            for h in ("feature", "environment", "traces", "p50", "p95", "errors", "scores"):
                row.cell(h)
            for f in monitoring["features"]:
                for env, d in (f.get("by_environment") or {"-": {}}).items():
                    row = t.row()
                    row.cell(f["feature"])
                    row.cell(env)
                    row.cell(str(int(d.get("traces") or 0)))
                    row.cell("-" if d.get("p50_s") is None else f"{d['p50_s']:g} s")
                    row.cell("-" if d.get("p95_s") is None else f"{d['p95_s']:g} s")
                    row.cell("-" if d.get("error_rate_pct") is None else f"{d['error_rate_pct']:g} %")
                    row.cell(", ".join(f"{k} {v['avg']:.2f}" for k, v in (d.get("scores") or {}).items()
                                       if v.get("avg") is not None) or "-")
    path.parent.mkdir(parents=True, exist_ok=True)
    p.output(str(path))
    return path


if __name__ == "__main__":
    print(f"Saved: {build()}")

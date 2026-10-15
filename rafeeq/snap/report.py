"""One reproducible report of everything Member 3 can measure. Only real runs of the real code: a metric with no data says NOT RUN and why.

    python -m rafeeq.snap.report --cer-from results/results.csv --cer-label "DEMO synthetic images"      # reuse a finished CER run
    python -m rafeeq.snap.report --run-cer --gt data/ground_truth.csv --images data/images --readers easyocr ollama --presets raw pre
    python -m rafeeq.snap.report --skip-cer                                                              # only the tutor measurements

  1. CER / WER with and without preprocessing, VLM against the OCR baseline          (needs images + ground truth, see init_gt.py)
  2. step-status accuracy (ok | check | error) and first-error localisation          (data/eval/work_cases.json, the real grader)
  3. hint relevance (keywords, a proxy), hint format (one sentence, none for ok) and hint leaks (the hint never holds the solution)
  4. uncertainty handling (unreliable lines are asked about, never judged)
  5. citation validity (every citation is a passage M2 returned) and refusal of unanswerable questions   (data/eval/questions.json)
Output: results/report.md and results/report.json (same numbers).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
_SENTENCES = re.compile(r"(?<=[.!?؟])\s+\S")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


# ----------------------------------------------------------------------------- 1) CER
def cer_section(df: pd.DataFrame | None, label: str, why_not: str = "") -> dict:
    if df is None or df.empty:
        return {"status": "NOT RUN", "reason": why_not or "no CER results"}
    table = df.pivot_table(index=["reader", "preset"], columns="type", values="cer", aggfunc="mean")
    table["ALL"] = df.groupby(["reader", "preset"])["cer"].mean()
    table = table.round(3)
    effect = {}
    for reader, g in df.groupby("reader"):
        raw, pre = g[g.preset == "raw"].set_index("image")["cer"], g[g.preset == "pre"].set_index("image")["cer"]
        both = raw.index.intersection(pre.index)
        if len(both):
            effect[reader] = {"raw": round(float(raw[both].mean()), 3), "pre": round(float(pre[both].mean()), 3),
                              "pre_minus_raw": round(float(pre[both].mean() - raw[both].mean()), 3), "images": len(both)}
    best = {}
    for reader, g in df.groupby("reader"):
        by = g.groupby("preset")["cer"].mean()
        best[reader] = {"preset": by.idxmin(), "cer": round(float(by.min()), 3)}
    return {"status": "RAN", "data": label, "images": int(df["image"].nunique()), "mean_cer": json.loads(table.to_json(orient="split")),
            "preprocessing_effect": effect, "best_per_reader": best,
            "baseline_available": bool({"easyocr", "paddleocr"} & set(df["reader"]) and set(df["reader"]) - {"easyocr", "paddleocr"})}


# ----------------------------------------------------------------------------- 2-4) the grader
def work_section(cases_path: Path, lang_filter: str | None = None) -> dict:
    from rafeeq.tutor import llm
    from rafeeq.tutor.check_work import check_work_detail
    from rafeeq.tutor.grading import _WORKING
    if not cases_path.exists():
        return {"status": "NOT RUN", "reason": f"{cases_path} not found"}
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    llm_up = llm.ask_json('Return the JSON {"ok": true}') is not None
    lines_total = lines_ok = 0
    fe_cases = fe_ok = 0
    hint_cases = hint_rel = 0
    non_ok = fmt_ok = leaks = working = 0
    unc_cases = unc_ok = 0
    inc_cases = inc_ok = 0
    per_case = []
    for c in cases:
        if lang_filter and c["lang"] != lang_filter:
            continue
        det = check_work_detail("r_eval", "", "\n".join(c["lines"]), c["lang"])
        items = det["items"]
        got = [it["status"] for it in items]
        n = len(c["lines"])
        match = [i < len(got) and got[i] in c["statuses"][i] for i in range(n)]
        lines_total += n
        lines_ok += sum(match)
        pred_first = next((it["i"] for it in items if it["status"] == "error"), None)
        fe_cases += 1
        fe_ok += pred_first == c["first_error"]
        if c.get("first_error") and c.get("hint_keywords"):
            hint = next((it["hint"] for it in items if it["i"] == c["first_error"]), "").lower()
            hint_cases += 1
            hint_rel += any(k.lower() in hint for k in c["hint_keywords"])
        for it in items:
            if it["status"] == "ok":
                fmt_ok += it["hint"] == ""
                non_ok += 1
            else:
                non_ok += 1
                fmt_ok += bool(it["hint"]) and len(_SENTENCES.findall(it["hint"])) == 0 and len(it["hint"].split()) <= 29
                if any(_norm(s) in _norm(it["hint"]) for s in c.get("solution", [])):
                    leaks += 1
                if _WORKING.search(it["hint"]):
                    working += 1
        if c.get("expect_clarification"):
            unc_cases += 1
            unc_ok += bool(det["needs_clarification"] and det["clarification"] and "check" in got)
        if c.get("complete") is False:
            inc_cases += 1
            inc_ok += det["complete"] is False
        per_case.append({"id": c["id"], "expected": c["statuses"], "got": got, "first_error_expected": c["first_error"], "first_error_got": pred_first,
                         "hints": [it["hint"] for it in items if it["hint"]]})
    return {"status": "RAN", "llm_available": llm_up, "model": os.getenv("OLLAMA_MODEL") if llm_up else None, "cases": len(per_case),
            "data": f"developer-written cases in {cases_path.name} (text, not photos)",
            "step_status_accuracy": {"correct": lines_ok, "total": lines_total, "rate": _rate(lines_ok, lines_total)},
            "first_error_localization": {"correct": fe_ok, "total": fe_cases, "rate": _rate(fe_ok, fe_cases)},
            "hint_relevance_keyword_proxy": {"relevant": hint_rel, "total": hint_cases, "rate": _rate(hint_rel, hint_cases)},
            "hint_format": {"compliant": fmt_ok, "total": non_ok, "rate": _rate(fmt_ok, non_ok), "rule": "one sentence <= 28 words for a step that is not ok, empty for ok"},
            "hint_leaks": {"hints_containing_a_solution_string": leaks, "hints_containing_a_line_of_working": working},
            "uncertainty_handling": {"handled": unc_ok, "total": unc_cases, "rate": _rate(unc_ok, unc_cases)},
            "incomplete_work_detected": {"detected": inc_ok, "total": inc_cases, "rate": _rate(inc_ok, inc_cases)},
            "per_case": per_case}


# ----------------------------------------------------------------------------- 5) citations
def citation_section(questions_path: Path) -> dict:
    from rafeeq.tutor import citations as cite
    from rafeeq.tutor.answer import tutor_answer
    from rafeeq.tutor.search import search_with_source
    if not questions_path.exists():
        return {"status": "NOT RUN", "reason": f"{questions_path} not found"}
    spec = json.loads(questions_path.read_text(encoding="utf-8"))
    room = spec["room_id"]
    total = valid = answerable = answered = unanswerable = refused = 0
    support: dict[str, int] = {}
    sources: set[str] = set()
    rows = []
    for q in spec["questions"]:
        retrieved, src = search_with_source(room, q["q"])
        sources.add(src)
        res = tutor_answer(room, q["q"], q["lang"], chunks=retrieved)
        v = cite.verify(res["citations"], retrieved, res["answer"] if res["grounded"] else "")
        total += len(res["citations"])
        valid += len(v["valid"])
        support[v["support"]] = support.get(v["support"], 0) + 1
        if q["answerable"]:
            answerable += 1
            answered += bool(res["grounded"])
        else:
            unanswerable += 1
            refused += (not res["grounded"]) and not res["citations"]
        rows.append({"q": q["q"], "grounded": res["grounded"], "citations": res["citations"], "support": v["support"], "flagged": bool(res.get("unsupported_answer"))})
    return {"status": "RAN", "room_id": room, "material_source": sorted(sources),
            "note": "material_source 'sample' means M2's real retrieval was not reachable: these are the agreed sample passages, not your database" if "sample" in sources else "",
            "citation_validity": {"valid": valid, "total": total, "rate": _rate(valid, total)}, "support": support,
            "answerable_answered": {"answered": answered, "total": answerable, "rate": _rate(answered, answerable)},
            "unanswerable_refused": {"refused": refused, "total": unanswerable, "rate": _rate(refused, unanswerable)}, "per_question": rows}


# ----------------------------------------------------------------------------- output
def _table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def to_markdown(rep: dict) -> str:
    out = [f"# Member 3 evaluation report", f"Generated {rep['generated']} on {rep['platform']}. Every number below comes from running the code in this repository; nothing is estimated.", ""]
    c = rep["cer"]
    out.append("## 1. CER with and without preprocessing, VLM against OCR baseline")
    if c["status"] != "RAN":
        out += [f"**NOT RUN**: {c['reason']}", ""]
    else:
        out.append(f"Data: **{c['data']}**, {c['images']} images. Lower is better.")
        cols = c["mean_cer"]["columns"]
        t = pd.DataFrame(c["mean_cer"]["data"], columns=cols)
        t.insert(0, "reader / preset", [" / ".join(i) for i in c["mean_cer"]["index"]])
        out += ["", _table(t), ""]
        for r, e in c["preprocessing_effect"].items():
            out.append(f"- {r}: raw {e['raw']} -> pre {e['pre']} (pre - raw = {e['pre_minus_raw']}, negative = preprocessing helps) on {e['images']} images")
        out.append(f"- baseline (EasyOCR/PaddleOCR) next to a vision model: {'yes' if c['baseline_available'] else 'no'}")
        out.append("")
    w = rep["work"]
    out.append("## 2-4. Step status, first error, hints, uncertainty")
    if w["status"] != "RAN":
        out += [f"**NOT RUN**: {w['reason']}", ""]
    else:
        out.append(f"Data: {w['data']}, {w['cases']} cases. LLM available: **{w['llm_available']}**" + (f" ({w['model']})" if w["model"] else " (rules only: the model was not reachable)") + ".")
        for k in ("step_status_accuracy", "first_error_localization", "hint_relevance_keyword_proxy", "hint_format", "uncertainty_handling", "incomplete_work_detected"):
            m = w[k]
            nums = [v for kk, v in m.items() if isinstance(v, int)]
            out.append(f"- {k}: {m['rate']} ({nums[0]}/{nums[1]})" + (f", rule: {m['rule']}" if "rule" in m else ""))
        out.append(f"- hint_leaks: {w['hint_leaks']}  (must both be 0)")
        out.append("- hint relevance is a keyword proxy against a hand-written list, not a human judgement.")
        out.append("")
    q = rep["citations"]
    out.append("## 5. Citation validity")
    if q["status"] != "RAN":
        out += [f"**NOT RUN**: {q['reason']}", ""]
    else:
        out.append(f"Material source: **{', '.join(q['material_source'])}**. {q['note']}")
        out.append(f"- citation_validity: {q['citation_validity']['rate']} ({q['citation_validity']['valid']}/{q['citation_validity']['total']})")
        out.append(f"- support of answers by their cited passages: {q['support']}")
        out.append(f"- answerable questions answered: {q['answerable_answered']['rate']} ({q['answerable_answered']['answered']}/{q['answerable_answered']['total']})")
        out.append(f"- unanswerable questions refused with no citation: {q['unanswerable_refused']['rate']} ({q['unanswerable_refused']['refused']}/{q['unanswerable_refused']['total']})")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cer-from", help="a results.csv written by rafeeq.snap.evaluate")
    ap.add_argument("--cer-label", default="unspecified (say what the images are with --cer-label)")
    ap.add_argument("--run-cer", action="store_true", help="run the CER now (slow: every reader x preset x image)")
    ap.add_argument("--skip-cer", action="store_true")
    ap.add_argument("--gt", default=str(ROOT / "data" / "ground_truth.csv"))
    ap.add_argument("--images", default=str(ROOT / "data" / "images"))
    ap.add_argument("--readers", nargs="+", default=["easyocr"])
    ap.add_argument("--presets", nargs="+", default=["raw", "pre"])
    ap.add_argument("--cases", default=str(ROOT / "data" / "eval" / "work_cases.json"))
    ap.add_argument("--questions", default=str(ROOT / "data" / "eval" / "questions.json"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    os.environ.setdefault("SNAP_MARK_UNCERTAIN", "0")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df, why = None, "skipped (--skip-cer)"
    if a.cer_from:
        p = Path(a.cer_from)
        df, why = (pd.read_csv(p, encoding="utf-8-sig"), "") if p.exists() else (None, f"{p} not found")
    elif a.run_cer:
        from .evaluate import evaluate_cer, load_gt
        gt = Path(a.gt)
        if not gt.exists():
            why = f"{gt} not found: put your photos in data/images and fill the ground truth (python -m rafeeq.snap.init_gt)"
        else:
            df = evaluate_cer(load_gt(gt), Path(a.images), a.readers, a.presets, out_dir=out, verbose=False)
            df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
            a.cer_label = "demo" if "demo" in gt.name else f"{gt.name}"
    elif not a.skip_cer:
        why = "no --cer-from / --run-cer given"
    label = a.cer_label + (" (synthetic, NOT real handwriting)" if "demo" in a.cer_label.lower() else "")
    rep = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "platform": platform.platform(),
           "cer": cer_section(df, label, why), "work": work_section(Path(a.cases)), "citations": citation_section(Path(a.questions))}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = to_markdown(rep)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

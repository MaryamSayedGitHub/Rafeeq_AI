"""Turn the results dictionary into a Markdown report and flat CSV rows.

The "Recommended fixes" are rule-based HYPOTHESES derived from the failure patterns in the results
(each states the evidence it rests on). They are suggestions to test, not conclusions, and nothing in
this project is changed automatically because of them.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Mapping, Sequence


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _retrieval_rows(groups: Mapping[str, Mapping[str, Any]], ks: Sequence[int]) -> list[list[str]]:
    rows = []
    for name, g in groups.items():
        if not g.get("n"):
            continue
        rows.append([name, str(g["n"])] + [pct(g[f"hit@{k}"]) for k in ks] + [pct(g[f"recall@{k}"]) for k in ks] + [num(g[f"mrr@{max(ks)}"])])
    return rows


def recommendations(results: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    ret = results["retrieval"]
    ks = ret["ks"]
    kmax = max(ks)
    cats = ret["failure_categories"]
    n_answerable = ret["overall"].get("n", 0)

    if results["items"]["skipped_draft"]:
        out.append(f"{len(results['items']['skipped_draft'])} draft item(s) were not scored. Finish their ground truth before trusting the aggregate numbers.")
    if n_answerable and n_answerable < 30:
        out.append(f"Only {n_answerable} answerable questions were scored; the confidence intervals are wide and small differences between runs mean little.")
    if cats.get("miss_wrong_document"):
        out.append(f"{cats['miss_wrong_document']} question(s) retrieved nothing from the expected document in the top {kmax}: chunks of another document outrank it. The store can filter by document_id, so document-aware routing is worth testing (this changes the retriever, so do it as a separate, measured experiment).")
    if cats.get("miss_near_page"):
        out.append(f"{cats['miss_near_page']} miss(es) retrieved a page within 2 pages of the expected one: the answer may span chunk/page boundaries. Test expanding each hit with its neighbouring chunks, and double-check those items' expected pages.")
    if cats.get("miss"):
        out.append(f"{cats['miss']} question(s) found nothing near the expected pages. Read the retrieved chunks in per_question.csv: if the wording differs from the book (paraphrase, Arabic), a hybrid lexical + dense score or query rewriting may help; if the evidence is in a figure or table, that is a parsing/vision limitation, not an embedding one.")
    late = cats.get("late_hit", 0) + cats.get("late_hit_partial_coverage", 0)
    if late:
        out.append(f"{late} question(s) had their first relevant chunk below rank 1. A reranker over the top {kmax}-10 is the standard remedy; compare MRR before and after.")
    multi = ret["by_qtype"].get("multi_passage")
    if multi and multi.get("n") and multi[f"full_recall@{kmax}"] < 0.5:
        out.append(f"Multi-passage questions rarely get all their pages into the top {kmax} (FullRecall@{kmax} = {pct(multi[f'full_recall@{kmax}'])}). Consider a larger k for these, or decomposing the question into sub-queries.")
    by_lang = ret["by_language"]
    if "en" in by_lang and "ar" in by_lang and by_lang["en"].get("n", 0) >= 5 and by_lang["ar"].get("n", 0) >= 5:
        gap = by_lang["en"][f"hit@{kmax}"] - by_lang["ar"][f"hit@{kmax}"]
        if gap >= 0.15:
            out.append(f"Arabic questions trail English ones by {gap * 100:.0f} points at Hit@{kmax}. The book is English, so this is the cross-lingual gap; test translating the query to English before retrieval, or adding BGE-M3's sparse/lexical signal.")
    weak = [t for t, g in ret["by_qtype"].items() if g.get("n", 0) >= 3 and g[f"hit@{kmax}"] <= ret["overall"][f"hit@{kmax}"] - 0.2]
    if weak:
        out.append(f"Question type(s) clearly below average at Hit@{kmax}: {', '.join(weak)}. Inspect those rows first.")

    gate = results["retrieval_gate"]
    min_score = results["min_score_in_use"]
    if gate["out_of_scope_pass_rate"]:
        out.append(f"{pct(gate['out_of_scope_pass_rate'])} of out-of-scope questions still pass the min_score={min_score} gate (top-1 score at or above it). The gate alone cannot be relied on to refuse them; the LLM instruction and a verifier carry the rest.")
    if gate["answerable_pass_rate"] is not None and gate["answerable_pass_rate"] < 0.95:
        out.append(f"{pct(1 - gate['answerable_pass_rate'])} of answerable questions would be refused by the min_score={min_score} gate before reaching the LLM: the threshold is too high for them.")
    sweep = results["threshold_sweep"]
    separating = [r for r in sweep if r["out_of_scope_pass_rate"] == 0 and r["answerable_pass_rate"] is not None]
    if separating:
        top_pass = max(r["answerable_pass_rate"] for r in separating)
        best = [r["threshold"] for r in separating if r["answerable_pass_rate"] == top_pass]
        out.append(f"Descriptively, on this benchmark any cut-off from {min(best):.2f} to {max(best):.2f} blocks every out-of-scope question while passing {pct(top_pass)} of answerable ones. With so few out-of-scope questions this is a lead for calibration, not a calibrated threshold.")

    rag = results["rag"]
    if rag.get("status") == "not_run":
        out.append("The RAG answer layer was not run (no LLM configured). Re-run with --llm-base-url / --llm-model to evaluate citations, abstention by the LLM and key-fact coverage.")
    else:
        if rag["out_of_scope"].get("unsupported_answer_rate"):
            out.append(f"The system answered {len(rag['out_of_scope']['unsupported_answer_ids'])} out-of-scope question(s) ({', '.join(rag['out_of_scope']['unsupported_answer_ids'])}). Treat as the most serious failure: read those answers, then strengthen the prompt / add the planned verifier.")
        if rag["answerable"].get("false_abstention_rate"):
            out.append(f"{pct(rag['answerable']['false_abstention_rate'])} of answerable questions were declined. Compare against the retrieval results: if retrieval found the page, the refusal came from the min_score gate or the LLM / citation check (see abstention stages).")
        cit = rag["citations"]
        if cit["n_answers"] and cit["hit_rate"] is not None and cit["hit_rate"] < 0.8:
            out.append(f"Only {pct(cit['hit_rate'])} of answers cite at least one correct page. Check whether the model cites the wrong excerpt numbers or whether retrieval put the right page too low.")
        if rag["judge"].get("status") == "not_run":
            out.append("Answer correctness and faithfulness beyond key-fact coverage were NOT evaluated (no judge). Use --judge, or review the answers manually.")
    return out or ["No failure pattern crossed a reporting rule."]


def render_report(results: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    ret = results["retrieval"]
    ks = ret["ks"]
    kmax = max(ks)
    items = results["items"]
    lines = [f"# Retrieval and RAG baseline: {meta.get('benchmark_name', 'benchmark')}", ""]

    lines += ["## Run", "", _table(["", ""], [[k, str(v)] for k, v in meta.items() if k != "benchmark_name"]), ""]
    lines += [
        "## Benchmark status", "",
        f"- Scored: **{items['scored']}** ({items['scored_answerable']} answerable, {items['scored_out_of_scope']} out-of-scope) of {items['total']} items",
        f"- Skipped because ground truth is not verified (`draft`): {len(items['skipped_draft'])}" + (f" ({', '.join(items['skipped_draft'])}" if items["skipped_draft"] else ""),
        "",
    ]

    lines += ["## Retrieval (independent of the LLM)", "", f"Relevance: {ret['relevance']}. Similarity scores are never used as correctness labels.", ""]
    headers = ["group", "n"] + [f"Hit@{k}" for k in ks] + [f"Recall@{k}" for k in ks] + [f"MRR@{kmax}"]
    lines += ["**Overall**", "", _table(headers, _retrieval_rows({"all answerable": ret["overall"]}, ks)), ""]
    overall = ret["overall"]
    if overall.get("n"):
        lines += [_table(["metric"] + [f"K={k}" for k in ks], [
            ["Hit rate (95% Wilson CI)"] + [f"{pct(overall[f'hit@{k}'])} [{pct(overall[f'hit@{k}_ci95'][0])}, {pct(overall[f'hit@{k}_ci95'][1])}]" for k in ks],
            ["Precision (relevant / K)"] + [pct(overall[f"precision@{k}"]) for k in ks],
            ["Recall (expected pages covered)"] + [pct(overall[f"recall@{k}"]) for k in ks],
            ["Full recall (all expected pages)"] + [pct(overall[f"full_recall@{k}"]) for k in ks],
            ["Share of top-K from another document"] + [pct(overall[f"other_document_share@{k}"]) for k in ks],
        ]), "", f"MRR@{kmax}: **{num(overall[f'mrr@{kmax}'])}**. Precision@K divides by K, so it is capped by how many relevant chunks exist; read it with Recall@K.", ""]
    lines += ["**By question type**", "", _table(headers, _retrieval_rows(ret["by_qtype"], ks)), ""]
    lines += ["**By language**", "", _table(headers, _retrieval_rows(ret["by_language"], ks)), ""]
    if "chunk_level" in ret:
        c = ret["chunk_level"]
        lines += [f"**Strict chunk-level relevance** ({c['n']} questions with expected chunk ids)", "",
                  _table(["metric"] + [f"K={k}" for k in ks], [
                      ["Hit"] + [pct(c[f"hit_chunk@{k}"]) for k in ks],
                      ["Precision"] + [pct(c[f"precision_chunk@{k}"]) for k in ks],
                      ["Recall"] + [pct(c[f"recall_chunk@{k}"]) for k in ks]]),
                  "", f"MRR@{kmax} (chunk level): {num(c[f'mrr_chunk@{kmax}'])}", ""]
    lines += ["**Outcome per answerable question**", "", _table(["category", "questions"], [[k, str(v)] for k, v in ret["failure_categories"].items()]), ""]

    problems = [r for r in results["per_question"] if r["answerable"] and r["failure_category"] != "ok"]
    if problems:
        lines += ["### Questions that did not hit at rank 1", "", _table(
            ["id", "lang", "type", "outcome", "expected pages", "first relevant rank", "top-3 retrieved (page:score)"],
            [[r["id"], r["language"], r["qtype"], r["failure_category"], ",".join(map(str, r["expected_pages"])), str(r["first_relevant_rank"] or "none"),
              " ".join(f"p{h['page']}:{h['score']:.2f}" + ("" if h["file_name"] == r["expected_file"] else "(other doc)") for h in r["retrieved"][:3])] for r in problems]), "",
            "Full retrieved chunk text and scores for every question are in `per_question.csv` and `results.json`.", ""]

    gate = results["retrieval_gate"]
    lines += ["## Abstention gate and score separation", "",
              f"With the provisional `min_score` = {results['min_score_in_use']}: {pct(gate['answerable_pass_rate'])} of answerable and {pct(gate['out_of_scope_pass_rate'])} of out-of-scope questions reach the LLM. "
              "The table is descriptive (share of questions whose top-1 cosine is at least the threshold); scores are not probabilities.", "",
              _table(["threshold", "answerable passing", "out-of-scope passing"],
                     [[f"{r['threshold']:.2f}", pct(r["answerable_pass_rate"]), pct(r["out_of_scope_pass_rate"])] for r in results["threshold_sweep"] if round(r["threshold"] * 100) % 10 == 0]), ""]

    rag = results["rag"]
    lines += ["## RAG answer layer", ""]
    if rag.get("status") == "not_run":
        lines += [f"**Not run.** {rag['reason']}", ""]
    else:
        a, o, c, k = rag["answerable"], rag["out_of_scope"], rag["citations"], rag["key_fact_coverage"]
        lines += [f"Scored {rag['n_scored']} questions; {rag['n_errors']} failed with an error and are excluded" + (": " + ", ".join(f"{e['id']} ({e['error'][:60]})" for e in rag["errors"]) if rag["errors"] else "") + ".", "",
                  _table(["measure", "value"], [
                      ["Answerable questions answered", f"{pct(a['answered_rate'])} (n={a['n']})"],
                      ["False abstention rate", pct(a["false_abstention_rate"])],
                      ["Out-of-scope correctly declined", f"{pct(o['correct_abstention_rate'])} (n={o['n']})"],
                      ["Out-of-scope answered anyway (unsupported)", f"{pct(o['unsupported_answer_rate'])}" + (f" ({', '.join(o['unsupported_answer_ids'])})" if o["unsupported_answer_ids"] else "")],
                      ["Abstention decisions correct", pct(rag.get("abstention_accuracy"))],
                      ["Abstention stages", ", ".join(f"{s}: {n}" for s, n in rag["abstention_stages"].items()) or "none"],
                      ["Answers with >=1 correct (file, page) citation", f"{pct(c['hit_rate'])} (n={c['n_answers']})"],
                      ["Mean citation precision / recall", f"{num(c['mean_precision'])} / {num(c['mean_recall'])}"],
                      ["Citations to a wrong document", str(c["wrong_document_citations"])],
                      ["Key-fact coverage (proxy), answered / end-to-end", f"{pct(k['mean_when_answered'])} / {pct(k['mean_end_to_end'])}"]]), ""]
        groups = rag.get("by_group", {})
        if groups:
            rows = []
            for name, g in groups.items():
                if name == "out_of_scope":
                    rows.append([name, str(g["n"]), "-", f"declined {pct(g['correct_abstention_rate'])}", "-", "-", "-"])
                else:
                    rows.append([name, str(g["n"]), pct(g["answered_rate"]), pct(g["key_fact_coverage_end_to_end"]), pct(g["citation_hit_rate"]),
                                 pct(g["full_citation_recall"]), pct(g.get("judge_correct_rate_end_to_end"))])
            lines += ["**By question group** (full citation recall = every expected page cited; judge column only with --judge):", "",
                      _table(["group", "n", "answered", "key-fact coverage (end-to-end) / OOS declined", "citation hit", "full citation recall", "judge correct (end-to-end)"], rows), ""]
        j = rag["judge"]
        if j.get("status") == "not_run":
            lines += [f"**LLM judge not run.** {j['reason']}", ""]
        else:
            lines += [f"**LLM judge** ({meta.get('judge', 'unknown model')}): judged {j['n_judged']} answers, {j['n_judge_errors']} unparseable/failed.",
                      f"- Correctness: {j['correctness']}; correct rate {pct(j['correct_rate_when_judged'])} of judged, {pct(j['correct_rate_end_to_end'])} end-to-end",
                      f"- Faithfulness: {j['faithfulness']}; fully faithful {pct(j['faithful_rate_when_judged'])} of judged",
                      "- Limitations: an LLM opinion, not ground truth; may favour answers from its own model family; judges only against the excerpts shown; not comparable across judge models.", ""]

    lines += ["## Recommended fixes (hypotheses to test)", ""] + [f"{i}. {text}" for i, text in enumerate(recommendations(results), start=1)] + [""]
    lines += ["## Limitations", "",
              "- Ground truth is page-level: any chunk on an expected page counts as relevant at page level, which is lenient. The chunk-level section (when present) is the strict view.",
              "- Pages are physical PDF pages as stored in the chunk metadata, not the numbers printed in the book.",
              "- Retrieval was measured on the room as the product uses it, so chunks from other documents in the same room can intrude (counted in the other-document share).",
              "- Small benchmark: see the confidence intervals. Do not tune the system on these questions and then report them as held-out results.", ""]
    return "\n".join(lines)


CSV_COLUMNS = ["id", "language", "qtype", "answerable", "question", "expected_pages", "first_relevant_rank", "failure_category",
               "top1_score", "gate_passed", "retrieved", "rag_outcome", "rag_grounded", "citation_hit", "citation_precision",
               "key_fact_coverage", "judge_correctness", "judge_faithfulness", "rag_error", "answer"]


def csv_text(results: Mapping[str, Any]) -> str:
    ks = results["retrieval"]["ks"]
    columns = CSV_COLUMNS[:7] + [f"{m}@{k}" for k in ks for m in ("hit", "precision", "recall")] + CSV_COLUMNS[7:]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in results["per_question"]:
        rag = r.get("rag", {})
        row = dict(r)
        row["expected_pages"] = ",".join(map(str, r["expected_pages"]))
        row["retrieved"] = " | ".join(f"p{h['page']}:{h['score']:.3f}:{h['chunk_id']}" + ("" if h["file_name"] == r["expected_file"] or r["expected_file"] is None else ":OTHER_DOC") for h in r["retrieved"])
        row["rag_outcome"] = rag.get("outcome", "")
        row["rag_grounded"] = rag.get("grounded", "")
        row["citation_hit"] = rag.get("citation", {}).get("hit", "")
        row["citation_precision"] = rag.get("citation", {}).get("precision", "")
        row["key_fact_coverage"] = rag.get("key_fact_coverage", "")
        row["judge_correctness"] = rag.get("judge", {}).get("correctness", "")
        row["judge_faithfulness"] = rag.get("judge", {}).get("faithfulness", "")
        row["rag_error"] = rag.get("error", "") or rag.get("judge_error", "")
        row["answer"] = rag.get("answer_md", "")
        writer.writerow(row)
    return buffer.getvalue()
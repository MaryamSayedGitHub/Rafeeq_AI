"""Parent batch (teacher): can the teacher upload the sheet they already have, and does every contact reach the RIGHT student?

Sheets are built in the shapes teachers really produce (Arabic and English headers, long headers, any column order, Excel numbers
for phones, a BOM). Sending a report to the wrong parent is the worst failure, so it is counted separately."""
from __future__ import annotations

import csv
import io

from evals.core import Metric
from rafeeq.teacher import parent_batch as pb
from rafeeq.teacher import parent_delivery as pd

NAME = "parent_batch"
DESCRIPTION = "Reading teachers' sheets (headers, order, numbers) and matching rows to the right student"

ROWS = [("عمر أحمد", "omar@x.com", "01012345678"), ("ليلى محمود", "layla@x.com", "01112345678"), ("ياسين علي", "yassin@x.com", "01212345678")]
# (headers student / parent email / parent phone, in this column order)
HEADERS = [
    ("اسم الطالب", "ايميل ولي الأمر", "واتساب"), ("student", "parent_email", "parent_phone"), ("Student Name", "Parent Email", "Parent Phone"),
    ("الاسم", "الإيميل", "الموبايل"), ("اسم الطالب", "بريد ولي الأمر", "رقم واتساب ولي الأمر"), ("Name", "Email", "WhatsApp"),
    ("الطالب", "البريد", "الهاتف"), ("اسم الطالب", "إيميل ولي الأمر", "رقم الموبايل"),
]
ORDERS = [(0, 1, 2), (2, 1, 0), (1, 0, 2)]

STUDENTS = [{"user_id": "u1", "name": "عمر أحمد", "email": "omar@school.eg"}, {"user_id": "u2", "name": "ليلى محمود", "email": "layla@school.eg"},
            {"user_id": "u3", "name": "ياسين علي", "email": "yassin@school.eg"}, {"user_id": "u4", "name": "مريم عبد الله", "email": "maryam@school.eg"},
            {"user_id": "u5", "name": "منى حسن", "email": "mona@school.eg"}, {"user_id": "u6", "name": "منى حسن", "email": "mona2@school.eg"}]


def _csv(header, rows, order) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([header[i] for i in order])
    for r in rows:
        w.writerow([r[i] for i in order])
    return buf.getvalue().encode("utf-8-sig")


def run(ctx: dict) -> list[Metric]:
    ok = total = 0
    failures = []
    for h in HEADERS:
        for order in ORDERS:
            total += 1
            try:
                got = pb.parse_sheet("a.csv", _csv(h, ROWS, order))
                good = [(r["student"], r["parent_email"], r["parent_phone"]) for r in got] == ROWS
            except ValueError as e:
                good = False
                failures.append(f"{h}/{order}: {e}")
            ok += good
            if not good and len(failures) < 3 and not any(f.startswith(str(h)) for f in failures):
                failures.append(f"{h}/{order}")
    xlsx_ok, xlsx_note = _xlsx()
    # matching: spelling variants must still match; an ambiguous name must NOT be guessed
    rows = [{"student": "عمر احمد", "student_email": "", "parent_email": "a@x.com", "parent_phone": ""},        # hamza dropped
            {"student": "ليلى محمود ", "student_email": "", "parent_email": "b@x.com", "parent_phone": ""},     # trailing space
            {"student": "ياسين  علي", "student_email": "", "parent_email": "c@x.com", "parent_phone": ""},       # double space
            {"student": "", "student_email": "MARYAM@school.eg", "parent_email": "d@x.com", "parent_phone": ""},  # by the student's email
            {"student": "منى حسن", "student_email": "", "parent_email": "e@x.com", "parent_phone": ""},         # two students share this name
            {"student": "طالب غير موجود", "student_email": "", "parent_email": "f@x.com", "parent_phone": ""},
            {"student": "عمر أحمد", "student_email": "", "parent_email": "g@x.com", "parent_phone": ""}]        # same student twice
    matched, unmatched = pb.match_rows(rows, STUDENTS)
    got = {r["parent_email"]: s["user_id"] for s, r in matched}
    expect = {"a@x.com": "u1", "b@x.com": "u2", "c@x.com": "u3", "d@x.com": "u4"}
    wrong = sum(1 for k, v in got.items() if k in expect and expect[k] != v) + sum(1 for k in ("e@x.com", "f@x.com", "g@x.com") if k in got)
    reasons = {u["row"]["parent_email"]: u["reason"] for u in unmatched}
    phones = {"01012345678": "201012345678", "1012345678": "201012345678", "+20 101 234 5678": "201012345678", "0020 101 234 5678": "201012345678",
              "01112345678": "201112345678", "1112345678": "201112345678", "01512345678": "201512345678", "1512345678": "201512345678"}
    reach = 0
    for raw, want in phones.items():
        try:
            reach += pd.normalize_phone(raw) == want
        except ValueError:
            pass
    return [
        Metric("phone_reaches_whatsapp", reach / len(phones) * 100, "%", gate=100.0,
               note=f"{len(phones)} ways an Egyptian mobile number is written (with 0, with +20/0020, and the Excel version that LOST its leading 0) become the number wa.me needs"),
        Metric("sheet_parse_accuracy", ok / total * 100, "%", gate=95.0, note=f"{total} csv sheets (8 header styles x 3 column orders)" + (f"; failed: {failures[:3]}" if failures else "")),
        Metric("xlsx_numeric_phone", 100.0 if xlsx_ok else 0.0, "%", gate=100.0, note=xlsx_note),
        Metric("name_match_recall", sum(got.get(k) == v for k, v in expect.items()) / len(expect) * 100, "%", gate=100.0, note="spelling variants, extra spaces and email matching find the right student"),
        Metric("wrong_match_count", float(wrong), "", higher_is_better=False, gate=0.0, note="a contact given to the wrong student, or a guess on an ambiguous / unknown / repeated row; must be 0"),
        Metric("unmatched_reasons_correct", 100.0 if (reasons.get("e@x.com"), reasons.get("f@x.com"), reasons.get("g@x.com")) == ("ambiguous", "not_found", "duplicate") else 0.0, "%", gate=100.0,
               note="the teacher is told WHY a row was skipped (ambiguous / not found / duplicate)"),
    ]


def _xlsx() -> tuple[bool, str]:
    try:
        from openpyxl import Workbook
    except ImportError:
        return False, "openpyxl is not installed"
    wb = Workbook()
    ws = wb.active
    ws.append(["اسم الطالب", "ايميل ولي الأمر", "واتساب"])
    ws.append(["عمر أحمد", "omar@x.com", 1012345678.0])               # Excel stores the phone as a number and drops the leading 0
    buf = io.BytesIO()
    wb.save(buf)
    try:
        row = pb.parse_sheet("a.xlsx", buf.getvalue())[0]
    except ValueError as e:
        return False, f"xlsx could not be read: {e}"
    return row["parent_phone"] == "1012345678", f"an Excel numeric phone comes out as '{row['parent_phone']}' (Excel drops the leading 0; see phone_reaches_whatsapp)"

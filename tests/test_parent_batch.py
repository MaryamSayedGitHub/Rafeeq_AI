"""M5: send all parent reports after the teacher approves. No network, no real store."""
import io

import pytest

from rafeeq.teacher import parent_batch as pb
from rafeeq.teacher import parent_delivery as pd

STUDENTS = [{"user_id": "u1", "name": "عمر أحمد", "email": "omar@school.eg"},
            {"user_id": "u2", "name": "ليلى محمود", "email": "layla@school.eg"},
            {"user_id": "u3", "name": "ياسين علي", "email": "yassin@school.eg"}]


class Item:
    def __init__(self, key, value):
        self.key, self.value = key, value


class FakeStore:
    def __init__(self):
        self.data = {}

    def put(self, ns, key, value):
        self.data[(ns, key)] = value

    def get(self, ns, key):
        return Item(key, self.data[(ns, key)]) if (ns, key) in self.data else None

    def search(self, ns, limit=10, offset=0):
        items = [Item(k, v) for (n, k), v in self.data.items() if n == ns]
        return items[offset: offset + limit]


def csv_bytes(text):
    return text.encode("utf-8-sig")


def test_parse_csv_arabic_headers():
    rows = pb.parse_sheet("a.csv", csv_bytes("اسم الطالب,ايميل ولي الأمر,واتساب\nعمر أحمد,p1@x.com,01012345678\n"))
    assert rows == [{"student": "عمر أحمد", "parent_email": "p1@x.com", "parent_phone": "01012345678", "student_email": ""}]


def test_parse_xlsx_with_numeric_phone():
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["student", "parent_email", "parent_phone"])
    ws.append(["Layla", "p2@x.com", 1012345678.0])
    buf = io.BytesIO()
    wb.save(buf)
    rows = pb.parse_sheet("sheet.xlsx", buf.getvalue())
    assert rows[0]["parent_phone"] == "1012345678" and rows[0]["student"] == "Layla"


@pytest.mark.parametrize("name,data,why", [("a.pdf", b"x", "bad_type"), ("a.csv", b"", "empty"),
                                             ("a.csv", csv_bytes("x,y\n1,2\n"), "columns"),
                                             ("a.xlsx", b"not a zip", "bad_file")])
def test_parse_errors(name, data, why):
    with pytest.raises(ValueError) as e:
        pb.parse_sheet(name, data)
    assert str(e.value) == why


def test_match_by_name_ignores_hamza_and_flags_problems():
    rows = [{"student": "عمر احمد", "student_email": "", "parent_email": "", "parent_phone": ""},   # no hamza
            {"student": "", "student_email": "LAYLA@school.eg", "parent_email": "", "parent_phone": ""},
            {"student": "غير موجود", "student_email": "", "parent_email": "", "parent_phone": ""},
            {"student": "عمر أحمد", "student_email": "", "parent_email": "", "parent_phone": ""}]   # duplicate
    matched, unmatched = pb.match_rows(rows, STUDENTS)
    assert [s["user_id"] for s, _ in matched] == ["u1", "u2"]
    assert [u["reason"] for u in unmatched] == ["not_found", "duplicate"]


def test_import_saves_contacts_and_reports_invalid():
    store = FakeStore()
    rows = [{"student": "عمر أحمد", "student_email": "", "parent_email": "p1@x.com", "parent_phone": "01012345678"},
            {"student": "ليلى محمود", "student_email": "", "parent_email": "bad-email", "parent_phone": ""}]
    res = pb.import_contacts(store, "r1", STUDENTS, rows)
    assert res["saved"] == 1 and res["invalid"][0]["reason"] == "email"
    assert pd.get_contact(store, "r1", "u1") == {"email": "p1@x.com", "phone": "201012345678"}


def _ready_store():
    store = FakeStore()
    pd.save_contact(store, "r1", "u1", "p1@x.com", "01012345678")
    pd.save_contact(store, "r1", "u2", "", "01099999999")
    return store


def _prepare(store):
    return pb.prepare(store, "r1", STUDENTS,
                      lambda uid: {"student_name": uid, "summary": ["ملخص"]}, lambda uid: "https://h/p/" + uid)


def test_prepare_makes_pending_drafts_only_for_students_with_contacts():
    store = _ready_store()
    res = _prepare(store)
    assert res["prepared"] == 2 and res["no_contact"] == ["ياسين علي"]
    d = pb.get_draft(store, "r1", "u1")
    assert d["status"] == "pending" and "https://h/p/u1" in d["text"]


def test_nothing_is_sent_until_approved(monkeypatch):
    store = _ready_store()
    _prepare(store)
    monkeypatch.setenv("BREVO_API_KEY", "k")
    monkeypatch.setenv("BREVO_SENDER", "t@x.com")
    sent = []
    res = pb.send_approved_emails(store, "r1", lambda to, s, b: sent.append(to))
    assert res["sent"] == 0 and sent == []


def test_edit_resets_approval_then_approve_and_send_once(monkeypatch):
    store = _ready_store()
    _prepare(store)
    monkeypatch.setenv("BREVO_API_KEY", "k")
    monkeypatch.setenv("BREVO_SENDER", "t@x.com")
    pb.set_status(store, "r1", "u1", "approved")
    d = pb.edit_draft(store, "r1", "u1", "عنوان جديد", "نص معدّل")
    assert d["status"] == "pending"                      # edited -> must be approved again
    assert pb.approve_all(store, "r1") == 2
    sent = []
    res = pb.send_approved_emails(store, "r1", lambda to, s, b: sent.append((to, s, b)))
    assert sent == [("p1@x.com", "عنوان جديد", "نص معدّل")] and res["sent"] == 1   # u2 has no email
    assert pb.send_approved_emails(store, "r1", lambda *a: sent.append(a))["sent"] == 0   # never twice
    assert pb.summary(pb.list_drafts(store, "r1"))["whatsapp_waiting"] == 2


def test_skipped_is_not_sent_and_failures_are_kept(monkeypatch):
    store = _ready_store()
    pd.save_contact(store, "r1", "u3", "p3@x.com", "")
    _prepare(store)
    monkeypatch.setenv("BREVO_API_KEY", "k")
    monkeypatch.setenv("BREVO_SENDER", "t@x.com")
    pb.set_status(store, "r1", "u1", "approved")
    pb.set_status(store, "r1", "u3", "skipped")

    def boom(to, s, b):
        raise pd.DeliveryError("رفض")
    res = pb.send_approved_emails(store, "r1", boom)
    assert res["sent"] == 0 and res["failed"][0]["name"] == "عمر أحمد"
    assert pb.get_draft(store, "r1", "u1")["error"] == "رفض" and not pb.get_draft(store, "r1", "u1")["email_sent"]


def test_whatsapp_done_needs_approval_and_reprepare_keeps_sent(monkeypatch):
    store = _ready_store()
    _prepare(store)
    with pytest.raises(pb.DraftError):
        pb.mark_whatsapp_sent(store, "r1", "u2")
    pb.set_status(store, "r1", "u2", "approved")
    pb.mark_whatsapp_sent(store, "r1", "u2")
    again = _prepare(store)
    assert again["kept_sent"] == 1 and pb.get_draft(store, "r1", "u2")["wa_sent"]


def test_email_off_raises(monkeypatch):
    for k in ("BREVO_API_KEY", "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(pd.DeliveryError):
        pb.send_approved_emails(FakeStore(), "r1")


def test_full_sentence_headers_are_understood():
    rows = pb.parse_sheet("t.csv", csv_bytes("اسم الطالب,إيميل ولي الأمر,رقم واتساب ولي الأمر\nعمر,p@x.com,01012345678\n"))
    assert rows[0]["parent_email"] == "p@x.com" and rows[0]["parent_phone"] == "01012345678" and rows[0]["student"] == "عمر"
    rows = pb.parse_sheet("t.csv", csv_bytes("Student name,Parent's email,Parent WhatsApp number\nOmar,p@x.com,0101\n"))
    assert rows[0]["parent_phone"] == "0101" and rows[0]["parent_email"] == "p@x.com"


def test_template_is_prefilled_and_round_trips():
    store = _ready_store()
    data = pb.build_template(store, "r1", STUDENTS)
    rows = pb.parse_sheet("rafeeq-parents.xlsx", data)
    by = {r["student"]: r for r in rows}
    assert set(by) == {"عمر أحمد", "ليلى محمود", "ياسين علي"}
    assert by["عمر أحمد"]["parent_email"] == "p1@x.com" and by["عمر أحمد"]["parent_phone"] == "+201012345678"
    assert by["عمر أحمد"]["student_email"] == "omar@school.eg" and by["ياسين علي"]["parent_email"] == ""
    # uploading it again matches every student exactly and keeps the saved numbers
    again = pb.import_contacts(FakeStore(), "r1", STUDENTS, rows)
    assert again["saved"] == 3 and not again["unmatched"] and not again["invalid"]

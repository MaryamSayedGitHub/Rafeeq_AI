"""M5: sending the parent report by email or WhatsApp. No network: SMTP is replaced by a fake."""
import smtplib

import pytest

from rafeeq.teacher import parent_delivery as pd

REPORT = {"student_name": "عمر", "room_name": "الرياضيات", "period": {"from": "2026-10-09", "to": "2026-10-16"},
          "summary": ["حلّ عمر اختبارين هذا الأسبوع.", "يحتاج إلى مراجعة إضافية في «الكسور»."]}


class FakeItem:
    def __init__(self, value):
        self.value = value


class FakeStore:
    def __init__(self):
        self.data = {}

    def put(self, ns, key, value):
        self.data[(ns, key)] = value

    def get(self, ns, key):
        return FakeItem(self.data[(ns, key)]) if (ns, key) in self.data else None


@pytest.mark.parametrize("raw,want", [
    ("01012345678", "201012345678"),
    ("+20 101 234 5678", "201012345678"),
    ("0020 1012345678", "201012345678"),
    ("", ""),
])
def test_phone_is_made_ready_for_whatsapp(raw, want):
    assert pd.normalize_phone(raw) == want


@pytest.mark.parametrize("raw", ["12", "abc123", "0" * 30])
def test_bad_phone_is_refused(raw):
    with pytest.raises(ValueError):
        pd.normalize_phone(raw)


def test_bad_email_is_refused():
    with pytest.raises(ValueError):
        pd.normalize_email("not-an-email")
    assert pd.normalize_email(" parent@example.com ") == "parent@example.com"


def test_contact_round_trip_is_per_room_and_student():
    store = FakeStore()
    assert pd.get_contact(store, "r1", "u1") == {"email": "", "phone": ""}
    pd.save_contact(store, "r1", "u1", "p@example.com", "01012345678")
    assert pd.get_contact(store, "r1", "u1") == {"email": "p@example.com", "phone": "201012345678"}
    assert pd.get_contact(store, "r1", "u2")["email"] == ""
    assert pd.get_contact(store, "r2", "u1")["email"] == ""


def test_message_uses_only_the_report_text_and_the_link():
    subject, text = pd.compose(REPORT, "https://rafeeq.example/p/abc")
    assert "عمر" in subject
    assert "حلّ عمر اختبارين هذا الأسبوع." in text and "https://rafeeq.example/p/abc" in text
    assert "14" in text


def test_whatsapp_link_is_encoded_and_empty_without_a_number():
    url = pd.whatsapp_url("201012345678", "مرحبا\nhttps://x/p/t")
    assert url.startswith("https://wa.me/201012345678?text=") and " " not in url and "\n" not in url
    assert pd.whatsapp_url("", "x") == ""


def test_email_needs_settings(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    assert not pd.email_configured()
    with pytest.raises(pd.DeliveryError):
        pd.send_email("p@example.com", "s", "b")


class FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port = host, port

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        if password != "good":
            raise smtplib.SMTPAuthenticationError(535, b"bad")

    def send_message(self, msg):
        FakeSMTP.sent.append(msg)


def test_email_is_sent_over_smtp(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "teacher@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "good")
    monkeypatch.setattr(pd.smtplib, "SMTP", FakeSMTP)
    FakeSMTP.sent.clear()
    pd.send_email("p@example.com", "تقرير", "نص التقرير")
    msg = FakeSMTP.sent[0]
    assert msg["To"] == "p@example.com" and msg["From"] == "teacher@example.com"
    assert "نص التقرير" in msg.get_content()


def test_wrong_password_gives_a_safe_message(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "teacher@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "secret-password")
    monkeypatch.setattr(pd.smtplib, "SMTP", FakeSMTP)
    with pytest.raises(pd.DeliveryError) as e:
        pd.send_email("p@example.com", "s", "b")
    assert "secret-password" not in str(e.value)


def test_email_errors_are_specific(monkeypatch):
    import smtplib, socket, ssl
    from rafeeq.teacher import parent_delivery as pd
    for k, v in {"SMTP_HOST": "smtp.x.com", "SMTP_USER": "a@x.com", "SMTP_PASSWORD": "secret-pw", "SMTP_PORT": "587"}.items():
        monkeypatch.setenv(k, v)
    cases = [(socket.gaierror(-2, "x"), "SMTP_HOST"), (ConnectionRefusedError(), "465"), (TimeoutError(), "465"),
             (ssl.SSLError("bad"), "SMTP_SSL"), (smtplib.SMTPServerDisconnected(), "465"),
             (smtplib.SMTPSenderRefused(550, b"no", "a@x.com"), "SMTP_FROM")]
    for exc, hint in cases:
        class Boom:
            def __init__(self, *a, **k): raise exc
        monkeypatch.setattr(pd.smtplib, "SMTP", Boom)
        try:
            pd.send_email("p@x.com", "s", "b")
            assert False
        except pd.DeliveryError as e:
            assert hint in str(e) and "secret-pw" not in str(e)


def test_port_465_uses_implicit_ssl(monkeypatch):
    from rafeeq.teacher import parent_delivery as pd
    for k, v in {"SMTP_HOST": "h", "SMTP_USER": "a@x.com", "SMTP_PASSWORD": "pw", "SMTP_PORT": "465"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("SMTP_SSL", raising=False)
    used = []

    class Fake:
        def __init__(self, *a, **k): used.append("ssl")
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def login(self, *a): pass
        def send_message(self, *a): pass
    monkeypatch.setattr(pd.smtplib, "SMTP_SSL", Fake)
    pd.send_email("p@x.com", "s", "b")
    assert used == ["ssl"]


def test_brevo_send_and_errors(monkeypatch):
    import io, json, urllib.error
    from rafeeq.teacher import parent_delivery as pd
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BREVO_API_KEY", "key-123")
    assert not pd.email_configured()
    monkeypatch.setenv("BREVO_SENDER", "t@x.com")
    assert pd.email_configured()
    seen = {}

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"{}"

    def ok(req, timeout=0):
        seen["body"] = json.loads(req.data.decode()); seen["key"] = req.get_header("Api-key"); seen["ua"] = req.get_header("User-agent"); return Resp()
    monkeypatch.setattr(pd.urllib.request, "urlopen", ok)
    pd.send_email("p@x.com", "subj", "body")
    assert seen["ua"].startswith("Rafeeq/")
    assert seen["body"]["to"] == [{"email": "p@x.com"}] and seen["body"]["sender"]["email"] == "t@x.com"

    def deny(req, timeout=0):
        raise urllib.error.HTTPError("u", 401, "no", {}, io.BytesIO(b'{"message":"bad"}'))
    monkeypatch.setattr(pd.urllib.request, "urlopen", deny)
    try:
        pd.send_email("p@x.com", "s", "b"); assert False
    except pd.DeliveryError as e:
        assert "BREVO_API_KEY" in str(e) and "key-123" not in str(e)

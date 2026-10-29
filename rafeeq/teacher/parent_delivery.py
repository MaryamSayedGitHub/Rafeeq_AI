"""Deliver a student's weekly parent report to the parent, by email or WhatsApp (M5).

The teacher decides. Nothing is sent on a schedule: the teacher opens the student's report, checks the
parent's contact details and presses a button.

    Email     sent by this server over SMTP (settings below). Works with any provider; for Gmail use an
              "app password" (Google Account > Security > 2-Step Verification > App passwords).
    WhatsApp  a wa.me link with the message already written. The teacher's own WhatsApp opens and the
              teacher presses Send. No WhatsApp Business account or paid API is needed.

The text comes from the report's fixed Arabic templates (parent_report.py), so it cannot contain an
invented claim. Parent contact details are kept per room and student in the class store.

Settings (.env):
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM (default SMTP_USER),
    SMTP_SSL=1 for port 465.
    PUBLIC_BASE_URL  the address parents can open, e.g. https://rafeeq.example.com. localhost links
    only work on the teacher's own computer.
"""
from __future__ import annotations

import logging
import os
import re
import smtplib
import json
import socket
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any
from urllib.parse import quote

log = logging.getLogger(__name__)

SHARE_DAYS = 14
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class DeliveryError(Exception):
    """A message that is safe to show the teacher (never contains a password)."""


# ---------------------------------------------------------------- contact details

def _ns(room_id: str) -> tuple:
    return ("class", room_id, "parent_contact")


def normalize_phone(raw: str) -> str:
    """Digits only, with the country code, as wa.me needs. '' for an empty value.

    01012345678 -> 201012345678 (Egypt). +20 101 234 5678 and 0020... also work. Raises ValueError otherwise.
    """
    digits = re.sub(r"\D", "", raw or "")
    if not digits:
        return ""
    if digits.startswith("00"):
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = "20" + digits[1:]
    elif re.fullmatch(r"1[0125]\d{8}", digits):          # an Egyptian mobile whose leading 0 was lost (Excel stores 01012345678 as a number)
        digits = "20" + digits
    if not 8 <= len(digits) <= 15:
        raise ValueError("phone")
    return digits


def normalize_email(raw: str) -> str:
    email = (raw or "").strip()
    if not email:
        return ""
    if len(email) > 254 or not _EMAIL.match(email):
        raise ValueError("email")
    return email


def get_contact(store: Any, room_id: str, user_id: str) -> dict:
    item = store.get(_ns(room_id), user_id) if store is not None else None
    value = item.value if item else {}
    return {"email": value.get("email", ""), "phone": value.get("phone", "")}


def save_contact(store: Any, room_id: str, user_id: str, email: str, phone: str) -> dict:
    """Validate and keep the parent's email and WhatsApp number. Raises ValueError('email' | 'phone')."""
    value = {"email": normalize_email(email), "phone": normalize_phone(phone)}
    store.put(_ns(room_id), user_id, value)
    return value


# ---------------------------------------------------------------- the message

def compose(report: dict, link: str) -> tuple[str, str]:
    """(subject, text) in Arabic, from the report's own summary lines and the share link."""
    student = report.get("student_name") or "الطالب"
    room = report.get("room_name") or ""
    period = report.get("period") or {}
    subject = f"تقرير {student} الأسبوعي على رفيق"
    lines = ["السلام عليكم ورحمة الله،", ""]
    head = f"هذا ملخص أسبوع {student} على رفيق"
    if room:
        head += f" في مادة «{room}»"
    if period.get("from") and period.get("to"):
        head += f" ({period['from']} إلى {period['to']})"
    lines += [head + ":", ""]
    lines += [f"• {s}" for s in report.get("summary", [])]
    lines += ["", f"التقرير كاملًا (الرابط يعمل {SHARE_DAYS} يومًا):", link, "", "رفيق"]
    return subject, "\n".join(lines)


def whatsapp_url(phone: str, text: str) -> str:
    """wa.me link that opens WhatsApp with the message ready. '' when there is no number."""
    return f"https://wa.me/{phone}?text={quote(text)}" if phone else ""


# ---------------------------------------------------------------- email

def _brevo_key() -> str:
    return os.environ.get("BREVO_API_KEY", "").strip()


def _brevo_sender() -> str:
    return (os.environ.get("BREVO_SENDER", "") or os.environ.get("SMTP_FROM", "")
            or os.environ.get("SMTP_USER", "")).strip()


def email_configured() -> bool:
    if _brevo_key():
        return bool(_brevo_sender())
    return bool(os.environ.get("SMTP_HOST", "").strip() and os.environ.get("SMTP_USER", "").strip()
                and os.environ.get("SMTP_PASSWORD", ""))


def send_email(to: str, subject: str, body: str) -> None:
    """Send one plain-text email. Raises DeliveryError with a message that is safe to show."""
    if not email_configured():
        raise DeliveryError("البريد غير مفعّل على هذا السيرفر. أضف BREVO_API_KEY وBREVO_SENDER "
                            "(أو SMTP_HOST وSMTP_USER وSMTP_PASSWORD) في ملف .env.")
    if _brevo_key():
        return _send_brevo(to, subject, body)
    host = os.environ["SMTP_HOST"].strip()
    user = os.environ["SMTP_USER"].strip()
    password = os.environ["SMTP_PASSWORD"]
    sender = os.environ.get("SMTP_FROM", "").strip() or user
    use_ssl = os.environ.get("SMTP_SSL", "").strip().lower() in {"1", "true", "yes"}
    try:
        port = int(os.environ.get("SMTP_PORT", "").strip() or (465 if use_ssl else 587))
    except ValueError:
        raise DeliveryError("SMTP_PORT يجب أن يكون رقمًا.")
    if port == 465:  # port 465 is always implicit TLS; plain STARTTLS there makes the server hang up
        use_ssl = True

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    msg.set_content(body, charset="utf-8")
    context = ssl.create_default_context()
    try:
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=20, context=context) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                smtp.starttls(context=context)
                smtp.login(user, password)
                smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        raise DeliveryError("رفض خادم البريد اسم المستخدم أو كلمة المرور. مع Gmail استخدم «كلمة مرور التطبيق» "
                            "(16 حرفًا، بدون مسافات) وليس كلمة مرور الحساب.")
    except Exception as e:  # noqa: BLE001 - every failure is mapped to a safe, specific message
        raise DeliveryError(_explain(e, host, port, use_ssl, sender, to))


def _explain(e: Exception, host: str, port: int, use_ssl: bool, sender: str, to: str) -> str:
    """Turn an SMTP/network exception into a specific Arabic message. Never includes the password."""
    name = type(e).__name__
    log.warning("parent email failed: %s: %s", name, str(e)[:200])
    where = f"{host}:{port}"
    alt = "جرّب المنفذ 465 مع SMTP_SSL=1" if not use_ssl else "جرّب المنفذ 587 وأزل SMTP_SSL"
    if isinstance(e, socket.gaierror):
        msg = f"لا يمكن العثور على الخادم «{host}». راجع كتابة SMTP_HOST (مثل smtp.gmail.com) واتصال الإنترنت."
    elif isinstance(e, (socket.timeout, TimeoutError)):
        msg = f"انتهت مهلة الاتصال بـ {where}. غالبًا المنفذ محجوب من الشبكة أو الجدار الناري. {alt}."
    elif isinstance(e, ConnectionRefusedError):
        msg = f"رفض الخادم الاتصال على {where}. تأكد من SMTP_PORT. {alt}."
    elif isinstance(e, ssl.SSLError):
        msg = (f"فشل التشفير مع {where}. {alt}. "
               "(المنفذ 465 يحتاج SMTP_SSL=1، والمنفذ 587 يحتاج SMTP_SSL فارغًا.)")
    elif isinstance(e, smtplib.SMTPNotSupportedError):
        msg = f"الخادم {where} لا يدعم التشفير أو تسجيل الدخول المطلوب. {alt}."
    elif isinstance(e, smtplib.SMTPServerDisconnected):
        msg = f"قطع الخادم {where} الاتصال. غالبًا الإعداد خاطئ: {alt}."
    elif isinstance(e, smtplib.SMTPSenderRefused):
        msg = f"رفض الخادم عنوان المرسل «{sender}». اجعل SMTP_FROM نفس SMTP_USER أو عنوانًا مسموحًا به."
    elif isinstance(e, (smtplib.SMTPRecipientsRefused, smtplib.SMTPDataError)):
        msg = f"رفض الخادم الرسالة أو عنوان ولي الأمر «{to}». تأكد من صحة الإيميل."
    elif isinstance(e, OSError):
        msg = f"تعذّر الاتصال بـ {where}: {name}. تأكد من الإنترنت والجدار الناري."
    else:
        msg = f"تعذّر إرسال البريد ({name})."
    return f"{msg} [{name}]"


def _send_brevo(to: str, subject: str, body: str) -> None:
    """Send through Brevo's HTTPS API (works on networks that block SMTP ports)."""
    payload = json.dumps({"sender": {"email": _brevo_sender(), "name": "رفيق"}, "to": [{"email": to}],
                          "subject": subject, "textContent": body}).encode("utf-8")
    req = urllib.request.Request("https://api.brevo.com/v3/smtp/email", data=payload, method="POST",
                                 headers={"api-key": _brevo_key(), "content-type": "application/json",
                                          "accept": "application/json",
                                          "user-agent": "Rafeeq/1.0 (+https://github.com/MaryamSayedGitHub/Rafeeq_AI)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = str(json.loads(e.read().decode("utf-8", "replace")).get("message", ""))[:160]
        except Exception:  # noqa: BLE001
            pass
        log.warning("brevo failed: HTTP %s %s", e.code, detail)
        if e.code in (401, 403):
            raise DeliveryError("رفضت Brevo مفتاح BREVO_API_KEY، أو عنوان المرسل غير مفعّل. "
                                "تأكد من المفتاح ومن تفعيل المرسل في Brevo (Senders).")
        raise DeliveryError(f"رفضت Brevo الرسالة (HTTP {e.code}). {detail}".strip())
    except (urllib.error.URLError, OSError) as e:
        log.warning("brevo unreachable: %s", type(e).__name__)
        raise DeliveryError(f"تعذّر الوصول إلى Brevo. تأكد من الإنترنت. [{type(e).__name__}]")

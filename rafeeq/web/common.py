"""Shared helpers for the web views: templates, session, CSRF, guards, API error text."""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.templating import Jinja2Templates

from rafeeq.web import api as api_mod
from rafeeq.web.i18n import DEFAULT_LANG, LANGS, direction, t as translate
from rafeeq.web.icons import icon
from rafeeq.web.tutor_md import render_md

HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.globals["icon"] = icon
templates.env.filters["md"] = render_md


def yt_embed(url: str, start=None, end=None) -> str:
    """Embeddable player URL for a YouTube link, playing only start..end. '' for any other link."""
    import re
    from urllib.parse import urlparse

    host = (urlparse(url or "").hostname or "").lower()
    if not (host.endswith("youtube.com") or host == "youtu.be"):
        return ""
    m = re.search(r"(?:[?&]v=|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})", url)
    if not m:
        return ""
    query = ["rel=0"]
    if start:
        query.append(f"start={int(start)}")
    if end:
        query.append(f"end={int(end)}")
    return f"https://www.youtube-nocookie.com/embed/{m.group(1)}?" + "&".join(query)


templates.env.globals["yt_embed"] = yt_embed
templates.env.trim_blocks = True
templates.env.lstrip_blocks = True


class Redirect(Exception):
    """Raised by guards; turned into a redirect (or HX-Redirect for HTMX requests)."""

    def __init__(self, url: str):
        self.url = url


# ---------------------------------------------------------------- session

def lang_of(request: Request) -> str:
    lang = request.session.get("lang", DEFAULT_LANG)
    return lang if lang in LANGS else DEFAULT_LANG


def tr(request: Request, key: str, _iso: bool = True, **kw: Any) -> str:
    return translate(lang_of(request), key, _iso, **kw)


def csrf_token(request: Request) -> str:
    token = request.session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf"] = token
    return token


async def check_csrf(request: Request) -> dict:
    """Returns the parsed form. Raises 403 if the token is missing or wrong."""
    form = await request.form()
    supplied = request.headers.get("x-csrf-token") or str(form.get("csrf_token", ""))
    expected = request.session.get("csrf", "")
    if not expected or not supplied or not secrets.compare_digest(expected, supplied):
        raise HTTPException(status_code=403, detail="csrf")
    return form


def current_user(request: Request) -> dict | None:
    if not request.session.get("token"):
        return None
    return {k: request.session.get(k) for k in ("token", "user_id", "name", "role")}


def require_user(request: Request, role: str | None = None) -> dict:
    user = current_user(request)
    if user is None:
        raise Redirect(url(request, "/login"))
    if role and user["role"] != role:
        raise HTTPException(status_code=403, detail="role")
    return user


def sign_in(request: Request, login_payload: dict) -> None:
    lang = lang_of(request)
    request.session.clear()
    request.session.update({k: login_payload[k] for k in ("token", "user_id", "name", "role")})
    request.session["lang"] = lang


def flash(request: Request, kind: str, text: str) -> None:
    request.session.setdefault("flash", [])
    request.session["flash"] = request.session["flash"] + [[kind, text]]


def pop_flashes(request: Request) -> list:
    return request.session.pop("flash", [])


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


# ---------------------------------------------------------------- urls

def prefix(request: Request) -> str:
    return request.scope.get("root_path", "").rstrip("/")


def url(request: Request, path: str, **query: Any) -> str:
    q = {k: v for k, v in query.items() if v not in (None, "")}
    return prefix(request) + path + (("?" + urlencode(q)) if q else "")


def redirect(request: Request, path: str, **query: Any) -> RedirectResponse:
    return RedirectResponse(url(request, path, **query), status_code=303)


def safe_next(request: Request, nxt: str | None) -> str:
    """Only same-site paths under the UI prefix; anything else goes to the UI home."""
    base = prefix(request) + "/"
    if (not nxt or "\\" in nxt or nxt.startswith("//") or urlparse(nxt).netloc or urlparse(nxt).scheme
            or not nxt.startswith(base)):
        return url(request, "/")
    return nxt


# ---------------------------------------------------------------- rendering

def api(request: Request) -> api_mod.Api:
    return request.app.state.api


def api_message(request: Request, result: api_mod.ApiResult, specific: dict | None = None) -> str:
    """User-facing text for a failed API call."""
    if specific and result.kind in specific:
        return tr(request, specific[result.kind])
    if result.kind == api_mod.UNAVAILABLE:
        return tr(request, "err.unavailable")
    if result.kind == api_mod.FORBIDDEN:
        return tr(request, "err.forbidden")
    if result.kind == api_mod.INVALID:
        return tr(request, "err.invalid") + (f" ({result.message})" if result.message else "")
    if result.kind == api_mod.ERROR and result.message == "timeout":
        return tr(request, "err.timeout")
    return tr(request, "err.generic")


def context(request: Request, **extra: Any) -> dict:
    lang = lang_of(request)
    user = current_user(request)
    return {
        "request": request,
        "lang": lang,
        "dir": direction(lang),
        "t": lambda key, **kw: translate(lang, key, **kw),
        "user": user,
        "csrf": csrf_token(request),
        "u": lambda path, **q: url(request, path, **q),
        "flashes": pop_flashes(request) if not is_htmx(request) else [],
        **extra,
    }


def render(request: Request, template: str, status: int = 200, **extra: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, template, context(request, **extra), status_code=status)


def notice(request: Request, text: str, kind: str = "info") -> HTMLResponse:
    return render(request, "partials/notice.html", text=text, kind=kind)

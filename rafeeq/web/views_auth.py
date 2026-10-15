"""Log in, register, log out, language switch."""
from __future__ import annotations

from starlette.requests import Request

from starlette.responses import RedirectResponse

from rafeeq.platform.service import DEMO_ACCOUNTS, DEMO_PASSWORD
from rafeeq.web import api as A
from rafeeq.web.common import api, check_csrf, current_user, lang_of, redirect, render, safe_next, sign_in, tr

DEMO_EMAILS = {role: email for _, _, email, role, _ in DEMO_ACCOUNTS}


async def home(request: Request):
    return redirect(request, "/rooms" if current_user(request) else "/login")


async def login_page(request: Request):
    if current_user(request):
        return redirect(request, "/rooms")
    return render(request, "auth/login.html", demo=bool(request.app.state.demo_accounts))


async def login_submit(request: Request):
    form = await check_csrf(request)
    demo_role = form.get("demo")
    if demo_role in DEMO_EMAILS and request.app.state.demo_accounts:
        email, password = DEMO_EMAILS[demo_role], DEMO_PASSWORD
    else:
        email, password = str(form.get("email", "")).strip(), str(form.get("password", ""))
    res = await api(request).post("/auth/login", json={"email": email, "password": password})
    if res.ok:
        sign_in(request, res.data)
        return redirect(request, "/rooms")
    error = tr(request, "err.login") if res.kind in (A.AUTH, A.INVALID) else tr(request, "err.generic")
    return render(request, "auth/login.html", status=400, error=error, email=email,
                  demo=bool(request.app.state.demo_accounts))


async def register_page(request: Request):
    if current_user(request):
        return redirect(request, "/rooms")
    return render(request, "auth/register.html", form={"role": request.query_params.get("role", "student")})


async def register_submit(request: Request):
    form = await check_csrf(request)
    data = {k: str(form.get(k, "")).strip() for k in ("name", "email", "role", "join_code")}
    data["password"] = str(form.get("password", ""))
    if data["role"] not in ("student", "teacher"):
        data["role"] = "student"
    body = {k: data[k] for k in ("name", "email", "password", "role")}
    if data["role"] == "student" and data["join_code"]:
        body["join_code"] = data["join_code"]
    res = await api(request).post("/auth/register", json=body)
    if res.ok:
        sign_in(request, res.data)
        return redirect(request, "/rooms")
    if res.kind == A.CONFLICT:
        error = tr(request, "err.email_taken")
    elif res.kind == A.INVALID and "join code" in res.message.lower():
        error = tr(request, "err.code_not_found")
    elif res.kind == A.INVALID:
        error = tr(request, "err.invalid") + (f" ({res.message})" if res.message else "")
    else:
        error = tr(request, "err.generic")
    data.pop("password")
    return render(request, "auth/register.html", status=400, error=error, form=data)


async def logout(request: Request):
    await check_csrf(request)
    lang = lang_of(request)
    request.session.clear()
    request.session["lang"] = lang
    return redirect(request, "/login")


async def set_language(request: Request):
    code = request.path_params["code"]
    if code in ("ar", "en"):
        request.session["lang"] = code
    return RedirectResponse(safe_next(request, request.query_params.get("next")), status_code=303)

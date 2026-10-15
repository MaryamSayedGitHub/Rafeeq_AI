"""Rafeeq web UI (Jinja2 + HTMX), served by the same FastAPI app as the API.

Add two lines where the FastAPI app is created:

    from rafeeq.web import install
    install(app)            # UI at /app, and "/" redirects there if "/" is free

Call it AFTER every app.include_router(...), so the "/" redirect never shadows a real route.

The UI only uses the HTTP routes in contracts/openapi.yaml (in-process by default), so
each teammate's real implementation shows up in the UI as soon as their route returns it.
"""
from __future__ import annotations

from typing import Any

from starlette.responses import RedirectResponse
from starlette.routing import Route


def install(app: Any, prefix: str = "/app", api_base: str | None = None) -> None:
    from rafeeq.web.app import create_web_app

    app.mount(prefix, create_web_app(api_app=app, api_base=api_base), name="web")
    if not any(getattr(r, "path", None) == "/" for r in app.router.routes):
        async def to_ui(request):
            return RedirectResponse(prefix + "/", status_code=307)
        app.router.routes.append(Route("/", to_ui, include_in_schema=False))


__all__ = ["install"]

"""The web UI's only door to the backend: the HTTP routes in contracts/openapi.yaml.

By default it calls the FastAPI app in-process (no network, same routes, same JSON).
Set RAFEEQ_API_BASE=http://host:port to call a separately deployed API instead.

Every call returns an ApiResult instead of raising, so each page section can show its own
state: data, "not connected yet", "not allowed", or an error.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

UNAVAILABLE = "unavailable"   # route missing (404 Not Found / 405 / 501) or service not configured (503)
INVALID = "invalid"           # 400 / 422
AUTH = "auth"                 # 401
FORBIDDEN = "forbidden"       # 403
NOT_FOUND = "not_found"       # 404 with a specific reason (e.g. unknown room)
CONFLICT = "conflict"         # 409
ERROR = "error"               # 5xx, timeout, connection problem


@dataclass
class ApiResult:
    ok: bool
    status: int
    data: Any = None
    kind: str | None = None
    message: str = ""
    content: bytes = b""
    headers: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default) if self.ok and isinstance(self.data, dict) else default


def _classify(status: int, detail: str) -> str:
    if status == 401:
        return AUTH
    if status == 403:
        return FORBIDDEN
    if status == 409:
        return CONFLICT
    if status in (400, 422):
        return INVALID
    if status in (405, 501, 503) or (status == 404 and detail in ("", "Not Found")):
        return UNAVAILABLE
    if status == 404:
        return NOT_FOUND
    return ERROR


class Api:
    def __init__(self, app: Any = None, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = (base_url or os.environ.get("RAFEEQ_API_BASE") or "").rstrip("/")
        self.app = app
        self.timeout = timeout
        if not self.base_url and app is None:
            raise ValueError("Api needs the FastAPI app or RAFEEQ_API_BASE")

    def _client(self, timeout: float) -> httpx.AsyncClient:
        if self.base_url:
            return httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        # raise_app_exceptions=False: a crash inside a route becomes a 500 result, not an exception here
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False),
                                 base_url="http://rafeeq.internal", timeout=timeout)

    async def request(self, method: str, path: str, *, token: str | None = None, json: Any = None,
                      params: dict | None = None, files: Any = None, data: dict | None = None,
                      timeout: float | None = None) -> ApiResult:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if params:
            params = {k: v for k, v in params.items() if v is not None}
        limit = timeout or self.timeout
        try:
            async with self._client(limit) as client:
                # httpx timeouts do not apply to in-process (ASGI) calls, so enforce the limit here too
                r = await asyncio.wait_for(client.request(method, path, headers=headers, json=json, params=params,
                                                          files=files, data=data), timeout=limit)
        except (httpx.TimeoutException, asyncio.TimeoutError):
            return ApiResult(False, 0, kind=ERROR, message="timeout")
        except httpx.HTTPError as e:
            return ApiResult(False, 0, kind=ERROR, message=str(e))

        ctype = r.headers.get("content-type", "")
        body: Any = None
        if "application/json" in ctype:
            try:
                body = r.json()
            except ValueError:
                body = None
        if 200 <= r.status_code < 300:
            return ApiResult(True, r.status_code, data=body, content=r.content, headers=dict(r.headers))
        detail = ""
        if isinstance(body, dict):
            d = body.get("detail", "")
            detail = d if isinstance(d, str) else "; ".join(str(x.get("msg", x)) for x in d if isinstance(x, dict)) or str(d)
        return ApiResult(False, r.status_code, data=body, kind=_classify(r.status_code, detail), message=detail)

    async def get(self, path: str, **kw: Any) -> ApiResult:
        return await self.request("GET", path, **kw)

    async def post(self, path: str, **kw: Any) -> ApiResult:
        return await self.request("POST", path, **kw)

    async def put(self, path: str, **kw: Any) -> ApiResult:
        return await self.request("PUT", path, **kw)

    async def delete(self, path: str, **kw: Any) -> ApiResult:
        return await self.request("DELETE", path, **kw)

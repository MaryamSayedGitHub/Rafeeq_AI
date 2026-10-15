"""Web search for Level 2 with Tavily, plus a cache and a monthly credit guard to protect the free quota.

Env: TAVILY_API_KEY, SEARCH_CACHE_DAYS (default 7), TAVILY_MONTHLY_LIMIT (default 900 credits; set it to your plan's limit,
check the dashboard: a 'basic' search costs 1 credit).
If M5 already has a Tavily wrapper, replace _tavily_http() with a call to it: everything else stays the same.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")           # also when the tutor is used without the rest of the app

from . import safety
from . import study_store as store

TAVILY_URL = "https://api.tavily.com/search"


class SearchError(RuntimeError):
    pass


def _key(query: str, n: int, depth: str) -> str:
    return hashlib.sha256(f"{' '.join(query.lower().split())}|{n}|{depth}".encode()).hexdigest()


def _tavily_http(query: str, max_results: int, depth: str) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise SearchError("TAVILY_API_KEY is not set. Get a free key at tavily.com and set it, then ask again.")
    r = requests.post(TAVILY_URL, timeout=30, headers={"Authorization": f"Bearer {api_key}"},
                      json={"query": query, "search_depth": depth, "max_results": max_results, "include_answer": False})
    if r.status_code in (401, 403):
        raise SearchError("Tavily rejected the API key.")
    if r.status_code in (429, 432, 433):
        raise SearchError("Tavily quota or rate limit reached. Try again later.")
    r.raise_for_status()
    return [{"title": x.get("title") or x.get("url"), "url": x.get("url"), "content": (x.get("content") or "").strip(),
             "score": float(x.get("score") or 0)} for x in r.json().get("results", []) if x.get("url")]


def cached_search(query: str, max_results: int = 5, depth: str = "basic"):
    """Returns (results, from_cache). Cached answers do not cost credits."""
    key = _key(query, max_results, depth)
    ttl = float(os.getenv("SEARCH_CACHE_DAYS", "7")) * 86400
    hit = store.cache_get(key, ttl)
    if hit is not None:
        store.usage_add(cache_hits=1)
        return hit, True
    cost = 2 if depth == "advanced" else 1
    if store.usage_get()["api_credits"] + cost > int(os.getenv("TAVILY_MONTHLY_LIMIT", "900")):
        raise SearchError("Monthly search limit reached (TAVILY_MONTHLY_LIMIT). Cached searches still work.")
    results = _tavily_http(query, max_results, depth)
    store.cache_put(key, query, results)
    store.usage_add(credits=cost)
    return results, False


def multi_search(queries: list[str], top: int = 5, per_query: int = 5) -> dict:
    """Run 2-3 queries, merge, de-duplicate by URL and keep the `top` best sources. Raises safety.Blocked for an illegal or 18+ search (nothing is sent)."""
    safety.assert_allowed(queries)
    merged: dict[str, dict] = {}
    cache_hits = api_calls = 0
    errors: list[str] = []
    for q in queries:
        try:
            results, cached = cached_search(q, per_query)
        except SearchError as e:
            errors.append(str(e))
            continue
        cache_hits += cached
        api_calls += not cached
        for r in results:
            if len(r["content"]) < 80:                         # empty snippets are useless for a study file
                continue
            m = merged.setdefault(r["url"], {**r, "hits": 0})
            m["hits"] += 1
            m["score"] = max(m["score"], r["score"])
    ranked = sorted(merged.values(), key=lambda m: m["score"] + 0.1 * (m["hits"] - 1), reverse=True)[:top]
    sources = [{"n": i, "title": m["title"], "url": m["url"], "content": m["content"], "score": round(m["score"], 3)}
               for i, m in enumerate(ranked, 1)]
    return {"sources": sources, "queries": queries, "cache_hits": cache_hits, "api_calls": api_calls, "errors": errors}

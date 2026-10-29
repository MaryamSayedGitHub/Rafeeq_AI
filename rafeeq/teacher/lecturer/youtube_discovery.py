"""YouTube video discovery: search real YouTube videos and jump to the exact
timestamp that answers a student's question.

This REPLACES searching the teacher's own sessions (video_discovery.py /
Lab 4) as the source behind the dashboard's search box, per the "Madrasa"
spec's Feature 2 (exact timestamp retrieval + Previous/Next navigation).
video_discovery.py is left in place (final_agent still indexes sessions
quietly) in case a "my own recordings" mode is wanted again later.

Pipeline for one search:
  1. youtube-search-python reads YouTube's own search page -> a handful of
     candidate videos (no API key or Google Cloud billing account needed;
     some candidates won't have a transcript, step 2 just skips them).
  2. For each NEW candidate (not already cached), fetch its transcript with
     youtube-transcript-api and chunk it into ~45-second windows.
  3. Embed each chunk (Gemini) and store it in Postgres/pgvector, cached by
     video_id -- a video is only ever transcribed/embedded once, no matter
     how many different questions later match it.
  4. Embed the student's question and rank the candidate videos' chunks by
     cosine similarity, picking the best segment per video (so the result
     list is diverse videos, not 5 chunks of the same one).

Every external call (YouTube search, transcript fetch, Gemini) is wrapped so
one failing video/provider never breaks the whole search -- it is just
skipped, consistent with video_discovery.py's "never raises" contract.
"""

import logging
import os

from rafeeq.teacher.lecturer.settings import DATABASE_URL
from rafeeq.observability import traced

log = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
CHUNK_SECONDS = 45          # group consecutive captions into ~45s segments
MAX_CANDIDATES = 12         # extra on purpose: many videos have no captions
MIN_SCORE = 0.5             # below this a segment is not really about the question


# ---------------------------------------------------------------------------
# Step 1 — find candidate videos (no API key / no Google Cloud billing needed)
# ---------------------------------------------------------------------------
def _walk_videos(node, out: list) -> None:
    """Collect every videoRenderer in YouTube's search data, wherever it sits."""
    if isinstance(node, dict):
        if "videoRenderer" in node and isinstance(node["videoRenderer"], dict):
            out.append(node["videoRenderer"])
        else:
            for v in node.values():
                _walk_videos(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_videos(v, out)


def parse_duration(text: str) -> int:
    """'12:34' -> 754, '1:02:03' -> 3723, '' or 'LIVE' -> 0."""
    parts = text.strip().split(":") if text else []
    if not parts or not all(p.isdigit() for p in parts):
        return 0
    total = 0
    for p in parts:
        total = total * 60 + int(p)
    return total


def parse_results_page(html: str, max_results: int = MAX_CANDIDATES) -> list[dict]:
    """Read the videos out of a YouTube results page (the JSON in `ytInitialData`)."""
    import json

    marker = "ytInitialData"
    at = html.find(marker)
    while at != -1:
        start = html.find("{", at)
        if start == -1:
            break
        try:
            data, _ = json.JSONDecoder().raw_decode(html[start:])
        except ValueError:
            at = html.find(marker, at + len(marker))
            continue
        found: list = []
        _walk_videos(data, found)
        candidates, seen = [], set()
        for r in found:
            vid = r.get("videoId")
            if not vid or vid in seen:
                continue
            seen.add(vid)
            title = "".join(run.get("text", "") for run in (r.get("title") or {}).get("runs", []))
            owner = "".join(run.get("text", "") for run in (r.get("ownerText") or {}).get("runs", []))
            thumbs = (r.get("thumbnail") or {}).get("thumbnails") or []
            length = (r.get("lengthText") or {}).get("simpleText", "")
            candidates.append({"video_id": vid, "title": title, "channel": owner,
                               "thumbnail": thumbs[-1].get("url", "") if thumbs else "",
                               "duration_sec": parse_duration(length)})
            if len(candidates) >= max_results:
                break
        if candidates:
            return candidates
        at = html.find(marker, at + len(marker))
    return []


def _scrape_candidates(query: str, max_results: int) -> list[dict]:
    """Our own reader of YouTube's results page (httpx only). Used because
    youtube-search-python is unmaintained and crashes with httpx >= 0.28."""
    import httpx

    resp = httpx.get(
        "https://www.youtube.com/results",
        params={"search_query": query, "sp": "EgIQAQ%3D%3D"},          # filter: videos only
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0 Safari/537.36",
                 "Accept-Language": "ar-EG,ar;q=0.9,en;q=0.8"},
        cookies={"CONSENT": "YES+1", "SOCS": "CAI"},
        follow_redirects=True, timeout=20,
    )
    resp.raise_for_status()
    return parse_results_page(resp.text, max_results)


def search_candidates(query: str, max_results: int = MAX_CANDIDATES) -> list[dict]:
    """Candidate videos for a question, no API key needed. [] (not an exception)
    if every method fails; the reason is logged (and shown by `diagnose`)."""
    try:
        found = _scrape_candidates(query, max_results)
        if found:
            return found
        log.warning("youtube_discovery: results page had no videos (layout changed or consent page)")
    except Exception as e:
        log.warning("youtube_discovery: page search failed: %s: %s", type(e).__name__, str(e)[:300])
    try:
        from youtubesearchpython import VideosSearch

        items = VideosSearch(query, limit=max_results, language="ar", region="EG").result().get("result", [])
    except Exception as e:
        log.warning("youtube_discovery: library search failed: %s: %s", type(e).__name__, str(e)[:300])
        return []
    candidates = []
    for item in items:
        video_id = item.get("id")
        if not video_id:
            continue
        thumbnails = item.get("thumbnails") or []
        candidates.append({
            "video_id": video_id,
            "title": item.get("title", ""),
            "channel": (item.get("channel") or {}).get("name", ""),
            "thumbnail": thumbnails[-1]["url"] if thumbnails else "",
            "duration_sec": parse_duration((item.get("duration") or "")),
        })
    return candidates


# ---------------------------------------------------------------------------
# Step 2 — Transcript + chunking
# ---------------------------------------------------------------------------
_NO_CAPTIONS: dict[str, float] = {}      # video_id -> when to try again (captions are off or missing)
_BLOCKED_UNTIL = 0.0                     # YouTube refused our IP: stop asking for a while
NO_CAPTIONS_HOURS = 24
BLOCK_MINUTES = 10


def fetch_transcript(video_id: str) -> list[dict] | None:
    """None when the video has no usable transcript (disabled, not found, blocked, ...).
    Works with youtube-transcript-api 1.x (`fetch`) and the old 0.6 (`get_transcript`).
    Failures are remembered, so one search never hammers YouTube again for videos that
    cannot work (that is what gets an IP blocked)."""
    global _BLOCKED_UNTIL
    import time

    now = time.time()
    if now < _BLOCKED_UNTIL or _NO_CAPTIONS.get(video_id, 0) > now:
        return None
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        languages = ["ar", "en"]                   # Arabic first (our students' content), English fallback
        api = YouTubeTranscriptApi() if hasattr(YouTubeTranscriptApi, "fetch") else None
        if api is not None:
            return [{"text": x.text, "start": x.start, "duration": x.duration}
                    for x in api.fetch(video_id, languages=languages)]
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except Exception as e:
        name = type(e).__name__
        if name in ("IpBlocked", "RequestBlocked"):
            _BLOCKED_UNTIL = now + BLOCK_MINUTES * 60
            log.warning("youtube_discovery: YouTube is blocking transcript requests from this IP; "
                        "pausing for %d minutes (videos already indexed still work)", BLOCK_MINUTES)
        elif name in ("TranscriptsDisabled", "NoTranscriptFound", "VideoUnavailable"):
            _NO_CAPTIONS[video_id] = now + NO_CAPTIONS_HOURS * 3600
        log.info("youtube_discovery: no transcript for %s (%s)", video_id, name)
        return None


def chunk_transcript(entries: list[dict], window_seconds: float = CHUNK_SECONDS) -> list[dict]:
    """Merge consecutive caption lines into ~window_seconds chunks, so each
    chunk is long enough to carry a full idea (one caption line alone is
    usually just a few words)."""
    chunks: list[dict] = []
    bucket_start: float | None = None
    bucket_text: list[str] = []
    bucket_end = 0.0

    for entry in entries:
        start = float(entry["start"])
        end = start + float(entry.get("duration", 0))
        if bucket_start is None:
            bucket_start = start
        bucket_text.append(entry["text"])
        bucket_end = end
        if bucket_end - bucket_start >= window_seconds:
            chunks.append({"start_sec": bucket_start, "end_sec": bucket_end,
                            "text": " ".join(bucket_text).strip()})
            bucket_start, bucket_text = None, []

    if bucket_text:
        chunks.append({"start_sec": bucket_start, "end_sec": bucket_end,
                        "text": " ".join(bucket_text).strip()})
    return [c for c in chunks if c["text"]]


# ---------------------------------------------------------------------------
# Embeddings + Postgres (same approach as video_discovery.py)
# ---------------------------------------------------------------------------
def _embed(text: str) -> list[float] | None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=EMBED_MODEL, contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return result.embeddings[0].values


def _embed_many(texts: list[str]) -> list[list[float]] | None:
    """Embed many texts in as few calls as possible (100 per call)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not texts:
        return None
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 100):
        result = client.models.embed_content(
            model=EMBED_MODEL, contents=texts[i:i + 100],
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
        vectors += [list(e.values) for e in result.embeddings]
    return vectors


def _text_model() -> str:
    return os.getenv("YOUTUBE_VIDEO_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"


def _json_from_gemini(contents, *, media_low: bool = False):
    """One Gemini call that must answer in JSON. Raises on any failure."""
    import json
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    kwargs = {"response_mime_type": "application/json"}
    try:
        config = types.GenerateContentConfig(**kwargs, **({"media_resolution": "MEDIA_RESOLUTION_LOW"} if media_low else {}))
    except Exception:                                   # an older SDK without media_resolution
        config = types.GenerateContentConfig(**kwargs)
    resp = client.models.generate_content(model=_text_model(), contents=contents, config=config)
    return json.loads(resp.text)


def expand_queries(question: str) -> list[str]:
    """The question itself plus up to 2 better YouTube search phrases (Arabic and English).
    A conversational question rarely matches video titles; a short topic phrase does."""
    queries = [question]
    if not os.getenv("GOOGLE_API_KEY"):
        return queries
    try:
        data = _json_from_gemini(
            "A teacher wants YouTube videos that explain this: " + question + "\n"
            "Write 2 short YouTube search phrases (3-6 words each) that would find good educational videos: "
            "one in the question's own language and one in English. "
            'Answer with a JSON array of strings only, e.g. ["...", "..."].')
        for q in data if isinstance(data, list) else []:
            q = str(q).strip()[:100]
            if q and q.lower() not in {x.lower() for x in queries}:
                queries.append(q)
    except Exception as e:
        log.info("youtube_discovery: query expansion skipped (%s)", type(e).__name__)
    return queries[:3]


MAX_OUTLINES_PER_SEARCH = 2           # Gemini watches the video: slower and paid, so only a couple per search
OUTLINE_MIN_SEC, OUTLINE_MAX_SEC = 60, 30 * 60
_NO_OUTLINE: set[str] = set()


def outline_with_gemini(video_id: str, duration_sec: int) -> list[dict]:
    """For videos with no captions: Gemini watches the YouTube video and returns timed
    segments (start, end, what is explained), which are then searched like a transcript.
    [] on any failure."""
    if not os.getenv("GOOGLE_API_KEY") or video_id in _NO_OUTLINE:
        return []
    from google.genai import types

    prompt = ("Watch this lesson video. Split it into consecutive segments of 30-120 seconds. For each segment "
              "write what is explained (1-2 sentences, in the language of the video). Answer with a JSON array of "
              '{"start": seconds, "end": seconds, "text": "..."} only.')
    try:
        data = _json_from_gemini(
            types.Content(parts=[types.Part(file_data=types.FileData(file_uri=f"https://www.youtube.com/watch?v={video_id}")),
                                 types.Part(text=prompt)]),
            media_low=True)
    except Exception as e:
        _NO_OUTLINE.add(video_id)
        log.info("youtube_discovery: Gemini could not outline %s (%s: %s)", video_id, type(e).__name__, str(e)[:120])
        return []
    chunks = []
    for item in data if isinstance(data, list) else []:
        try:
            start, end, text = float(item["start"]), float(item["end"]), str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if text and 0 <= start < end and (not duration_sec or start < duration_sec):
            chunks.append({"start_sec": start, "end_sec": min(end, duration_sec) if duration_sec else end, "text": text})
    if not chunks:
        _NO_OUTLINE.add(video_id)
    return chunks


def _to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _connect():
    import psycopg
    return psycopg.connect(DATABASE_URL, autocommit=True)


def setup_table() -> None:
    if not DATABASE_URL:
        return
    with _connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS youtube_segments (
                id          SERIAL PRIMARY KEY,
                video_id    TEXT NOT NULL,
                title       TEXT NOT NULL,
                channel     TEXT NOT NULL,
                thumbnail   TEXT NOT NULL,
                start_sec   FLOAT NOT NULL,
                end_sec     FLOAT NOT NULL,
                text        TEXT NOT NULL,
                embedding   VECTOR({EMBED_DIM})
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS youtube_segments_video_idx
            ON youtube_segments (video_id)
        """)


def _index_video(conn, candidate: dict, outline_budget: list[int]) -> None:
    """Fetches (or, with no captions, asks Gemini to outline), chunks, embeds and stores one
    video -- skipped entirely if this video_id is already cached from an earlier search."""
    video_id = candidate["video_id"]
    already = conn.execute(
        "SELECT 1 FROM youtube_segments WHERE video_id = %s LIMIT 1", (video_id,)
    ).fetchone()
    if already:
        return

    entries = fetch_transcript(video_id)
    chunks = chunk_transcript(entries) if entries else []
    if not chunks and outline_budget[0] > 0:
        duration = int(candidate.get("duration_sec") or 0)
        if OUTLINE_MIN_SEC <= duration <= OUTLINE_MAX_SEC:
            outline_budget[0] -= 1
            chunks = outline_with_gemini(video_id, duration)
    if not chunks:
        return
    vectors = _embed_many([c["text"] for c in chunks])
    if vectors is None or len(vectors) != len(chunks):
        log.warning("youtube_discovery: embedding failed for %s", video_id)
        return
    for chunk, vector in zip(chunks, vectors):
        conn.execute(
            """
            INSERT INTO youtube_segments
                (video_id, title, channel, thumbnail, start_sec, end_sec, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            """,
            (video_id, candidate["title"], candidate["channel"], candidate["thumbnail"],
             chunk["start_sec"], chunk["end_sec"], chunk["text"], _to_vector_literal(vector)),
        )


# ---------------------------------------------------------------------------
# Step 3 — Put it all together
# ---------------------------------------------------------------------------
@traced("video_discovery.search", "retriever")
def search(question: str, top_k: int = 5) -> list[dict]:
    """Always returns a list (possibly empty) -- never raises, so the
    dashboard's search box degrades gracefully instead of crashing the page."""
    if not DATABASE_URL:
        return []

    candidates, seen = [], set()
    for q in expand_queries(question):
        for c in search_candidates(q, max_results=6):
            if c["video_id"] not in seen:
                seen.add(c["video_id"])
                candidates.append(c)
    candidates = candidates[:MAX_CANDIDATES]
    if not candidates:
        return []

    try:
        setup_table()
        with _connect() as conn:
            outline_budget = [MAX_OUTLINES_PER_SEARCH]
            for candidate in candidates:
                try:
                    _index_video(conn, candidate, outline_budget)
                except Exception as e:
                    log.warning("youtube_discovery: indexing %s failed: %s: %s",
                                candidate["video_id"], type(e).__name__, str(e)[:200])

            query_vector = _embed(question)
            if query_vector is None:
                return []
            video_ids = [c["video_id"] for c in candidates]
            # DISTINCT ON (video_id): the single best-matching chunk per video,
            # so results are diverse videos instead of 5 chunks of one video.
            rows = conn.execute(
                """
                SELECT DISTINCT ON (video_id)
                       video_id, title, channel, thumbnail, start_sec, end_sec, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM youtube_segments
                WHERE video_id = ANY(%s)
                ORDER BY video_id, embedding <=> %s::vector
                """,
                (_to_vector_literal(query_vector), video_ids, _to_vector_literal(query_vector)),
            ).fetchall()
    except Exception as e:
        log.warning("youtube_discovery: search failed: %s: %s", type(e).__name__, str(e)[:200])
        return []

    results = []
    for video_id, title, channel, thumbnail, start_sec, end_sec, text, score in rows:
        if float(score) < MIN_SCORE:
            continue
        results.append({
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "thumbnail": thumbnail,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "open_at": f"{int(start_sec) // 60:02d}:{int(start_sec) % 60:02d}",
            "text": text,
            "score": round(float(score), 3),
            "url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_sec)}s",
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Which step is failing?   uv run python -m rafeeq.teacher.lecturer.youtube_discovery "سؤالك"
# ---------------------------------------------------------------------------
def diagnose(question: str) -> None:
    logging.basicConfig(level=logging.INFO)
    print("DATABASE_URL set:", bool(DATABASE_URL), "| GOOGLE_API_KEY set:", bool(os.getenv("GOOGLE_API_KEY")))
    found = search_candidates(question)
    print(f"1) YouTube search: {len(found)} videos", [c["title"][:40] for c in found[:3]])
    if not found:
        return
    print("2) transcripts are fetched during step 5 (failures are listed in the log above)")
    try:
        vec = _embed(question)
        print("3) embedding:", "ok" if vec else "FAILED (GOOGLE_API_KEY missing?)")
    except Exception as e:
        print("3) embedding FAILED:", type(e).__name__, str(e)[:200])
    try:
        setup_table()
        print("4) Postgres/pgvector: ok")
    except Exception as e:
        print("4) Postgres/pgvector FAILED:", type(e).__name__, str(e)[:200])
    print("5) final results:", len(search(question)))


if __name__ == "__main__":
    import sys

    diagnose(" ".join(sys.argv[1:]) or "ما هي الخلية")

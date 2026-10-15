"""Video discovery: semantic search with timestamps over a teacher's own sessions.

This is the real, app-wired version of Lab 4 (rafeeq/teacher/labs/lab4_video_discovery).
Instead of a sample script, it indexes the SPEAKER NOTES of every slide a teacher's
session already generates (see schemas.Slide.notes): each slide becomes one "segment"
of a future lesson narration, with start/end seconds estimated from how long it takes
to SAY that many words out loud (~150 words/minute -- the same assumption used by the
TTS labs). A student question then finds the exact slide that explains it, with a
timestamp a future video player can jump to -- even before the HyperFrames video
pipeline (Lab 2) is wired into the app.

Storage: the SAME Postgres database as everything else in Rafeeq (DATABASE_URL),
in a `video_segments` table, scoped per teacher with a `lecturer_id` column so one
teacher never sees another teacher's sessions in their search results.

Every function below degrades to a safe no-op/empty-result when DATABASE_URL or
GOOGLE_API_KEY is missing, instead of raising -- indexing happens inside
final_agent (must never break a teacher's approved session) and search happens
inside a dashboard search box (must never crash the page).
"""

import logging
import os

from rafeeq.teacher.lecturer.long_term_store import lecturer_key
from rafeeq.teacher.lecturer.settings import DATABASE_URL

log = logging.getLogger(__name__)

EMBED_MODEL = "gemini-embedding-001"  # current Gemini embedding model
EMBED_DIM = 768                        # shrunk from its native 3072 dims to keep the table small
WORDS_PER_SECOND = 150 / 60             # same reading-speed assumption as the TTS labs
MIN_SEGMENT_SECONDS = 3.0               # a one-word slide still gets a few seconds on screen


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _embed(text: str) -> list[float] | None:
    """None (not an exception) when GOOGLE_API_KEY is not configured -- callers
    treat that the same as "feature not available yet", like settings.DATABASE_URL."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return result.embeddings[0].values


def _to_vector_literal(vector: list[float]) -> str:
    """pgvector text format '[0.1,-0.2,...]', cast with ::vector in SQL.
    (A plain Python list auto-casts on INSERT but NOT inside a WHERE/ORDER BY
    expression -- see Lab 4's notes -- so every query here casts explicitly.)"""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------
def _connect():
    import psycopg
    return psycopg.connect(DATABASE_URL, autocommit=True)


def setup_table() -> None:
    """Creates the pgvector extension and the video_segments table (once)."""
    if not DATABASE_URL:
        return
    with _connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS video_segments (
                id          SERIAL PRIMARY KEY,
                lecturer_id TEXT NOT NULL,
                video_id    TEXT NOT NULL,
                title       TEXT NOT NULL,
                start_sec   FLOAT NOT NULL,
                end_sec     FLOAT NOT NULL,
                text        TEXT NOT NULL,
                embedding   VECTOR({EMBED_DIM})
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS video_segments_embedding_idx
            ON video_segments USING hnsw (embedding vector_cosine_ops)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS video_segments_lecturer_idx
            ON video_segments (lecturer_id)
        """)


# ---------------------------------------------------------------------------
# Indexing -- called once by final_agent, right after a session is approved
# ---------------------------------------------------------------------------
def index_session(lecturer_id: str, session_id: str, topic: str, slides: list[dict]) -> int:
    """One segment per slide, timed back-to-back from slide 1. Returns how many
    segments were indexed (0 if skipped: no Postgres, no slides, or already done)."""
    if not DATABASE_URL or not slides:
        return 0
    try:
        setup_table()
        key = lecturer_key(lecturer_id)
        with _connect() as conn:
            already = conn.execute(
                "SELECT 1 FROM video_segments WHERE video_id = %s LIMIT 1", (session_id,)
            ).fetchone()
            if already:
                return 0

            cursor_sec, count = 0.0, 0
            for slide in slides:
                notes = (slide.get("notes") or "").strip()
                if not notes:
                    continue
                vector = _embed(notes)
                if vector is None:
                    log.warning("video_discovery: GOOGLE_API_KEY not set, skipping the rest")
                    break
                duration = max(len(notes.split()) / WORDS_PER_SECOND, MIN_SEGMENT_SECONDS)
                start_sec, end_sec = cursor_sec, cursor_sec + duration
                conn.execute(
                    """
                    INSERT INTO video_segments
                        (lecturer_id, video_id, title, start_sec, end_sec, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (key, session_id, topic, start_sec, end_sec, notes, _to_vector_literal(vector)),
                )
                cursor_sec = end_sec
                count += 1
        log.info("video_discovery: indexed %d segment(s) for session '%s'", count, session_id)
        return count
    except Exception as e:  # indexing is a bonus feature: never break an approved session
        log.warning("video_discovery: indexing failed: %s: %s", type(e).__name__, str(e)[:200])
        return 0


# ---------------------------------------------------------------------------
# Search -- called by the dashboard's "ابحث داخل جلساتك" box
# ---------------------------------------------------------------------------
def search(lecturer_id: str, question: str, top_k: int = 5) -> list[dict]:
    """This teacher's own sessions, searched by meaning (not keywords). Always
    returns a list -- [] when Postgres/the embedding key is unavailable or the
    query fails, so the search box degrades instead of crashing the page."""
    if not DATABASE_URL:
        return []
    vector = _embed(question)
    if vector is None:
        return []
    key = lecturer_key(lecturer_id)
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT video_id, title, start_sec, end_sec, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM video_segments
                WHERE lecturer_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_to_vector_literal(vector), key, _to_vector_literal(vector), top_k),
            ).fetchall()
    except Exception as e:
        log.warning("video_discovery: search failed: %s: %s", type(e).__name__, str(e)[:200])
        return []

    results = []
    for video_id, title, start_sec, end_sec, text, score in rows:
        results.append({
            "video_id": video_id,
            "title": title,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "open_at": f"{int(start_sec) // 60:02d}:{int(start_sec) % 60:02d}",
            "text": text,
            "score": round(float(score), 3),
        })
    return results

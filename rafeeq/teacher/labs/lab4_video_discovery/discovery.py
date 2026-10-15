"""
Lab 4 — Video Discovery with Timestamps
=========================================

Idea borrowed from the "Madrasa" spec (Feature 2: Exact Timestamp Retrieval),
but adapted to OUR pipeline instead of YouTube.

Goal: Given a student's question, find the exact moment inside one of OUR OWN
generated lesson videos (Lab 1 TTS + Lab 2 HyperFrames) that answers it,
instead of making the student watch the whole video from the start.

Why this is EASIER than searching YouTube: we are not scraping someone else's
videos and guessing timestamps with speech-to-text. We wrote the script
ourselves, so we already know -- line by line -- what is said and exactly
when (from how long the TTS took to say each line). No transcription needed.

NOTE: this lab uses its own table (lab4_video_segments) so it never clashes with the
real video_segments table in rafeeq/teacher/lecturer/video_discovery.py.

Steps:
  1. Break a lesson script into timed segments (text + start/end seconds)
  2. Turn each segment into an embedding vector (Gemini text-embedding-004)
  3. Store segments + vectors in Postgres using pgvector (same DATABASE_URL
     as the rest of Rafeeq -- short-term memory, long-term memory, and now
     video search all live in ONE Postgres database)
  4. Given a student question, embed it and find the closest segment
     (cosine distance -- smaller distance = more similar meaning)
  5. Return {video_id, title, start_sec, end_sec, text, score} so the
     web player can jump straight to that moment instead of 00:00

Setup (one-time):
    cd D:\\NTI\\Final_Project\\rafeeq
    uv add pgvector google-generativeai
    # GOOGLE_API_KEY must already be in .env (same key used for Gemini chat)

Run:
    python -m rafeeq.teacher.labs.lab4_video_discovery.discovery
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rafeeq:rafeeq@localhost:5432/rafeeq")
EMBED_MODEL = "gemini-embedding-001"  # current Gemini embedding model (text-embedding-004 is retired)
EMBED_DIM = 768                            # this model always returns 768 numbers


# ---------------------------------------------------------------------------
# Step 1 — A sample lesson, already split into timed segments
# ---------------------------------------------------------------------------
# In the real pipeline these segments come from the Script Agent: each line of
# the script becomes one segment, and start/end seconds come from how long
# edge-tts / ElevenLabs took to speak that line (or word-level timestamps if
# the TTS provider gives them, e.g. ElevenLabs does).

@dataclass
class Segment:
    video_id: str
    title: str
    start_sec: float
    end_sec: float
    text: str


SAMPLE_LESSON_ID = "lesson-python-functions-01"
SAMPLE_LESSON_TITLE = "Python Functions — Lecture 1"

SAMPLE_SEGMENTS: list[Segment] = [
    Segment(SAMPLE_LESSON_ID, SAMPLE_LESSON_TITLE, 0, 25,
            "Welcome. Today we introduce Python functions and why we use them."),
    Segment(SAMPLE_LESSON_ID, SAMPLE_LESSON_TITLE, 25, 68,
            "A function is defined with the def keyword, followed by a name and "
            "parentheses for parameters. The indented block below it is the function body."),
    Segment(SAMPLE_LESSON_ID, SAMPLE_LESSON_TITLE, 68, 112,
            "Parameters let you pass values into a function. Default parameter values "
            "are used when the caller does not supply that argument."),
    Segment(SAMPLE_LESSON_ID, SAMPLE_LESSON_TITLE, 112, 150,
            "The return statement sends a value back to the caller and immediately "
            "exits the function. A function with no return statement returns None."),
    Segment(SAMPLE_LESSON_ID, SAMPLE_LESSON_TITLE, 150, 190,
            "Let's look at recursion: a function that calls itself, with a base case "
            "that stops the recursion so it does not run forever."),
]


# ---------------------------------------------------------------------------
# Step 2 — Turn text into an embedding vector
# ---------------------------------------------------------------------------
def embed_text(text: str) -> list[float]:
    """
    Uses the same GOOGLE_API_KEY already configured for Gemini chat (see .env).
    Returns a 768-number vector (we ask Gemini to shrink its native 3072-dim
    vector down to 768 with output_dimensionality, so it fits our table).
    Two sentences about the same idea end up close together in this vector
    space, even when worded completely differently -- that's why this beats
    plain keyword search.

    Uses the new `google-genai` SDK (the old `google-generativeai` package is
    deprecated and `text-embedding-004` is retired on it).
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set in .env")

    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return result.embeddings[0].values


# ---------------------------------------------------------------------------
# Step 3 — Postgres table + pgvector storage
# ---------------------------------------------------------------------------
def _raw_connection():
    """Plain connection, used only to create the extension/table the first time."""
    import psycopg
    return psycopg.connect(DATABASE_URL, autocommit=True)


def get_connection():
    """Plain connection. We send vectors as a pgvector text literal + ::vector
    cast (see to_vector_literal below) instead of relying on an adapter, so
    this works the same in every query -- INSERT, WHERE, and ORDER BY."""
    import psycopg
    return psycopg.connect(DATABASE_URL, autocommit=True)


def to_vector_literal(vector: list[float]) -> str:
    """pgvector accepts text like '[0.12,-0.04,...]' cast with ::vector.
    Using this everywhere (instead of a Python-list adapter) avoids a subtle
    bug: Postgres auto-casts a float array to `vector` on INSERT (assignment
    context) but NOT inside a WHERE/ORDER BY expression -- so a plain list
    works for inserting but fails with "operator does not exist" on search."""
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def setup_table() -> None:
    """Creates the pgvector extension and the lab4_video_segments table (once)."""
    with _raw_connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS lab4_video_segments (
                id          SERIAL PRIMARY KEY,
                video_id    TEXT NOT NULL,
                title       TEXT NOT NULL,
                start_sec   FLOAT NOT NULL,
                end_sec     FLOAT NOT NULL,
                text        TEXT NOT NULL,
                embedding   VECTOR({EMBED_DIM})
            )
        """)
        # HNSW index = fast search even once we have thousands of segments
        conn.execute("""
            CREATE INDEX IF NOT EXISTS lab4_video_segments_embedding_idx
            ON lab4_video_segments USING hnsw (embedding vector_cosine_ops)
        """)
    print("[DB] lab4_video_segments table ready.")


def index_segments(segments: list[Segment]) -> None:
    """Embeds and inserts each segment. Skips a video_id that is already indexed."""
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM lab4_video_segments WHERE video_id = %s LIMIT 1",
            (segments[0].video_id,),
        ).fetchone()
        if existing:
            print(f"[DB] '{segments[0].video_id}' already indexed, skipping.")
            return

        for seg in segments:
            vector = to_vector_literal(embed_text(seg.text))
            conn.execute(
                """
                INSERT INTO lab4_video_segments (video_id, title, start_sec, end_sec, text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (seg.video_id, seg.title, seg.start_sec, seg.end_sec, seg.text, vector),
            )
        print(f"[DB] Indexed {len(segments)} segments for '{segments[0].video_id}'.")


# ---------------------------------------------------------------------------
# Step 4 — Semantic search: find the closest segment to a student's question
# ---------------------------------------------------------------------------
def find_relevant_moment(question: str, top_k: int = 1) -> list[dict]:
    """
    `<=>` is pgvector's cosine-distance operator: smaller = more similar.
    We convert it to a 0..1 relevance score so the UI can show e.g. "92% match".
    """
    query_vector = to_vector_literal(embed_text(question))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT video_id, title, start_sec, end_sec, text,
                   1 - (embedding <=> %s::vector) AS score
            FROM lab4_video_segments
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, top_k),
        ).fetchall()

    results = []
    for video_id, title, start_sec, end_sec, text, score in rows:
        results.append({
            "video_id": video_id,
            "title": title,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": text,
            "score": round(float(score), 3),
            # This is what the <video> player uses to jump straight to the answer:
            "open_at": f"{int(start_sec) // 60:02d}:{int(start_sec) % 60:02d}",
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_table()
    index_segments(SAMPLE_SEGMENTS)

    # EXPERIMENT 1: ask a question worded totally differently from the script,
    # and watch it still match the "return statement" segment
    question = "how does a function send a value back to the code that called it?"
    matches = find_relevant_moment(question, top_k=2)

    print(f"\n[QUERY] {question}\n")
    for m in matches:
        print(f"  {m['score']:.0%} match -> {m['title']} @ {m['open_at']} ({m['start_sec']:.0f}s)")
        print(f"       \"{m['text'][:90]}...\"")

    # EXPERIMENT 2: uncomment to try a different question
    # matches = find_relevant_moment("what is recursion and how does it stop?")

    # EXPERIMENT 3: uncomment to see it correctly NOT match an unrelated question
    # matches = find_relevant_moment("what is a Docker container?")


if __name__ == "__main__":
    main()

"""Temporary course material: the LLM book as one chunk per page.
Same fields as M2's retrieve, so it can be swapped for the real one later."""
from __future__ import annotations

import re
import sys

import fitz  # PyMuPDF


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "chapter"


def load_chunks(pdf_path: str, file_name: str = "hands-on-llms.pdf", min_chars: int = 200) -> list[dict]:
    doc = fitz.open(pdf_path)
    # chapter starts from the PDF's own table of contents (level-1 entries)
    toc = sorted((page, title) for level, title, page in doc.get_toc() if level == 1 and page >= 1)
    chunks = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < min_chars:  # skip blank or figure-only pages
            continue
        title = "front matter"
        for start, t in toc:
            if start <= i:
                title = t
            else:
                break
        chunks.append({
            "chunk_id": f"{file_name}-p{i}",
            "text": text,
            "file_name": file_name,
            "page": i,
            "block_type": "text",
            "topic_id": _slug(title),
            "topic_name": title,
            "score": 1.0,
        })
    return chunks


def list_topics(chunks: list[dict]) -> list[dict]:
    seen: dict[str, str] = {}
    for c in chunks:
        seen.setdefault(c["topic_id"], c["topic_name"])
    return [{"topic_id": k, "name": v} for k, v in seen.items()]


def retrieve(chunks: list[dict], topic_id: str, k: int = 6) -> list[dict]:
    return [c for c in chunks if c["topic_id"] == topic_id][:k]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/book.pdf"
    chunks = load_chunks(path)
    print(f"{len(chunks)} page chunks")
    for t in list_topics(chunks):
        n = sum(1 for c in chunks if c["topic_id"] == t["topic_id"])
        print(f'{t["topic_id"]:<42} {n:>3} pages  {t["name"]}')
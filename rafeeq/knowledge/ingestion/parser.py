"""Page-aware PDF parser (PyMuPDF).

Returns exactly one record per PDF page so page identity is never lost.
Page numbers are 1-indexed, matching what a reader sees in a PDF viewer.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


def clean_text(text: str) -> str:
    """Normalise extracted text while keeping paragraph breaks.

    - removes NUL characters
    - converts Windows/old-Mac line endings to "\\n"
    - collapses runs of spaces/tabs into a single space
    - trims whitespace around line breaks
    - allows at most one blank line in a row
    - strips leading/trailing whitespace
    """
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(file_path: str) -> list[dict]:
    """Parse a PDF into one dict per page.

    Returns:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    Rules:
        - one entry per page, in order
        - "page" is 1-indexed
        - blank pages are kept, with text == ""
        - text is cleaned with clean_text()

    Raises:
        FileNotFoundError: the path does not exist or is not a file.
        ValueError: the file exists but cannot be opened as a PDF.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        doc = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises several error types
        raise ValueError(f"Could not open '{path}' as a PDF: {exc}") from exc

    pages: list[dict] = []
    with doc:
        for index, page in enumerate(doc):
            raw_text = page.get_text("text") or ""
            pages.append({"page": index + 1, "text": clean_text(raw_text)})

    return pages
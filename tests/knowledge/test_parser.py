from pathlib import Path

import fitz  # PyMuPDF
import pytest

from rafeeq.knowledge.ingestion.parser import clean_text, parse_pdf
BOOK_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "hands_on_llm.pdf"


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> str:
    """Create a 3-page PDF: text page, blank page, text page."""
    pdf_path = tmp_path / "sample.pdf"

    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one explains self-attention.")
    doc.new_page()  # page 2 is intentionally blank
    page3 = doc.new_page()
    page3.insert_text((72, 72), "Page three explains transformers.")
    doc.save(str(pdf_path))
    doc.close()

    return str(pdf_path)


# ---------- small synthetic PDF ----------

def test_page_count(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert len(pages) == 3


def test_page_numbering_is_one_indexed_and_ordered(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert [p["page"] for p in pages] == [1, 2, 3]


def test_blank_page_preserved_as_empty_string(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert pages[1]["page"] == 2
    assert pages[1]["text"] == ""


def test_text_extraction(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert "self-attention" in pages[0]["text"]
    assert "transformers" in pages[2]["text"]


def test_each_entry_has_expected_keys(sample_pdf):
    for entry in parse_pdf(sample_pdf):
        assert set(entry.keys()) == {"page", "text"}
        assert isinstance(entry["page"], int)
        assert isinstance(entry["text"], str)


def test_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        parse_pdf(str(missing))


def test_invalid_pdf_raises_value_error(tmp_path):
    fake = tmp_path / "not_a_pdf.pdf"
    fake.write_text("this is not a pdf")
    with pytest.raises(ValueError):
        parse_pdf(str(fake))


# ---------- text cleaning ----------

def test_clean_text_collapses_whitespace_and_blank_lines():
    assert clean_text("  a   b \n\n\n\n c  ") == "a b\n\nc"


def test_clean_text_normalises_line_endings_and_nul():
    assert clean_text("a\r\nb\x00") == "a\nb"


# ---------- optional: the real book ----------

@pytest.mark.skipif(not BOOK_PATH.exists(), reason="data/raw/hands_on_llm.pdf not found")
def test_real_book():
    pages = parse_pdf(str(BOOK_PATH))

    assert len(pages) > 0
    assert [p["page"] for p in pages] == list(range(1, len(pages) + 1))

    non_empty = sum(1 for p in pages if p["text"])
    # Most pages of a text-based book should have a text layer.
    assert non_empty / len(pages) > 0.8

    print(f"\nBook pages: {len(pages)} | non-empty pages: {non_empty}")
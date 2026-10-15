import inspect

import pytest

from rafeeq.knowledge.ingestion.scanned_pages import (
    DEFAULT_MIN_TEXT_CHARS,
    ScannedPagesNotAvailableError,
    extract_pages,
    find_pages_needing_extraction,
    has_usable_text,
    merge_extracted_pages,
)

GOOD = "This page has plenty of real extractable text content."


def pages_fixture():
    return [{"page": 1, "text": GOOD}, {"page": 2, "text": ""}, {"page": 3, "text": "  12  "},
            {"page": 4, "text": GOOD}]


# ---------- detecting pages without a usable text layer ----------

def test_has_usable_text_basic_cases():
    assert has_usable_text(GOOD)
    assert not has_usable_text("")
    assert not has_usable_text("   \n\t ")
    assert not has_usable_text("12 — 3")  # page-number debris only


def test_has_usable_text_counts_arabic_letters():
    assert has_usable_text("يعتمد الترانسفورمر على الانتباه الذاتي بين الكلمات")


def test_has_usable_text_threshold_is_configurable():
    assert not has_usable_text("short text", min_chars=50)
    assert has_usable_text("short text", min_chars=5)
    assert DEFAULT_MIN_TEXT_CHARS > 0


def test_has_usable_text_rejects_non_strings():
    with pytest.raises(TypeError):
        has_usable_text(None)


def test_find_pages_needing_extraction():
    assert find_pages_needing_extraction(pages_fixture()) == [2, 3]
    assert find_pages_needing_extraction([{"page": 1, "text": GOOD}]) == []


# ---------- the stub (future M4 contract) ----------

def test_stub_raises_a_clear_not_implemented_error():
    with pytest.raises(ScannedPagesNotAvailableError, match="not implemented"):
        extract_pages("data/raw/scan.pdf", [2, 3])


def test_stub_error_is_a_not_implemented_error_so_callers_can_catch_it_generically():
    assert issubclass(ScannedPagesNotAvailableError, NotImplementedError)


def test_stub_signature_is_the_m4_contract():
    params = list(inspect.signature(extract_pages).parameters)
    assert params == ["file_path", "pages"]


def test_module_has_no_vision_or_ocr_dependency():
    import rafeeq.knowledge.ingestion.scanned_pages as module

    source = inspect.getsource(module).lower()
    for forbidden in ("import torch", "import cv2", "pytesseract", "import transformers", "import fitz"):
        assert forbidden not in source


# ---------- merging an extractor's result ----------

def test_merge_replaces_only_requested_pages_and_reports_them():
    pages = pages_fixture()
    merged, replaced = merge_extracted_pages(
        pages, [{"page": 2, "text": "  recovered text of page two  ", "bbox": [0, 0, 1, 1]}], requested=[2, 3])
    assert replaced == [2]
    assert [p["text"] for p in merged] == [GOOD, "recovered text of page two", "  12  ", GOOD]
    assert pages == pages_fixture()  # input untouched


def test_merge_ignores_blank_extracted_text():
    merged, replaced = merge_extracted_pages(pages_fixture(), [{"page": 3, "text": "   "}], requested=[2, 3])
    assert replaced == []
    assert merged[2]["text"] == "  12  "


def test_merge_with_no_results_changes_nothing():
    merged, replaced = merge_extracted_pages(pages_fixture(), [], requested=[2, 3])
    assert merged == pages_fixture() and replaced == []


def test_merge_rejects_pages_that_were_not_requested():
    with pytest.raises(ValueError, match="not requested"):
        merge_extracted_pages(pages_fixture(), [{"page": 1, "text": "hijack"}], requested=[2, 3])


@pytest.mark.parametrize("bad", [{"page": 2}, {"text": "x"}, "not a dict", {"page": 2, "text": 5}, {"page": True, "text": "x"}])
def test_merge_rejects_malformed_entries(bad):
    with pytest.raises(ValueError):
        merge_extracted_pages(pages_fixture(), [bad], requested=[1, 2, 3])


def test_merge_rejects_duplicate_pages():
    with pytest.raises(ValueError, match="more than once"):
        merge_extracted_pages(pages_fixture(), [{"page": 2, "text": "a"}, {"page": 2, "text": "b"}], requested=[2])


def test_merge_rejects_non_list_result():
    with pytest.raises(TypeError):
        merge_extracted_pages(pages_fixture(), "oops", requested=[2])
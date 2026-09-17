"""Unit tests for src.extractor.extract().

All tests here exercise the rule-based fallback path only (no network, no
API key needed) - this is the path guaranteed to run in CI / on the
reviewer's machine per the task requirement.
"""
import glob
import os

import pytest

from src.extractor import extract

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# --- Required assertions from the task brief -----------------------------
@pytest.mark.unit
def test_amount_space_thousands_comma_decimal():
    assert extract("Сумма: 1 250 000,00 руб.")["amount"] == 1_250_000.0


@pytest.mark.unit
def test_inn_basic():
    assert extract("ИНН 7701234567")["inn"] == "7701234567"


@pytest.mark.unit
def test_amount_none_when_no_digits():
    assert extract("без цифр")["amount"] is None


# --- Additional amount format coverage ------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Сумма: 1 250 000,00 руб.", 1_250_000.0),
        ("Итого: 1250000.00 ₽", 1_250_000.0),
        ("на сумму 1,250,000.00 RUB", 1_250_000.0),
        ("Итого стоимость работ: 500000.00 ₽", 500_000.0),
    ],
)
def test_amount_formats(text, expected):
    assert extract(text)["amount"] == expected


@pytest.mark.unit
def test_amount_in_words():
    result = extract("Стоимость услуг составляет девятьсот тысяч рублей 00/100.")
    assert result["amount"] == 900_000.0


# --- Date format coverage --------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Договор от 01.03.2025", "2025-03-01"),
        ("составлен 1 марта 2025 г.", "2025-03-01"),
        ("дата: 03/01/25", "2025-03-01"),  # MM/DD/YY per task brief
    ],
)
def test_date_formats(text, expected):
    assert extract(text)["date"] == expected


# --- Field coverage on the provided sample dataset -------------------------
@pytest.mark.unit
def test_extract_all_fields_on_clean_documents():
    """All 6 non-OCR sample documents should yield every field."""
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        name = os.path.basename(path)
        if name in ("subjects_test.txt", "scan_ocr_001.txt"):
            continue
        text = open(path, encoding="utf-8").read()
        result = extract(text)
        for field in ("amount", "date", "inn", "contractor", "subject"):
            assert result[field] is not None, f"{name}: missing {field}"


@pytest.mark.unit
def test_extract_ocr_document_recovers_critical_fields():
    """The OCR-degraded document is a known-hard case: contractor/subject
    are allowed to fail (see RESULTS.md), but the numeric/date/id fields
    must still be recovered via OCR-homoglyph normalization.
    """
    path = os.path.join(DATA_DIR, "scan_ocr_001.txt")
    text = open(path, encoding="utf-8").read()
    result = extract(text)
    assert result["date"] == "2025-03-01"
    assert result["inn"] == "7701234567"
    assert result["amount"] == 1_250_000.0


@pytest.mark.unit
def test_extract_missing_fields_are_none():
    result = extract("Просто текст без каких-либо реквизитов.")
    assert result == {
        "amount": None,
        "date": None,
        "inn": None,
        "contractor": None,
        "subject": None,
    }


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="requires GOOGLE_API_KEY"
)
def test_extract_via_llm_smoke():
    """Only runs when a real API key is configured; exercises the Gemini
    code path end-to-end. Skipped in CI / offline environments."""
    result = extract("Счёт № 1 от 01.03.2025. ИНН 7701234567. Сумма: 100 000,00 руб.")
    assert result["inn"] == "7701234567"

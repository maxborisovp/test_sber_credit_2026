import glob
import os

import pytest

from src.classifier import classify

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

EXPECTED_TYPES = {
    "act_001.txt": "act",
    "act_002.txt": "act",
    "contract_001.txt": "contract",
    "invoice_001.txt": "invoice",
    "invoice_002.txt": "invoice",
    "spec_001.txt": "spec",
    "scan_ocr_001.txt": "unknown",
}


# --- Required assertion from the task brief -------------------------------
@pytest.mark.unit
def test_classify_invoice_basic():
    doc_type, confidence = classify("Счёт на оплату №12 от 01.03.2025 Поставщик ООО Ромашка")
    assert doc_type == "invoice"
    assert confidence > 0.5


# --- Full dataset coverage --------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("filename, expected_type", EXPECTED_TYPES.items())
def test_classify_sample_documents(filename, expected_type):
    text = open(os.path.join(DATA_DIR, filename), encoding="utf-8").read()
    doc_type, confidence = classify(text)
    assert doc_type == expected_type, f"{filename}: expected {expected_type}, got {doc_type}"
    assert 0.0 <= confidence <= 1.0


@pytest.mark.unit
def test_classify_unknown_on_empty_text():
    doc_type, confidence = classify("")
    assert doc_type == "unknown"


@pytest.mark.unit
def test_classify_does_not_confuse_cross_references():
    """A document that merely *mentions* another document type by name in
    passing (as almost every sample document does, e.g. "Основание:
    Договор поставки № 47/2025") must not be classified as that type just
    because the word appears somewhere in the body.
    """
    text = (
        "СЧЁТ НА ОПЛАТУ № 99 от 01.01.2025\n"
        "Основание: Договор поставки № 1 от 01.01.2025, Спецификация № 1\n"
        "Банковские реквизиты: р/с 40702810500000012345, БИК 044525225\n"
    )
    doc_type, _ = classify(text)
    assert doc_type == "invoice"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="requires GOOGLE_API_KEY"
)
def test_classify_via_llm_smoke():
    doc_type, confidence = classify("СПЕЦИФИКАЦИЯ № 1 к Договору поставки № 1 от 01.01.2025")
    assert doc_type == "spec"

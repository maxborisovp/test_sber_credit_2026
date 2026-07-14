"""
test_py

Pytest-тесты для py.

Тесты для *_llm функций подменяют langchain_core/langchain_google_genai через
monkeypatch и не требуют реального GOOGLE_API_KEY или сети.
"""

import json
import os
import sys
import types

import pytest

from src.extraction import (_extract_amount, _extract_contractor, _extract_date,
                            _extract_inn, _extract_subject, extract,
                            extract_keywords, extract_llm, _get_llm,
                            _normalize_numeric_amount, _normalize_year,
                            _try_make_date, _words_to_number, _empty_result)
from src.config import EXTRACT_FIELDS


# Тестовые документы (загружаются из ../data/*.txt)
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def _load_sample(filename):
    """
    Загружает тестовый документ из ../data/<filename> (относительно этого
    файла). Если файла нет — возвращает None; тесты, использующие
    соответствующую константу, в этом случае пропускаются (skip), а не падают.
    """
    path = os.path.join(_DATA_DIR, filename)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


ACT_001 = _load_sample("act_001.txt")
ACT_002 = _load_sample("act_002.txt")
CONTRACT_001 = _load_sample("contract_001.txt")
INVOICE_001 = _load_sample("invoice_001.txt")
INVOICE_002 = _load_sample("invoice_002.txt")
SPEC_001 = _load_sample("spec_001.txt")
SCAN_OCR = _load_sample("scan_ocr_001.txt")

_SAMPLE_DOCS = [
        pytest.param(ACT_001, {"amount":1250000.0, "date": "2025-03-24", "inn":"7701234567", "contractor":"ООО «ТехАгро»", "subject":'Карбамид марки Б'}, id='act_001'),
        pytest.param(ACT_002, {"amount":500000.0, "date":"2025-04-01", "inn":"504712345678", "contractor":"ИП Смирнов В.А.", "subject":'Внесение жидких комплексных удобрений (КАС-32) на площади 500 га'}, id='act_002'),
        pytest.param(CONTRACT_001, {"amount": 1250000.0, "date": "2025-03-01", "inn":"7701234567", "contractor":"ООО «ТехАгро»", "subject":'карбамид марки Б, ГОСТ 2081-2010'}, id='contract_001'),
        pytest.param(INVOICE_001, {"amount": 1250000.0, "date": "2025-03-03", "inn":"7701234567", "contractor":"ООО «ТехАгро»", "subject":'Карбамид марки Б, ГОСТ 2081-2010'}, id='invoice_001'),
        pytest.param(INVOICE_002, {"amount": 900000.0, "date": "2025-02-15", "inn":"5047123456", "contractor":'АО «АгроСнаб»', "subject":'поставка семян подсолнечника сорта «Командор», посевная партия 2025'}, id='invoice_002'),
        pytest.param(SPEC_001, {"amount": 1250000.0, "date": "2025-03-01", "inn":"7701234567", "contractor":"ООО «ТехАгро»", "subject":'Карбамид марки Б, ГОСТ 2081-2010'}, id='spec_001'),
        pytest.param(SCAN_OCR, {"amount": None, "date": "2025-03-01", "inn":None, "contractor":None, "subject":None}, id='scan_ocr_001'),
]
_AVAILABLE_SAMPLE_DOCS = [p for p in _SAMPLE_DOCS if p.values[0] is not None]


def _install_fake_langchain(monkeypatch, response_text):
    fake_core = types.ModuleType("langchain_core")
    fake_messages = types.ModuleType("langchain_core.messages")

    class HumanMessage:
        def __init__(self, content):
            self.content = content

    class SystemMessage:
        def __init__(self, content):
            self.content = content

    fake_messages.HumanMessage = HumanMessage
    fake_messages.SystemMessage = SystemMessage
    fake_core.messages = fake_messages

    fake_openai = types.ModuleType("langchain_google_genai")

    class _FakeResponse:
        def __init__(self, content):
            self.content = content

    class ChatGoogleGenerativeAI:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, messages):
            return _FakeResponse(response_text)

    fake_openai.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI

    monkeypatch.setitem(sys.modules, "langchain_core", fake_core)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", fake_messages)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_openai)



def test_empty_result_has_all_fields_none():
    result = _empty_result()
    assert set(result.keys()) == set(EXTRACT_FIELDS)
    assert all(v is None for v in result.values())


class TestNormalizeYear:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (2025, 2025),
            (0, 2000),
            (25, 2025),
            (69, 2069),
            (70, 1970),
            (99, 1999),
        ],
    )
    def test_normalize_year(self, raw, expected):
        assert _normalize_year(raw) == expected


class TestTryMakeDate:
    def test_valid_date(self):
        assert _try_make_date(2025, 3, 1) == "2025-03-01"

    def test_invalid_month_returns_none(self):
        assert _try_make_date(2025, 13, 1) is None

    def test_invalid_day_returns_none(self):
        assert _try_make_date(2025, 2, 30) is None


class TestExtractDate:
    def test_dot_format(self):
        assert _extract_date("Дата: 01.03.2025") == "2025-03-01"

    def test_dot_format_two_digit_year(self):
        assert _extract_date("Оплата до: 28.02.25") == "2025-02-28"

    def test_word_month_format(self):
        assert _extract_date("Составлено 1 марта 2025 г.") == "2025-03-01"

    def test_word_month_format_without_trailing_g(self):
        assert _extract_date("от 24 марта 2025") == "2025-03-24"

    def test_mdy_slash_format(self):
        assert _extract_date("Оплата до: 03/01/2025") == "2025-03-01"

    def test_picks_earliest_occurrence(self):
        text = "Договор от 01.03.2025, поставка не позднее 25.03.2025"
        assert _extract_date(text) == "2025-03-01"

    def test_no_date_returns_none(self):
        assert _extract_date("текст без даты") is None

    @pytest.mark.parametrize("text,expected_label", _AVAILABLE_SAMPLE_DOCS)
    def test_all_sample_documents(self, text, expected_label):
        assert _extract_date(text) == expected_label["date"]


class TestWordsToNumber:
    def test_hundreds_of_thousands(self):
        assert _words_to_number(["девятьсот", "тысяч"]) == 900000

    def test_tens_and_thousands(self):
        assert _words_to_number(["двести", "пятьдесят", "тысяч"]) == 250000

    def test_plain_hundred(self):
        assert _words_to_number(["сто"]) == 100

    def test_empty_list(self):
        assert _words_to_number([]) == 0

    def test_ignores_currency_words(self):
        assert _words_to_number(["сто", "рублей"]) == 100


class TestNormalizeNumericAmount:
    def test_space_thousands_comma_decimal(self):
        assert _normalize_numeric_amount("1 250 000,00") == 1250000.0

    def test_comma_thousands_dot_decimal(self):
        assert _normalize_numeric_amount("1,250,000.00") == 1250000.0

    def test_dot_decimal_no_thousands(self):
        assert _normalize_numeric_amount("500000.00") == 500000.0

    def test_plain_integer(self):
        assert _normalize_numeric_amount("1250000") == 1250000.0

    def test_nbsp_as_thousands_separator(self):
        assert _normalize_numeric_amount("1\u00A0250\u00A0000") == 1250000.0

    def test_empty_string_returns_none(self):
        assert _normalize_numeric_amount("") is None


class TestExtractAmount:
    def test_space_thousands_format(self):
        assert _extract_amount("Сумма: 1 250 000 руб.") == 1250000.0

    def test_dot_decimal_ruble_sign_format(self):
        assert _extract_amount("Total: 1250000.00 ₽") == 1250000.0

    def test_comma_thousands_rub_format(self):
        assert _extract_amount("Amount: 1,250,000.00 RUB") == 1250000.0

    def test_skips_parenthetical_between_number_and_currency(self):
        text = "Стоимость 1 250 000,00 (Один миллион двести пятьдесят тысяч) рублей 00 копеек"
        assert _extract_amount(text) == 1250000.0

    def test_picks_max_when_several_amounts_present(self):
        text = "Сумма 1 250 000,00 руб., в т.ч. НДС 208 333,33 руб."
        assert _extract_amount(text) == 1250000.0

    def test_falls_back_to_words_when_no_numeric_amount(self):
        text = "Стоимость услуг составляет девятьсот тысяч рублей 00/100."
        assert _extract_amount(text) == 900000.0

    def test_no_amount_returns_none(self):
        assert _extract_amount("текст без суммы") is None
    
    @pytest.mark.parametrize("text,expected_label", _AVAILABLE_SAMPLE_DOCS)
    def test_all_sample_documents(self, text, expected_label):
        assert _extract_amount(text) == expected_label["amount"]


class TestExtractInn:
    def test_simple_label(self):
        assert _extract_inn("ИНН 7701234567") == "7701234567"

    def test_inn_kpp_combined_label_picks_inn_not_kpp(self):
        text = "ИНН/КПП: 7701234567 / 770101001"
        assert _extract_inn(text) == "7701234567"

    def test_twelve_digit_inn(self):
        assert _extract_inn("ИНН 504712345678") == "504712345678"

    def test_no_inn_returns_none(self):
        assert _extract_inn("текст без ИНН") is None

    @pytest.mark.parametrize("text,expected_label", _AVAILABLE_SAMPLE_DOCS)
    def test_all_sample_documents(self, text, expected_label):
        assert _extract_inn(text) == expected_label["inn"]

class TestExtractContractor:
    def test_postavshik_label(self):
        text = "Поставщик: ООО «ТехАгро»\nПокупатель: УК «Русское поле»"
        assert _extract_contractor(text) == "ООО «ТехАгро»"

    def test_stops_at_double_space_two_column_layout(self):
        text = "Поставщик: ООО «ТехАгро»  Покупатель: УК «Русское поле»"
        assert _extract_contractor(text) == "ООО «ТехАгро»"

    def test_ispolnitel_label(self):
        text = "Исполнитель: ИП Смирнов В.А., ИНН 504712345678"
        assert _extract_contractor(text) == "ИП Смирнов В.А."

    def test_no_label_returns_none(self):
        assert _extract_contractor("текст без меток") is None
    
    @pytest.mark.parametrize("text,expected_label", _AVAILABLE_SAMPLE_DOCS)
    def test_all_sample_documents(self, text, expected_label):
        assert _extract_contractor(text) == expected_label["contractor"]


class TestExtractSubject:
    def test_explicit_label(self):
        text = "Предмет: поставка семян подсолнечника сорта «Командор»"
        assert _extract_subject(text) == "поставка семян подсолнечника сорта «Командор»"

    def test_parenthetical_after_verb(self):
        text = (
            "Поставщик обязуется передать в собственность Покупателя минеральные\n"
            "удобрения (карбамид марки Б, ГОСТ 2081-2010), а Покупатель обязуется"
        )
        assert _extract_subject(text) == "карбамид марки Б, ГОСТ 2081-2010"

    def test_table_row(self):
        text = "| 1 | Карбамид марки Б, ГОСТ 2081-2010 | тонна | 50 |"
        assert _extract_subject(text) == "Карбамид марки Б, ГОСТ 2081-2010"

    def test_numbered_item_skips_all_caps_heading(self):
        text = (
            "1. ПРЕДМЕТ ДОГОВОРА\n\n"
            "2. Внесение жидких комплексных удобрений на площади 500 га — 250 000,00 руб."
        )
        subject = _extract_subject(text)
        assert subject is not None
        assert "Внесение" in subject

    def test_hyphen_in_item_name_not_truncated(self):
        text = "1. Внесение жидких комплексных удобрений (КАС-32) на площади 500 га — 250 000,00 руб."
        subject = _extract_subject(text)
        assert subject == "Внесение жидких комплексных удобрений (КАС-32) на площади 500 га"

    def test_no_subject_returns_none(self):
        assert _extract_subject("текст без предмета") is None


class TestExtractKeywords:

    @pytest.mark.parametrize("text,expected_label", _AVAILABLE_SAMPLE_DOCS)
    def test_all_sample_documents(self, text, expected_label):
        result = extract_keywords(text)
        assert result["amount"] == expected_label["amount"]
        assert result["date"] == expected_label["date"]
        assert result["inn"] == expected_label["inn"]
        assert result["contractor"] == expected_label["contractor"]
        assert set(result.keys()) == set(EXTRACT_FIELDS)

    def test_missing_fields_are_none(self):
        result = extract_keywords("случайный текст без каких-либо полей")
        assert result == _empty_result()



class TestGetLlm:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            _get_llm()

    def test_returns_client_with_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        _install_fake_langchain(monkeypatch, response_text="{}")
        assert _get_llm() is not None


class TestExtractLlm:
    def test_parses_valid_json_response(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        response = json.dumps({
            "amount": 1250000.0,
            "date": "2025-03-01",
            "inn": "7701234567",
            "contractor": "ООО «ТехАгро»",
            "subject": "поставка удобрений",
        })
        _install_fake_langchain(monkeypatch, response_text=response)

        result = extract_llm("любой текст")
        assert result["amount"] == 1250000.0
        assert result["date"] == "2025-03-01"
        assert result["inn"] == "7701234567"
        assert result["contractor"] == "ООО «ТехАгро»"
        assert result["subject"] == "поставка удобрений"

    def test_null_fields_become_none(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        response = json.dumps({
            "amount": None, "date": None, "inn": None,
            "contractor": None, "subject": None,
        })
        _install_fake_langchain(monkeypatch, response_text=response)

        result = extract_llm("любой текст")
        assert result == _empty_result()


    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            extract_llm("любой текст")

    def test_raises_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        _install_fake_langchain(monkeypatch, response_text="это не json")
        with pytest.raises(json.JSONDecodeError):
            extract_llm("любой текст")


class TestExtract:

    def test_falls_back_to_keywords_without_api_key(self, monkeypatch):
        if CONTRACT_001 is None:
            pytest.skip("../data/contract_001.txt не найден")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert extract(CONTRACT_001) == extract_keywords(CONTRACT_001)

    def test_return_types(self):
        if CONTRACT_001 is None:
            pytest.skip("../data/contract_001.txt не найден")
        result = extract(CONTRACT_001)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(EXTRACT_FIELDS)
        assert isinstance(result["amount"], float)
        assert isinstance(result["date"], str)
        assert isinstance(result["inn"], str)
        assert isinstance(result["contractor"], str)
        assert isinstance(result["subject"], str)

    def test_empty_text_return_empty(self):
        assert extract("что угодно") == _empty_result()

    
    @pytest.mark.parametrize("text,expected_label", _AVAILABLE_SAMPLE_DOCS)
    def test_sample_documents_extracted_correctly(self, text, expected_label):
        result = extract(text)
        assert result["amount"] == expected_label["amount"]
        assert result["date"] == expected_label["date"]
        assert result["inn"] == expected_label["inn"]
        assert result["contractor"] == expected_label["contractor"]
        assert set(result.keys()) == set(EXTRACT_FIELDS)

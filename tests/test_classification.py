"""
Pytest-тесты для py.

Тесты для *_llm функций не требуют реального GOOGLE_API_KEY и не ходят в
сеть: langchain_core/langchain_google_genai подменяются на минимальные фейковые
модули через monkeypatch, что позволяет проверить логику формирования
запроса и разбора ответа изолированно от реального API.
"""

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classification import _eval_best, classify_keywords, classify_llm, classify, _get_llm


# ---------------------------------------------------------------------------
# Тестовые документы (загружаются из tests/../data/*.txt)
# ---------------------------------------------------------------------------

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

_SAMPLE_DOCS = [
        pytest.param(ACT_001, "act", id='act_001'),
        pytest.param(ACT_002, "act", id='act_002'),
        pytest.param(CONTRACT_001, "contract", id='contract_001'),
        pytest.param(INVOICE_001, "invoice", id='invoice_001'),
        pytest.param(INVOICE_002, "invoice", id='invoice_002'),
        pytest.param(SPEC_001, "spec", id='spec_001'),
]
# если файла в ../data/ нет - соответствующий кейс просто не попадает в
# параметризацию, и теста для него не будет (а не падение с ошибкой)
_AVAILABLE_SAMPLE_DOCS = [p for p in _SAMPLE_DOCS if p.values[0] is not None]


def _install_fake_langchain(monkeypatch, response_text):
    """
    Подменяет sys.modules['langchain_core.messages'] и
    sys.modules['langchain_google_genai'] минимальными фейками, чтобы classify_llm
    можно было протестировать без установленных пакетов и без сети.
    ChatGoogleGenerativeAI.invoke() всегда возвращает объект с заданным .content.
    """

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


class TestEvalBest:
    def test_clear_winner(self):
        scores = {"contract": 0.8, "spec": 0.1, "invoice": 0.05, "act": 0.05}
        label, confidence = _eval_best(scores)
        assert label == "contract"
        assert confidence == 0.8

    def test_gap_below_threshold_returns_unknown(self):
        scores = {"contract": 0.4, "spec": 0.35, "invoice": 0.15, "act": 0.1}
        label, confidence = _eval_best(scores)
        assert label == "unknown"
        assert confidence == 0.4

    def test_top1_below_min_confidence_returns_unknown(self):
        scores = {"contract": 0.03, "spec": 0.0, "invoice": 0.0, "act": 0.0}
        label, confidence = _eval_best(scores)
        assert label == "unknown"

    def test_empty_scores_returns_unknown(self):
        label, confidence = _eval_best({})
        assert (label, confidence) == ("unknown", 0.0)

    def test_all_zero_scores_returns_unknown(self):
        scores = {"contract": 0.0, "spec": 0.0, "invoice": 0.0, "act": 0.0}
        label, confidence = _eval_best(scores)
        assert (label, confidence) == ("unknown", 0.0)

    def test_single_category_with_no_competitor(self):
        scores = {"contract": 0.5}
        label, confidence = _eval_best(scores)
        assert label == "contract"
        assert confidence == 0.5


class TestClassifyKeywords:

    @pytest.mark.parametrize("text,expected_label", _AVAILABLE_SAMPLE_DOCS)
    def test_sample_documents_classified_correctly(self, text, expected_label):
        label, confidence = classify_keywords(text)
        assert label == expected_label
        assert confidence > 0.5

    def test_no_keywords_returns_unknown(self):
        label, confidence = classify_keywords("Просто случайный текст без маркеров.")
        assert (label, confidence) == ("unknown", 0.0)

    def test_empty_text_returns_unknown(self):
        label, confidence = classify_keywords("")
        assert (label, confidence) == ("unknown", 0.0)

    def test_mixed_markers_from_all_categories_returns_unknown(self):
        text = (
            "Договор поставки, счёт на оплату, акт выполненных работ "
            "и спецификация в одном документе."
        )
        label, confidence = classify_keywords(text)
        assert label == "unknown"

    def test_is_case_insensitive(self):
        if CONTRACT_001 is None:
            pytest.skip("../data/contract_001.txt не найден")
        label_lower, _ = classify_keywords(CONTRACT_001.lower())
        label_upper, _ = classify_keywords(CONTRACT_001.upper())
        assert label_lower == "contract"
        assert label_upper == "contract"




class TestGetLlm:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            _get_llm()

    def test_returns_client_with_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        _install_fake_langchain(monkeypatch, response_text="{}")
        llm = _get_llm()
        assert llm is not None


class TestClassifyLlm:
    def test_parses_valid_json_response(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        response = json.dumps({"contract": 0.9, "spec": 0.05, "invoice": 0.03, "act": 0.02})
        _install_fake_langchain(monkeypatch, response_text=response)

        label, confidence = classify_llm("любой текст")
        assert label == "contract"
        assert confidence == 0.9

    def test_ambiguous_llm_scores_return_unknown(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        response = json.dumps({"contract": 0.5, "spec": 0.45, "invoice": 0.03, "act": 0.02})
        _install_fake_langchain(monkeypatch, response_text=response)

        label, _ = classify_llm("любой текст")
        assert label == "unknown"

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            classify_llm("любой текст")

    def test_raises_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        _install_fake_langchain(monkeypatch, response_text="это не json")
        with pytest.raises(json.JSONDecodeError):
            classify_llm("любой текст")


class TestClassify:

    def test_falls_back_to_keywords_without_api_key(self, monkeypatch):
        if CONTRACT_001 is None:
            pytest.skip("../data/contract_001.txt не найден")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert classify(CONTRACT_001) == classify_keywords(CONTRACT_001)

    def test_return_types(self):
        label, confidence = classify("что угодно")
        assert isinstance(label, str)
        assert isinstance(confidence, float)
        assert label in ("contract", "act", "spec", "invoice", 'unknown')
        assert 0.0 <= confidence <= 1.0

    def test_empty_text_return_unknow_zero(self):
        assert classify("что угодно") == ('unknown', 0.0)

    
    @pytest.mark.parametrize("text,expected_label", _AVAILABLE_SAMPLE_DOCS)
    def test_sample_documents_classified_correctly(self, text, expected_label):
        label, confidence = classify(text)
        assert label == expected_label
        assert confidence > 0.5
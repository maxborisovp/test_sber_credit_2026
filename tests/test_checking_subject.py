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

from src.checking_subject import (_best_category_match, check_subject_keywords, 
                                  check_subject_llm, check_subject) 
from src.config import (CHECK_SUBJECT_NEGATIVE_CATEGORIES, CHECK_SUBJECT_POSITIVE_CATEGORIES)

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


class TestBestCategoryMatch:
    def test_finds_strongest_positive_match(self):
        match = _best_category_match(
            "поставка минеральных удобрений", CHECK_SUBJECT_POSITIVE_CATEGORIES
        )
        assert match is not None
        category, matched, weight = match
        assert category == "агрохимия"
        assert weight == 3

    def test_finds_strongest_negative_match(self):
        match = _best_category_match(
            "аренда офиса в бизнес-центре", CHECK_SUBJECT_NEGATIVE_CATEGORIES
        )
        assert match is not None
        category, matched, weight = match
        assert category == "аренда офиса/помещения"

    def test_no_match_returns_none(self):
        match = _best_category_match(
            "случайный текст без маркеров", CHECK_SUBJECT_POSITIVE_CATEGORIES
        )
        assert match is None

    def test_picks_highest_weight_when_several_categories_match(self):
        # "корм" (вес 2) встречается внутри "комбикорм" (вес 3) - должен выиграть более специфичный
        match = _best_category_match(
            "закупка комбикорма для скота", CHECK_SUBJECT_POSITIVE_CATEGORIES
        )
        assert match is not None
        _, _, weight = match
        assert weight == 3


class TestCheckSubjectKeywords:
    @pytest.mark.parametrize(
        "subject",
        [
            "Поставка минеральных удобрений (карбамид марки Б)",
            "Внесение жидких комплексных удобрений (КАС-32) на площади 500 га",
            "Поставка сельскохозяйственной техники: трактор МТЗ-82.1, 2 единицы",
            "Закупка семян пшеницы яровой",
            "Закупка кормов для крупного рогатого скота",
        ],
    )
    def test_agricultural_subjects_are_eligible(self, subject):
        eligible, confidence, explanation = check_subject_keywords(subject)
        assert eligible is True
        assert 0.0 < confidence <= 1.0
        assert explanation

    @pytest.mark.parametrize(
        "subject",
        [
            "Аренда офиса в бизнес-центре",
            "Оказание консультационных услуг по внедрению CRM-системы",
            "Ремонт легкового автомобиля директора",
            "Оплата рекламной кампании в интернете",
        ],
    )
    def test_non_agricultural_subjects_are_not_eligible(self, subject):
        eligible, confidence, explanation = check_subject_keywords(subject)
        assert eligible is False
        assert 0.0 < confidence <= 1.0
        assert explanation

    def test_ambiguous_subject_defaults_to_not_eligible_with_low_confidence(self):
        eligible, confidence, explanation = check_subject_keywords(
            "Оказание транспортных услуг"
        )
        assert eligible is False
        assert confidence == 0.5

    def test_empty_subject(self):
        eligible, confidence, explanation = check_subject_keywords("")
        assert eligible is False
        assert confidence == 0.5
        assert "не указан" in explanation

    def test_is_case_insensitive(self):
        lower = check_subject_keywords("удобрения")
        upper = check_subject_keywords("УДОБРЕНИЯ")
        assert lower[0] == upper[0] is True

    def test_genitive_case_seeds_recognized(self):
        # "семян" (родительный падеж мн.ч.) должно матчиться так же, как "семена"
        eligible, _, explanation = check_subject_keywords("закупка семян подсолнечника")
        assert eligible is True
        assert "семена" in explanation or "семян" in explanation

    def test_positive_match_wins_over_weaker_negative_match(self):
        # если позитивное совпадение сильнее (вес выше), должно побеждать оно
        eligible, _, _ = check_subject_keywords(
            "закупка удобрений для сельхозработ"
        )
        assert eligible is True


class TestCheckSubjectLlm:
    def test_parses_valid_json_response_eligible(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        response = json.dumps({
            "eligible": True,
            "confidence": 0.87,
            "explanation": "удобрения относятся к категории 'агрохимия'",
        })
        _install_fake_langchain(monkeypatch, response_text=response)

        eligible, confidence, explanation = check_subject_llm("удобрения")
        assert eligible is True
        assert confidence == 0.87
        assert "агрохимия" in explanation

    def test_parses_valid_json_response_not_eligible(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        response = json.dumps({
            "eligible": False,
            "confidence": 0.91,
            "explanation": "аренда офиса не относится к сельхоз-деятельности",
        })
        _install_fake_langchain(monkeypatch, response_text=response)

        eligible, confidence, explanation = check_subject_llm("аренда офиса")
        assert eligible is False
        assert confidence == 0.91


    def test_confidence_is_clamped_to_valid_range(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        response = json.dumps({"eligible": True, "confidence": 1.5, "explanation": "x"})
        _install_fake_langchain(monkeypatch, response_text=response)
        _, confidence, _ = check_subject_llm("x")
        assert confidence == 1.0

        response = json.dumps({"eligible": False, "confidence": -0.3, "explanation": "x"})
        _install_fake_langchain(monkeypatch, response_text=response)
        _, confidence, _ = check_subject_llm("x")
        assert confidence == 0.0

    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError):
            check_subject_llm("удобрения")

    def test_raises_on_malformed_json(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        _install_fake_langchain(monkeypatch, response_text="это не json")
        with pytest.raises(json.JSONDecodeError):
            check_subject_llm("удобрения")

    def test_raises_on_missing_required_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        response = json.dumps({"confidence": 0.5, "explanation": "нет поля eligible"})
        _install_fake_langchain(monkeypatch, response_text=response)
        with pytest.raises(KeyError):
            check_subject_llm("удобрения")



class TestCheckSubject:
    def test_falls_back_to_keywords_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert check_subject("удобрения") == check_subject_keywords("удобрения")

    def test_return_types(self):
        result = check_subject_keywords("что угодно")
        assert isinstance(result, tuple)
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        assert isinstance(result[2], str)

    def test_empty_subject_return_unknown(self):
        assert check_subject_keywords("") == (False, 0.5, "предмет оплаты не указан, отнести к сельхоз-программе невозможно")

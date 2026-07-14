"""
check_subject.py

Проверка соответствия предмета оплаты (subject) условиям льготной программы
кредитования: льготные кредиты выдаются под сельскохозяйственные нужды.

Два режима работы:
  1. LLM-режим (langchain + langchain_google_genai) — используется, если задан GOOGLE_API_KEY.
  2. Fallback-режим — проверка по базе ключевых категорий/фраз, не требует
     сети/ключей и работает полностью локально.

Публичная функция check_subject() сама выбирает режим и в случае любой ошибки
при обращении к LLM (нет ключа, нет пакета, нет сети, ошибка API)
прозрачно откатывается на fallback, так что код всегда возвращает результат
и запускается локально без ключей.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple
from src.config import (CHECK_SUBJECT_POSITIVE_CATEGORIES, CHECK_SUBJECT_NEGATIVE_CATEGORIES,
                        CHECK_SUBJECT_POSITIVE_CONFIDENCE_BY_WEIGHT,
                        CHECK_SUBJECT_NEGATIVE_CONFIDENCE_BY_WEIGHT,)
from src.llm_client import _response_content_to_text, _get_llm

def _best_category_match(
    text: str, categories: List[Tuple[str, List[Tuple[str, float]]]]
) -> Optional[Tuple[str, str, float]]:
    """
    Возвращает (категория, совпавший_фрагмент, вес) для самого сильного
    совпадения среди категорий, либо None, если совпадений не было.
    """
    best: Optional[Tuple[str, str, float]] = None
    for category, rules in categories:
        for pattern, weight in rules:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m and (best is None or weight > best[2]):
                best = (category, m.group(0), weight)
    return best


def check_subject_keywords(subject: str) -> Tuple[bool, float, str]:
    """
    Локальная проверка без LLM: ищет в тексте предмета оплаты маркеры
    "сельхоз"-категорий и заведомо непрофильных категорий, выбирает наиболее
    специфичное совпадение и формирует объяснение.
    """
    text = subject.lower().strip()
    if not text:
        return False, 0.5, "предмет оплаты не указан, отнести к сельхоз-программе невозможно"

    positive = _best_category_match(text, CHECK_SUBJECT_POSITIVE_CATEGORIES)
    negative = _best_category_match(text, CHECK_SUBJECT_NEGATIVE_CATEGORIES)

    if positive and (not negative or positive[2] >= negative[2]):
        category, matched, weight = positive
        confidence = CHECK_SUBJECT_POSITIVE_CONFIDENCE_BY_WEIGHT.get(weight, 0.6)
        explanation = f"'{matched.strip()}' относится к категории '{category}', которая покрывается сельхоз-программой"
        return True, confidence, explanation

    if negative:
        category, matched, weight = negative
        confidence = CHECK_SUBJECT_NEGATIVE_CONFIDENCE_BY_WEIGHT.get(weight, 0.6)
        explanation = f"'{matched.strip()}' относится к категории '{category}' и не относится к сельхоз-деятельности"
        return False, confidence, explanation

    return (
        False,
        0.5,
        "не удалось однозначно отнести предмет к сельскохозяйственной деятельности по имеющимся признакам",
    )


def check_subject_llm(subject: str) -> Tuple[bool, float, str]:
    """
    Проверка через LLM (langchain + langchain_google_genai). Требует GOOGLE_API_KEY.
    Бросает исключение при отсутствии ключа/пакета/сети или некорректном
    ответе модели — вызывающая сторона (check_subject()) обязана поймать её
    и откатиться на check_subject_keywords().
    """
    llm = _get_llm()
    my_prompt = """Ты — эксперт по льготной программе кредитования сельского
        хозяйства. По условиям программы льготные кредиты выдаются ТОЛЬКО под
        сельскохозяйственные нужды: закупка семян, удобрений, средств защиты
        растений, сельхозтехники, кормов, ГСМ для полевых работ, ветеринарные нужды,
        племенной скот, мелиорация/орошение, аренда сельхозземель, полевые работы
        (посев, обработка почвы, уборка урожая) и аналогичные направления.

        Тебе дают краткое описание предмета оплаты (subject) из документа. Определи:
        - eligible — соответствует ли предмет оплаты условиям программы (true/false)
        - confidence — уверенность в оценке, число от 0 до 1
        - explanation — краткое объяснение на русском языке (одно предложение),
        почему предмет относится или не относится к сельхоз-деятельности

        Верни ТОЛЬКО JSON без пояснений и без markdown, строго в формате:
        {"eligible": <true|false>, "confidence": <float>, "explanation": "<str>"}
        """
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke(
        [
            SystemMessage(content=my_prompt),
            HumanMessage(content=subject[:2000]),
        ]
    )

    content = _response_content_to_text(response.content).strip()
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()

    data = json.loads(content)
    eligible = bool(data["eligible"])
    confidence = float(data["confidence"])
    explanation = str(data["explanation"]).strip()

    confidence = max(0.0, min(1.0, confidence))

    return eligible, confidence, explanation


def check_subject(subject: str) -> Tuple[bool, float, str]:
    """
    Проверяет, подходит ли предмет оплаты под льготную программу кредитования
    сельхоз-нужд.

    Пытается использовать LLM (langchain_google_genai), если задан GOOGLE_API_KEY.
    При любой ошибке — прозрачно откатывается на локальную проверку по базе
    ключевых категорий (check_subject_keywords), поэтому функция всегда
    возвращает результат и код можно запускать без ключей и сети.

    :param subject: текст предмета оплаты, например "поставка минеральных удобрений"
    :return: (eligible, confidence, explanation)
        eligible — True, если предмет относится к сельхоз-нуждам
        confidence — уверенность в оценке, 0..1
        explanation — краткое объяснение на русском языке
    """
    if not subject or not subject.strip():
        return False, 0.5, "предмет оплаты не указан, отнести к сельхоз-программе невозможно"

    try:
        return check_subject_llm(subject)
    except Exception as e:
        print(e)
        return check_subject_keywords(subject)


"""
checking_subject.py

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
from src.config import (CHECK_SUBJECT_POSITIVE_CATEGORIES, CHECK_SUBJECT_NEGATIVE_CATEGORIES,)
from src.llm_client import _response_content_to_text, _get_llm

def _category_match(
    text: str, categories: List[Tuple[str, List[Tuple[str, float]]]]
) -> Tuple[float, List[Tuple[str, str, float]]]:
    """
    Подсчитывает суммарный вес всех совпадений и возвращает список найденных совпадений.
    Возвращает (общий_вес, список_совпадений).
    """
    matches = []
    total_weight = 0.0
    
    for category, rules in categories:
        for pattern, weight in rules:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                matches.append((category, m.group(0), weight))
                total_weight += weight
    
    return total_weight, matches

def check_subject_keywords(subject: str) -> Tuple[bool, float, str]:
    """
    Локальная проверка без LLM: ищет в тексте предмета оплаты маркеры
    "сельхоз"-категорий и заведомо непрофильных категорий, вычисляет
    суммарные веса и определяет confidence на основе их соотношения.
    """
    text = subject.lower().strip()
    if not text:
        return False, 0.5, "предмет оплаты не указан, отнести к сельхоз-программе невозможно"
    
    positive_weight, positive_matches = _category_match(text, CHECK_SUBJECT_POSITIVE_CATEGORIES)
    negative_weight, negative_matches = _category_match(text, CHECK_SUBJECT_NEGATIVE_CATEGORIES)

    all_matches = []
    if positive_matches:
        pos_summary = ", ".join([f"'{m[1]}'" for m in positive_matches[:3]])
        if len(positive_matches) > 3:
            pos_summary += f" и еще {len(positive_matches) - 3} совпад."
        all_matches.append(f"сельхоз-маркеры: {pos_summary}")
    
    if negative_matches:
        neg_summary = ", ".join([f"'{m[1]}'" for m in negative_matches[:3]])
        if len(negative_matches) > 3:
            neg_summary += f" и еще {len(negative_matches) - 3} совпад."
        all_matches.append(f"несельхоз-маркеры: {neg_summary}")
    
    matches_summary = "; ".join(all_matches) if all_matches else "совпадений не найдено"

    total_weight = positive_weight + negative_weight
    confidence = positive_weight / total_weight if total_weight != 0 else 0.5
    confidence = max(0.1, min(0.9, confidence))
    
    if confidence > 0.6:
        return True, confidence, f"{matches_summary}"
    elif confidence < 0.4:
        return False, 1.0-confidence, f"{matches_summary}"
    else:
        return False, confidence, f"спорный/неочевидный случай: {matches_summary}"



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
    except Exception:
        return check_subject_keywords(subject)


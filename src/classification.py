"""
classify.py

Классификация типа документа: contract, spec, invoice, act, unknown.

Два режима работы:
  1. LLM-режим (langchain + langchain_google_genai) — используется, если задан GOOGLE_API_KEY.
  2. Fallback-режим — классификация по ключевым словам, не требует сети/ключей
     и работает полностью локально.

Функция classify() сама выбирает режим и в случае любой ошибки при
обращении к LLM (нет ключа, нет пакета, нет сети, ошибка API)
прозрачно откатывается на fallback, так что код всегда возвращает результат
и запускается локально без ключей.
"""

import re
import os
import json
from typing import Dict, Tuple
from src.config import (CLASSIFY_KEYWORD_RULES, CLASSIFY_MIN_BEST_CONFIDENCE,
                        CLASSIFY_GAP_THRESHOLD, CLASSIFY_LABELS, CLASSIFY_SYSTEM_PROMPT)


def eval_best(scores: Dict[str, float]) -> Tuple[str, float]:
    """
    Функция принятия решения по словарю {категория: нормированная оценка 0..1}.
    Если top1 confidence меньше минимального порога или confidence top1-top2
    меньше порога разницы -> категория unknown.
    Используется и keywords-, и LLM-классификатором, чтобы работало одинаково 
    в обоих режимах.
    """
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    best_type, best_score = ranked[0]
    second_score = ranked[1][1]
    
    gap = best_score - second_score
    
    if best_score < CLASSIFY_MIN_BEST_CONFIDENCE or gap < CLASSIFY_GAP_THRESHOLD:
        return "unknown", round(best_score, 3)
    return best_type, round(best_score, 3)

def classify_keywords(text: str) -> Tuple[str, float]:
    """
    Локальная классификация без LLM: подсчёт взвешенных вхождений ключевых
    фраз/слов по каждой категории типа документа, нормализация в доли и выбор 
    победителя с учётом порога разницы между top-1 и top-2.
    """
    scores = {}
    text_lower = text.lower()
    
    for doc_type, rules in CLASSIFY_KEYWORD_RULES.items():
        score = 0.0
        for pattern, weight in rules:
            matches = re.findall(pattern, text_lower, flags=re.IGNORECASE)
            if matches:
                score += weight * len(matches)
        scores[doc_type] = score
    
    total = sum(scores.values())
    if total == 0:
        return 'unknown', 0.0
    
    normalized_scores = {doc_type: score / total for doc_type, score in scores.items()}
    best_type, best_score = eval_best(normalized_scores)
    return best_type, best_score




def _get_llm():
    """
    Создаёт объект ChatGoogleGenerativeAI. Бросает исключение, 
    если ключ не задан или пакет не установлен — это сигнал 
    вызывающему коду сделать fallback.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set")

    # Локальный импорт: модуль должен нормально грузиться и без установленного
    # langchain_google_genai, если используется только fallback-режим.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)


def classify_llm(text: str) -> Tuple[str, float]:
    """
    Классификация через LLM (langchain + langchain_google_genai). Требует
    GOOGLE_API_KEY в окружении. Бросает исключение при отсутствии ключа,
    пакета, ошибке сети/API или некорректном ответе модели — вызывающая
    сторона (classify()) обязана поймать её и откатиться на classify_keywords().
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()

    response = llm.invoke(
        [
            SystemMessage(content=CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=text[:8000]),  # на всякий случай ограничим длину
        ]
    )

    content = response.content.strip()
    # На случай, если модель всё же обернула JSON в ```json ... ```
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()

    scores = json.loads(content)
    scores = {label: float(scores.get(label, 0.0)) for label in CLASSIFY_LABELS}
    best_type, best_score = eval_best(scores)
    return best_type, best_score

def classify(text: str) -> Tuple[str, float]:
    """
    Классифицирует тип документа: contract, spec, invoice, act или unknown.

    Пытается использовать LLM (langchain_google_genai), если задан GOOGLE_API_KEY.
    При любой ошибке (нет ключа, пакет не установлен, ошибка сети/API,
    некорректный ответ модели) — прозрачно откатывается на локальную
    классификацию по ключевым словам (classify_keywords), поэтому функция
    всегда возвращает результат, и код можно запускать без ключей и сети.

    :param text: текст документа
    :return: (label, confidence), label из {"contract","spec","invoice","act","unknown"}
    """
    #на случай входа пустого текста 
    if not text or not text.strip():
        return "unknown", 0.0
    
    try:
        return classify_llm(text)
    except Exception:
        return classify_keywords(text)

if __name__ == "__main__":
    # os.environ["GOOGLE_API_KEY"] = ""
    os.environ.pop("GOOGLE_API_KEY", None)
    filenames = ["../data/act_001.txt", "../data/act_002.txt", "../data/contract_001.txt", "../data/invoice_001.txt", "../data/invoice_002.txt", "../data/spec_001.txt", "../data/scan_ocr_001.txt"]
    for filename in filenames: 
        with open(filename, 'r', encoding='utf-8') as file:
            doc_text = file.read()
        result = classify(doc_text)
        print(f"\nText - {filename},  result - {result}")

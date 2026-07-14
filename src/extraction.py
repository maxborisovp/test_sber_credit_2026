"""
extract.py

Извлечение полей из текста документа: amount, date, inn, contractor, subject.

Два режима работы:
  1. LLM-режим (langchain + langchain_google_genai) — используется, если задан GOOGLE_API_KEY.
  2. Fallback-режим — извлечение регулярными выражениями, не требует сети/ключей
     и работает полностью локально.

Функция extract() сама выбирает режим и в случае любой ошибки при
обращении к LLM (нет ключа, нет пакета, нет сети, ошибка API)
прозрачно откатывается на fallback, так что код всегда возвращает результат
и запускается локально без ключей.
"""

import json
import os
import re
from datetime import date as _date
from typing import Any, Dict, Optional
from src.config import (EXTRACT_FIELDS, EXTRACT_MONTHS_RU, EXTRACT_WORD_NUMS,
                    EXTRACT_WORD_SCALES, EXTRACT_CONTRACTOR_LABELS,)


def _empty_result() -> Dict[str, Any]:
    return {field: None for field in EXTRACT_FIELDS}


def _normalize_year(year: int) -> int:
    if year < 100:
        return 2000 + year if year < 70 else 1900 + year
    return year


def _try_make_date(year: int, month: int, day: int) -> Optional[str]:
    try:
        return _date(year, month, day).isoformat()
    except ValueError:
        return None


def _extract_date(text: str) -> Optional[str]:
    """
    Находит первую по позиции в тексте дату (обычно это дата/номер документа,
    указанная в начале) и возвращает её в формате ISO (YYYY-MM-DD).
    """
    # три основных формата дат:
    #   1) 01.03.2025 / 1.3.25            -> день.месяц.год (европейский/русский)
    #   2) 1 марта 2025 г.                -> день + месяц словом + год
    #   3) 03/01/2025                     -> месяц/день/год (американский формат)
    date_patterns = [
        ("dmy_dot", re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")),
        ("dmy_word", re.compile(
            r"\b(\d{1,2})\s+(" + "|".join(EXTRACT_MONTHS_RU.keys()) + r")\s+(\d{4})\s*г?\.?",
            re.IGNORECASE,
        )),
        ("mdy_slash", re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")),
    ]
    
    candidates = []

    for kind, pattern in date_patterns:
        for m in pattern.finditer(text):
            if kind == "dmy_dot":
                day, month, year = int(m.group(1)), int(m.group(2)), _normalize_year(int(m.group(3)))
                iso = _try_make_date(year, month, day)
            elif kind == "dmy_word":
                day = int(m.group(1))
                month = EXTRACT_MONTHS_RU[m.group(2).lower()]
                year = int(m.group(3))
                iso = _try_make_date(year, month, day)
            else:  # mdy_slash
                month, day, year = int(m.group(1)), int(m.group(2)), _normalize_year(int(m.group(3)))
                iso = _try_make_date(year, month, day)

            if iso:
                candidates.append((m.start(), iso))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def _words_to_number(words: list) -> int:
    total = 0
    chunk = 0
    for w in words:
        if w in ("рубль", "рублей", "рубля", "копеек", "копейки", "копейка"):
            continue
        if w in EXTRACT_WORD_SCALES:
            chunk = chunk or 1
            total += chunk * EXTRACT_WORD_SCALES[w]
            chunk = 0
        elif w in EXTRACT_WORD_NUMS:
            chunk += EXTRACT_WORD_NUMS[w]
    total += chunk
    return total


def _normalize_numeric_amount(raw: str) -> Optional[float]:
    """
    Приводит захваченную числовую строку (с разными разделителями тысяч/десятичных)
    к float. Поддерживает:
      '1 250 000,00' (пробел - тысячи, запятая - десятичные)
      '1,250,000.00' (запятая - тысячи, точка - десятичные)
      '500000.00' (без разделителя тысяч, точка - десятичные)
      '1250000' (целое число без разделителей)
    """
    s = raw.strip().replace("\u00A0", " ")
    # найдём последний разделитель (, или .) — если после него ровно 2 цифры
    # до конца строки, это десятичная часть
    last_sep_match = None
    for m in re.finditer(r"[.,]", s):
        last_sep_match = m

    if last_sep_match:
        pos = last_sep_match.start()
        frac = s[pos + 1:]
        if re.fullmatch(r"\d{2}", frac):
            integer_part = re.sub(r"[^\d]", "", s[:pos])
            fractional_part = frac
            if not integer_part:
                return None
            return float(f"{integer_part}.{fractional_part}")

    # иначе весь разделители - разделители тысяч, десятичной части нет
    integer_part = re.sub(r"[^\d]", "", s)
    if not integer_part:
        return None
    return float(integer_part)


def _extract_amount(text: str) -> Optional[float]:
    """
    Ищет сумму в тексте. Сначала пробует числовые форматы с валютным маркером
    (руб./₽/RUB), выбирая наибольшее найденное значение (обычно это итоговая
    сумма, а не промежуточная строка в таблице). Если числовых сумм не найдено —
    пробует сумму, записанную словами.
    """
    
    numeric_candidates = []

    # между числом и валютой иногда встречается пояснение суммы прописью в скобках,
    # например "1 250 000,00 (Один миллион ...) рублей" — пропускаем его
    amount_num_re = re.compile(
        r"(\d[\d\s\u00A0.,]*\d|\d)\s*(?:\([^)]*\)\s*)?" + r"(?:руб(?:лей|ль|\.)?|₽|RUB)",
        re.IGNORECASE,
    )
    for m in amount_num_re.finditer(text):
        value = _normalize_numeric_amount(m.group(1))
        if value is not None:
            numeric_candidates.append(value)

    if numeric_candidates:
        return max(numeric_candidates)

    amount_words_re = re.compile(
        r"((?:(?:" + "|".join(sorted(list(EXTRACT_WORD_NUMS.keys()) + list(EXTRACT_WORD_SCALES.keys()), key=len, reverse=True)) + r")\s*)+)"
        r"рубл(?:ей|я|ь)",
        re.IGNORECASE,
    )
    for m in amount_words_re.finditer(text):
        words = m.group(1).lower().split()
        value = _words_to_number(words)
        if value > 0:
            return float(value)

    return None


def _extract_inn(text: str) -> Optional[str]:
    inn_re = re.compile(r"ИНН(?:/КПП)?\s*[:№]?\s*(\d{12}|\d{10})(?!\d)")
    m = inn_re.search(text)
    return m.group(1) if m else None


def _extract_contractor(text: str) -> Optional[str]:
    contractor_re = re.compile(
        r"(?:^|\n)\s*(?:" + "|".join(EXTRACT_CONTRACTOR_LABELS) + r")\s*:\s*([^\n,]+?)(?=\s{2,}|,|\n|$)",
    )
    m = contractor_re.search(text)
    if m:
        return m.group(1).strip().rstrip(",")
    return None


def _looks_like_heading(candidate: str) -> bool:
    letters = [c for c in candidate if c.isalpha()]
    return bool(letters) and all(c.upper() == c for c in letters)


def _extract_subject(text: str) -> Optional[str]:
    subject_label_re = re.compile(r"Предмет(?:\s+договора)?\s*:\s*([^\n\.]+)", re.IGNORECASE)
    m = subject_label_re.search(text)
    if m:
        return m.group(1).strip()

    subject_after_verb_re = re.compile(
        r"(?:передать|поставить|поставк[аи]|выполнить|оказать)[\s\S]{0,80}?\(([^)]{5,120})\)",
        re.IGNORECASE,
    )
    m = subject_after_verb_re.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate and not _looks_like_heading(candidate):
            return candidate

    subject_table_row_re = re.compile(r"\|\s*\d+\s*\|\s*([^\|]+?)\s*\|")
    m = subject_table_row_re.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate and not _looks_like_heading(candidate):
            return candidate

    subject_numbered_item_re = re.compile(r"(?:^|\n)\s*\d+\.\s+([^\n—]+)")
    for m in subject_numbered_item_re.finditer(text):
        candidate = m.group(1).strip()
        if candidate and len(candidate) >= 8 and not _looks_like_heading(candidate):
            return candidate

    return None


def extract_keywords(text: str) -> Dict[str, Any]:
    """
    Локальное извлечение полей без LLM: регулярные выражения для каждого поля.
    Поля, которые не удалось найти, возвращаются как None.
    """
    return {
        "amount": _extract_amount(text),
        "date": _extract_date(text),
        "inn": _extract_inn(text),
        "contractor": _extract_contractor(text),
        "subject": _extract_subject(text),
    }


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

    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)

def extract_llm(text: str) -> Dict[str, Any]:
    """
    Извлечение полей через LLM (langchain + langchain_google_genai). Требует
    GOOGLE_API_KEY. Бросает исключение при отсутствии ключа/пакета/сети или
    некорректном ответе модели — вызывающая сторона (extract()) обязана
    поймать её и откатиться на extract_keywords().
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _get_llm()
    my_prompt = """Ты извлекаешь структурированные поля из текста делового
        документа на русском языке (договор, спецификация, счёт, акт и т.п.).

        Извлеки следующие поля:
        - amount — итоговая сумма документа (число, без валюты и разделителей тысяч,
        например 1250000.00). Если сумм несколько — бери итоговую/общую сумму.
        - date — дата документа в формате ISO (YYYY-MM-DD)
        - inn — ИНН контрагента (10 или 12 цифр, строка)
        - contractor — наименование контрагента (поставщика/исполнителя/продавца)
        - subject — краткий предмет документа (что поставляется/какие работы выполняются)

        Если поле не удаётся найти в тексте — верни null для него.

        Верни ТОЛЬКО JSON без пояснений и без markdown, строго в формате:
        {"amount": <float|null>, "date": "<YYYY-MM-DD>"|null, "inn": "<str>"|null, "contractor": "<str>"|null, "subject": "<str>"|null}
        """
    response = llm.invoke(
        [
            SystemMessage(content=my_prompt),
            HumanMessage(content=text[:8000]),
        ]
    )

    content = response.content.strip()
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()

    data = json.loads(content)
    result = _empty_result()
    for field in EXTRACT_FIELDS:
        value = data.get(field)
        result[field] = value if value not in ("", "null") else None

    if result["amount"] is not None:
        try:
            result["amount"] = float(result["amount"])
        except (TypeError, ValueError):
            result["amount"] = None

    return result


def extract(text: str) -> Dict[str, Any]:
    """
    Извлекает поля amount, date, inn, contractor, subject из текста документа.

    Пытается использовать LLM (langchain_google_genai), если задан GOOGLE_API_KEY.
    При любой ошибке — откатывается на локальное извлечение
    регулярными выражениями (extract_keywords), поэтому функция всегда
    возвращает результат и код можно запускать без ключей и сети.

    Поля, которые не удалось найти, возвращаются как None.
    :param text: текст документа
    :return: dict = {label: value}, label из ("amount", "date", "inn", "contractor", "subject")
    """
    if not text or not text.strip():
        return _empty_result()

    try:
        return extract_llm(text)
    except Exception:
        return extract_keywords(text)


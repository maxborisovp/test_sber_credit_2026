"""check_subject(subject) -> (matches, confidence, reason).

Determines whether a payment's subject (predmet oplaty) fits the target
use of an agricultural (сельхоз) preferential-credit program.

Two strategies, same fallback philosophy as extractor.py / classifier.py:
1. LLM (few-shot prompt, JSON output) when GOOGLE_API_KEY is configured.
2. Keyword/category matching - always available, zero dependencies.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# LLM backend: Pydantic-схема + JsonOutputParser + PromptTemplate — тот же
# паттерн, что и в extractor.py/classifier.py (см. RESULTS.md).
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.output_parsers import JsonOutputParser  # type: ignore
    from langchain_core.prompts import PromptTemplate  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore

    _LANGCHAIN_AVAILABLE = True

    class SubjectCheckResult(BaseModel):
        matches: bool = Field(description="Подходит ли предмет оплаты под программу")
        confidence: float = Field(description="Уверенность от 0.0 до 1.0", ge=0.0, le=1.0)
        reason: str = Field(description="Краткое обоснование вывода")

    _subject_parser = JsonOutputParser(pydantic_object=SubjectCheckResult)
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore
    _subject_parser = None  # type: ignore
    _LANGCHAIN_AVAILABLE = False

# --------------------------------------------------------------------------
# Category definitions for the keyword fallback
# --------------------------------------------------------------------------
# Each entry: category label -> list of keyword stems (lowercase). A stem
# matches if it appears as a substring of any word in the (lowercased,
# punctuation-stripped) subject text.
_ALLOWED_CATEGORIES: dict[str, list[str]] = {
    "агрохимия (удобрения, СЗР)": [
        "удобрен", "агрохим", "гербицид", "фунгицид", "пестицид",
        "средств.{0,3}защит.{0,3}растен", "кас-32", "кас32",
        # common fertilizer/agrochemical product names seen on invoices,
        # which often name the product rather than say "удобрение":
        "карбамид", "аммофос", "нитроаммофоск", "суперфосфат", "хлористы\\w*\\s*кали",
    ],
    "семена и посадочный материал": [
        "семен", "посевн", "посадочн", "сорта",
    ],
    "сельхозтехника (обслуживание, ремонт)": [
        "трактор", "комбайн", "сеялк", "опрыскивател", "культиватор",
        "зерноуборочн", "мтз", "john\\s*deere",
    ],
    "топливо и ГСМ": [
        "дизел", "топлив", "\\bгсм\\b", "бензин",
    ],
    "запчасти": [
        "запчаст", "запасн\\w*\\s*част",
    ],
    "полевые/агрохимические работы": [
        "полев\\w*\\s*работ", "обработк\\w*\\s*почв", "внесени\\w*\\s*удобрен",
        "агрохимическ\\w*\\s*работ", "агрохимическ\\w*\\s*обследован",
        "почвенн\\w*\\s*анализ", "уборк\\w*\\s*урожа",
    ],
    "страхование урожая": [
        "страхован\\w*\\s*урожа",
    ],
}

# Phrases that clearly place the subject OUTSIDE the program (office costs,
# generic professional/administrative services, unrelated retail/IT).
# Each entry is (regex, human-readable label used in the explanation).
_BLOCKED_PHRASES: list[tuple[str, str]] = [
    ("аренд\\w*\\s*офис", "аренда офиса"),
    ("офисн\\w*\\s*мебел", "офисная мебель"),
    ("канцеляр", "канцелярские товары"),
    ("юридическ", "юридические услуги"),
    ("консультационн\\w*\\s*услуг", "консультационные услуги общего характера"),
    ("разработк\\w*\\s*сайт", "разработка сайта"),
    ("\\bseo\\b", "SEO-продвижение"),
    ("клининг", "клининговые услуги"),
    ("уборк\\w*\\s*(администра|помещен|здани|офис)", "уборка помещений"),
    ("обучени", "обучение персонала"),
    ("тренинг", "обучение персонала"),
]

# Role-specific keywords that, when present, indicate the service IS
# agriculture-specific even though a generic blocked word (e.g.
# "консультационные") might also loosely apply - these override the block.
_POSITIVE_OVERRIDE = ["агроном"]

_WORD_RE = re.compile(r"[а-яё\-]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


def _count_matches(text: str, stems: list[str]) -> int:
    count = 0
    for stem in stems:
        if re.search(stem, text, re.IGNORECASE):
            count += 1
    return count


if _LANGCHAIN_AVAILABLE:
    _subject_prompt = PromptTemplate(
        template=(
            "Ты помогаешь проверить, подходит ли предмет оплаты под льготную "
            "сельскохозяйственную кредитную программу (покрывает: удобрения/СЗР, "
            "семена, сельхозтехнику и её обслуживание, топливо для сельхозработ, "
            "запчасти, полевые/агрохимические работы, страхование урожая). "
            "НЕ подходят: аренда офисов, юридические/консультационные услуги "
            "общего характера, офисные товары, IT/сайты, клининг, обучение персонала.\n\n"
            "Примеры:\n"
            '"Поставка минеральных удобрений" -> matches=true, "удобрения относятся к агрохимии"\n'
            '"Аренда офисного помещения" -> matches=false, "аренда офиса не относится к сельхозу"\n\n'
            "{format_instructions}\n\n"
            'Предмет оплаты: "{subject}"\n'
        ),
        input_variables=["subject"],
        partial_variables={"format_instructions": _subject_parser.get_format_instructions()},
    )


def _check_via_llm(subject: str) -> Optional[tuple[bool, float, str]]:
    if not _LANGCHAIN_AVAILABLE:
        return None
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        chain = _subject_prompt | llm | _subject_parser
        data = chain.invoke({"subject": subject})
        return bool(data["matches"]), float(data["confidence"]), str(data["reason"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM subject check failed, falling back to keywords: %s", exc)
        return None


def _check_via_keywords(subject: str) -> tuple[bool, float, str]:
    text = _normalize(subject)
    has_override = any(re.search(kw, text) for kw in _POSITIVE_OVERRIDE)

    if not has_override:
        for phrase, readable in _BLOCKED_PHRASES:
            if re.search(phrase, text):
                return (
                    False,
                    0.9,
                    f"формулировка указывает на непрофильные расходы («{readable}»), "
                    f"не связанные напрямую с сельхоз-деятельностью",
                )

    best_category = None
    best_count = 0
    for category, stems in _ALLOWED_CATEGORIES.items():
        count = _count_matches(text, stems)
        if count > best_count:
            best_count = count
            best_category = category

    if best_category is not None:
        confidence = min(0.95, 0.6 + 0.15 * best_count)
        return True, round(confidence, 2), f"предмет относится к категории «{best_category}»"

    return (
        False,
        0.35,
        "не найдено явных признаков соответствия ни одной из разрешённых категорий "
        "программы - рекомендуется ручная проверка",
    )


def check_subject(subject: str) -> tuple[bool, float, str]:
    """Return (matches, confidence, reason) for whether *subject* (a
    payment's stated purpose) fits the target use of the agricultural
    preferential-credit program.
    """
    if not subject or not subject.strip():
        return False, 0.3, "предмет оплаты не указан"
    result = _check_via_llm(subject)
    if result is not None:
        return result
    return _check_via_keywords(subject)

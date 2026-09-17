"""Field extraction: extract(text) -> dict.

Two extraction strategies are implemented:

1. LLM-based (``_extract_via_llm``): uses LangChain + Gemini
   (``gemini-3.5-flash``) with a structured JSON response. Used when
   ``GOOGLE_API_KEY`` is set and the ``langchain-google-genai`` package is
   installed.
2. Regex/rule-based (``_extract_via_rules``): zero dependencies, zero
   network calls. Always available and used as a fallback whenever the LLM
   path is unavailable or raises an error.

``extract()`` tries (1) first and transparently falls back to (2) - this
mirrors the "MUST work locally without an API key" requirement from the
task brief.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from src.utils import fix_ocr_digits, find_first_date, parse_amount_number, parse_amount_words

logger = logging.getLogger(__name__)

FIELDS = ("amount", "date", "inn", "contractor", "subject")

# --------------------------------------------------------------------------
# Optional LLM backend - degrades gracefully if the packages aren't
# installed. Structured output is built the way the LangGraph course
# (habr.com/ru/companies/amvera/articles/948000, часть 2, раздел
# "Структурированные JSON-ответы") recommends: Pydantic-модель описывает
# схему -> JsonOutputParser генерирует инструкции и парсит ответ ->
# PromptTemplate подставляет инструкции в промпт -> всё собирается в
# LCEL-цепочку `prompt | llm | parser`. Это заменяет более хрупкий вариант
# "попроси модель вернуть JSON и вручную обрежь ```json/```" из первой
# версии этого файла: цепочка не полагается на дисциплину модели, а даёт
# ей явную JSON-схему и парсит/валидирует результат через Pydantic.
# --------------------------------------------------------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.output_parsers import JsonOutputParser  # type: ignore
    from langchain_core.prompts import PromptTemplate  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore

    _LANGCHAIN_AVAILABLE = True

    class ExtractedFields(BaseModel):
        """Pydantic-схема ответа LLM — задаёт JsonOutputParser структуру,
        которую тот требует от модели и по которой валидирует ответ."""

        amount: Optional[float] = Field(
            default=None,
            description="Итоговая сумма к оплате с учётом НДС, либо null, если не найдена",
        )
        date: Optional[str] = Field(
            default=None,
            description="Дата документа в формате YYYY-MM-DD, либо null",
        )
        inn: Optional[str] = Field(
            default=None,
            description="ИНН контрагента (поставщика/исполнителя), строка из 10 или 12 цифр, либо null",
        )
        contractor: Optional[str] = Field(
            default=None,
            description="Название контрагента-поставщика/исполнителя, либо null",
        )
        subject: Optional[str] = Field(
            default=None,
            description="Краткое описание предмета оплаты (товар/услуга), либо null",
        )

    _extraction_parser = JsonOutputParser(pydantic_object=ExtractedFields)
    _extraction_prompt = PromptTemplate(
        template=(
            "Ты — модуль извлечения данных из финансовых документов "
            "(договор, спецификация, счёт, УПД/акт).\n"
            "Извлеки из текста документа поля amount, date, inn, contractor, subject.\n\n"
            "{format_instructions}\n\n"
            "Текст документа:\n---\n{text}\n---\n"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": _extraction_parser.get_format_instructions()},
    )
except ImportError:  # pragma: no cover - exercised when dependencies missing
    ChatGoogleGenerativeAI = None  # type: ignore
    _extraction_parser = None  # type: ignore
    _extraction_prompt = None  # type: ignore
    _LANGCHAIN_AVAILABLE = False


def _extract_via_llm(text: str) -> Optional[dict[str, Any]]:
    """Attempt extraction via Gemini. Returns None on any failure so the
    caller can fall back to the rule-based path."""
    if not _LANGCHAIN_AVAILABLE:
        return None
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        # LCEL-цепочка: промпт (с готовыми JSON-инструкциями) -> модель ->
        # парсер, который сразу валидирует ответ по схеме ExtractedFields.
        chain = _extraction_prompt | llm | _extraction_parser
        data = chain.invoke({"text": text})
        return {k: data.get(k) for k in FIELDS}
    except Exception as exc:  # noqa: BLE001 - any LLM/network/parse failure -> fallback
        logger.warning("LLM extraction failed, falling back to rules: %s", exc)
        return None


# --------------------------------------------------------------------------
# Rule-based fallback
# --------------------------------------------------------------------------
_INN_RE = re.compile(r"ИНН\S*[:\s]*([\dOolI]{10,12})", re.IGNORECASE)

# Amount candidates: a currency-shaped number optionally followed by a
# currency marker (руб / ₽ / RUB, or a homoglyph-corrupted OCR rendering of
# "руб" such as "pyб"). We capture the *label context* preceding each
# candidate to score it. The leading/trailing character class also accepts
# O/o/l/I so OCR-mangled digit groups (e.g. "l 25O OOO") are still found.
_AMOUNT_CANDIDATE_RE = re.compile(
    r"([\dOolI][\dOolI\s.,]{1,}[\dOolI])\s*(?:руб|₽|RUB|[рp][уy][бb6])",
    re.IGNORECASE,
)

_AMOUNT_LABEL_SCORES: list[tuple[re.Pattern, int]] = [
    (re.compile(r"итого\s*к\s*оплате", re.IGNORECASE), 100),
    (re.compile(r"итого\s*с\s*ндс", re.IGNORECASE), 90),
    (re.compile(r"итого\s*стоимость", re.IGNORECASE), 85),
    (re.compile(r"стоимость.{0,20}составляет", re.IGNORECASE), 80),
    (re.compile(r"составляет", re.IGNORECASE), 70),
    (re.compile(r"на\s+сумму", re.IGNORECASE), 65),
    (re.compile(r"итого\s*:", re.IGNORECASE), 60),
    (re.compile(r"\bитого\b", re.IGNORECASE), 55),
    # tolerant match for "Сумма:" that also catches OCR homoglyph corruption
    # such as "Cyммa:" (Latin C/y mixed with Cyrillic м/а).
    (re.compile(r"[сc][уy][мm][мm][аa]\s*[:о]", re.IGNORECASE), 50),
]
_AMOUNT_NEGATIVE_LABELS = [
    re.compile(r"без\s*ндс", re.IGNORECASE),
    re.compile(r"^\s*ндс\s*\d", re.IGNORECASE),
    re.compile(r"в\s*т\.?ч\.?\s*ндс", re.IGNORECASE),
    re.compile(r"в\s*том\s*числе\s*ндс", re.IGNORECASE),
    # unit price ("Цена за ед.") is not the document total - also written
    # tolerantly to catch the OCR variant "Цeнa".
    (re.compile(r"[цc][eе]н[аa]\s*за", re.IGNORECASE)),
]

_CONTRACTOR_REVERSED_RE = re.compile(
    r"([A-ZА-ЯЁ]{2,5}\s*«[^»]+»)\s*,\s*ИНН\s*[\dOolI]{10,12}\s*,[^\n]*?"
    r"именуем\w*[^\n]*?«(?:Поставщик|Исполнитель|Продавец)»",
    re.IGNORECASE,
)
_CONTRACTOR_LABEL_RE = re.compile(
    r"(?:Поставщик|Продавец|Исполнитель)\s*:\s*([^\n,]+?)(?=\s{2,}|,|\n|$)",
    re.IGNORECASE,
)

_SUBJECT_TABLE_ROW_RE = re.compile(r"\|\s*\d+\s*\|\s*([^|]+?)\s*\|")
_SUBJECT_LABEL_RE = re.compile(r"Предмет\s*:\s*(.+)", re.IGNORECASE)
_SUBJECT_CONTRACT_CLAUSE_RE = re.compile(
    r"1\.1\.[^.]*?(?:удобрени[а-я]*|товар)[^(]*\(([^)]+)\)", re.IGNORECASE | re.DOTALL
)
_SUBJECT_OSNOVANIE_RE = re.compile(r"Основание\s*:\s*(.+?)\s*№", re.IGNORECASE)


def _find_inn(text: str) -> Optional[str]:
    m = _INN_RE.search(text)
    if not m:
        return None
    digits = fix_ocr_digits(m.group(1))
    return digits if digits.isdigit() and len(digits) in (10, 12) else None


def _find_amount(text: str) -> Optional[float]:
    best_score = None
    best_value = None
    for m in _AMOUNT_CANDIDATE_RE.finditer(text):
        start = m.start()
        context = text[max(0, start - 45) : start]
        line_start = text.rfind("\n", 0, start) + 1
        full_context = text[line_start:start]

        if any(neg.search(full_context) for neg in _AMOUNT_NEGATIVE_LABELS):
            continue

        score = 0
        for pattern, weight in _AMOUNT_LABEL_SCORES:
            if pattern.search(context):
                score = max(score, weight)
        raw = fix_ocr_digits(m.group(1))
        value = parse_amount_number(raw)
        if value is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_value = value

    if best_value is not None:
        return best_value

    # No digit amount found anywhere - try a Russian number-in-words amount
    # (e.g. invoice_002.txt: "девятьсот тысяч рублей 00/100").
    words_match = re.search(r"составляет\s+([а-яА-ЯёЁ\s]+рубл\w*)", text)
    if words_match:
        value = parse_amount_words(words_match.group(1))
        if value is not None:
            return value
    return parse_amount_words(text)


def _find_contractor(text: str) -> Optional[str]:
    # The "именуемое в дальнейшем «Поставщик»" legal-boilerplate pattern
    # (used in contracts) is checked first because it *authoritatively*
    # defines who the counterparty is; a plain "Поставщик:" label appearing
    # later (e.g. in a signature block) is a secondary confirmation.
    m = _CONTRACTOR_REVERSED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _CONTRACTOR_LABEL_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(",")
    return None


def _find_subject(text: str) -> Optional[str]:
    m = _SUBJECT_TABLE_ROW_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _SUBJECT_LABEL_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    m = _SUBJECT_CONTRACT_CLAUSE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _SUBJECT_OSNOVANIE_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _extract_via_rules(text: str) -> dict[str, Any]:
    return {
        "amount": _find_amount(text),
        "date": find_first_date(text),
        "inn": _find_inn(text),
        "contractor": _find_contractor(text),
        "subject": _find_subject(text),
    }


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def extract(text: str) -> dict[str, Any]:
    """Extract ``amount``, ``date``, ``inn``, ``contractor``, ``subject``
    from a document's raw text. Fields that cannot be found are ``None``.

    Tries the LLM backend first (if configured), falls back to the
    dependency-free regex/rule backend otherwise or on any LLM error.
    """
    result = _extract_via_llm(text)
    if result is not None:
        return result
    return _extract_via_rules(text)

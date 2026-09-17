"""Document type classification: classify(text) -> (doc_type, confidence).

Types: "contract", "spec", "invoice", "act", "unknown".

Design note on the fallback classifier
---------------------------------------
A naive keyword search for words like "договор" or "спецификация" fails on
this dataset because every document *cross-references* the others by name
("Основание: Договор поставки № 47/2025 ...", "Спецификация № 1 ..."). If we
scored every mention of these words equally, an act or invoice would score
high on "contract" just because it quotes the contract number it was issued
under.

The fix used here: in these documents, a *document title* is written in
full capital letters at the very top ("ДОГОВОР ПОСТАВКИ № 47/2025",
"СПЕЦИФИКАЦИЯ № 1", "АКТ ВЫПОЛНЕННЫХ РАБОТ № 14", "УНИВЕРСАЛЬНЫЙ
ПЕРЕДАТОЧНЫЙ ДОКУМЕНТ"), while an inline cross-reference to another
document is written in normal sentence case ("Договор поставки № 47/2025
от ..."). So title-defining patterns are matched case-sensitively (or
restricted to the first ~100 characters) and only count when they look
like an actual heading, not a citation. Secondary, unambiguous contextual
phrases (bank requisites for invoices, "именуемое в дальнейшем" for
contracts, "Товар передан / Работы выполнены" for acts) are scored
case-insensitively anywhere in the text, since they never appear as mere
citations in other document types.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

DOC_TYPES = ("contract", "spec", "invoice", "act", "unknown")

# The minimum gap between the top score and the runner-up score required to
# commit to a classification. See RESULTS.md for the empirical tuning
# rationale behind this specific value.
CONFIDENCE_GAP_THRESHOLD = 15

# --------------------------------------------------------------------------
# LLM backend: Pydantic-схема + JsonOutputParser + PromptTemplate (см.
# extractor.py и RESULTS.md за ссылкой на источник паттерна). `Literal`
# здесь дополнительно ограничивает допустимые значения doc_type прямо на
# уровне схемы - JsonOutputParser зашивает это в инструкции модели.
# --------------------------------------------------------------------------
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.output_parsers import JsonOutputParser  # type: ignore
    from langchain_core.prompts import PromptTemplate  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
    from typing import Literal

    _LANGCHAIN_AVAILABLE = True

    class ClassificationResult(BaseModel):
        doc_type: Literal["contract", "spec", "invoice", "act", "unknown"] = Field(
            description="Тип финансового документа"
        )
        confidence: float = Field(
            description="Уверенность в классификации от 0.0 до 1.0", ge=0.0, le=1.0
        )

    _classify_parser = JsonOutputParser(pydantic_object=ClassificationResult)
    _classify_prompt = PromptTemplate(
        template=(
            "Определи тип финансового документа: contract (договор), "
            "spec (спецификация), invoice (счёт на оплату), act (акт/УПД), "
            "unknown (если непонятно).\n\n"
            "{format_instructions}\n\n"
            "Текст документа:\n---\n{text}\n---\n"
        ),
        input_variables=["text"],
        partial_variables={"format_instructions": _classify_parser.get_format_instructions()},
    )
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore
    _classify_parser = None  # type: ignore
    _classify_prompt = None  # type: ignore
    _LANGCHAIN_AVAILABLE = False


def _classify_via_llm(text: str) -> Optional[tuple[str, float]]:
    if not _LANGCHAIN_AVAILABLE:
        return None
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
        chain = _classify_prompt | llm | _classify_parser
        data = chain.invoke({"text": text})
        doc_type = data.get("doc_type")
        confidence = float(data.get("confidence", 0.0))
        if doc_type not in DOC_TYPES:
            return None
        return doc_type, confidence
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM classification failed, falling back to rules: %s", exc)
        return None


# --------------------------------------------------------------------------
# Rule-based fallback
# --------------------------------------------------------------------------
# Each signal: (compiled_pattern, weight, title_only)
# title_only=True  -> only counted if the match starts within the first
#                      TITLE_WINDOW characters of the (stripped) document.
# title_only=False -> counted anywhere in the document.
TITLE_WINDOW = 100

_SIGNALS: dict[str, list[tuple[re.Pattern, int, bool]]] = {
    "contract": [
        (re.compile(r"ДОГОВОР\s+ПОСТАВКИ\s*№"), 30, True),  # case-sensitive: real title is ALL CAPS
        (re.compile(r"именуем\w*\s+в\s+дальнейшем", re.IGNORECASE), 20, False),
        (re.compile(r"ПРЕДМЕТ\s+ДОГОВОРА"), 15, True),
        (re.compile(r"заключили\s+настоящий\s+договор", re.IGNORECASE), 10, False),
        (re.compile(r"\bдоговор\b", re.IGNORECASE), 1, False),  # weak: appears as citation everywhere
    ],
    "spec": [
        (re.compile(r"СПЕЦИФИКАЦИЯ\s*№", re.IGNORECASE), 30, True),
        (re.compile(r"Наименование\s+товара", re.IGNORECASE), 5, False),
        (re.compile(r"\bспецификаци\w*\b", re.IGNORECASE), 1, False),
    ],
    "invoice": [
        (re.compile(r"сч[её]т(?!\s*-\s*фактура)(?:\s+на\s+оплату)?\s*№", re.IGNORECASE), 30, True),
        (re.compile(r"Банковские\s+реквизиты", re.IGNORECASE), 15, False),
        (re.compile(r"\bр/с\s*\d", re.IGNORECASE), 10, False),
        (re.compile(r"\bБИК\b", re.IGNORECASE), 8, False),
    ],
    "act": [
        (re.compile(r"УНИВЕРСАЛЬНЫЙ\s+ПЕРЕДАТОЧНЫЙ\s+ДОКУМЕНТ|\bУПД\b"), 30, True),
        (re.compile(r"АКТ\s+ВЫПОЛНЕННЫХ\s+РАБОТ\s*№"), 30, True),
        (re.compile(r"Товар\s+передан|Работы\s+выполнены\s*:|Товар\s+получен|Работы\s+приняты", re.IGNORECASE), 15, False),
        (re.compile(r"Претензий\s+по\s+качеству", re.IGNORECASE), 10, False),
    ],
}


def _score(text: str) -> dict[str, int]:
    stripped = text.lstrip()
    scores = {k: 0 for k in _SIGNALS}
    for doc_type, patterns in _SIGNALS.items():
        total = 0
        for pattern, weight, title_only in patterns:
            m = pattern.search(text if not title_only else stripped)
            if not m:
                continue
            if title_only and m.start() > TITLE_WINDOW:
                continue
            total += weight
        scores[doc_type] = total
    return scores


def _classify_via_rules(text: str) -> tuple[str, float]:
    scores = _score(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_type, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score == 0:
        return "unknown", 0.5

    gap = top_score - second_score
    if gap < CONFIDENCE_GAP_THRESHOLD:
        # Ambiguous: two categories scored too close together to be sure.
        confidence = 0.5 + 0.02 * gap  # small residual confidence based on gap
        return "unknown", round(min(confidence, 0.6), 2)

    # Map the (unbounded) score gap onto a 0.5-0.97 confidence range.
    confidence = 0.5 + min(gap, 45) / 45 * 0.47
    return top_type, round(confidence, 2)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def classify(text: str) -> tuple[str, float]:
    """Classify a document's type. Returns (doc_type, confidence).

    doc_type is one of "contract", "spec", "invoice", "act", "unknown".
    Falls back to "unknown" whenever the top two candidate scores are too
    close to be confident (see CONFIDENCE_GAP_THRESHOLD), rather than
    guessing.
    """
    result = _classify_via_llm(text)
    if result is not None:
        return result
    return _classify_via_rules(text)

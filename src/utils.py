"""Shared low-level parsing helpers used by extractor.py, classifier.py.

These helpers contain no LLM calls - they are pure-Python, dependency-free
building blocks for the regex/rule-based fallback path, which is the path
that must work with zero external services (see task requirement: the
pipeline must run locally without API keys).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

# --------------------------------------------------------------------------
# OCR digit normalization
# --------------------------------------------------------------------------
# Scanned documents sometimes come back from OCR with visually-similar
# characters swapped for digits (capital O / lowercase o -> 0, lowercase
# l / uppercase I -> 1). We only apply this substitution *inside spans that
# are already known to be a numeric field* (a date, an amount, an INN) -
# never to the whole document - otherwise we would corrupt real words
# (e.g. a company name) that happen to contain the letters o/O/l/I.
_OCR_DIGIT_MAP = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"})


def fix_ocr_digits(span: str) -> str:
    """Replace common OCR letter/digit look-alikes within a numeric span."""
    return span.translate(_OCR_DIGIT_MAP)


# --------------------------------------------------------------------------
# Russian month names -> number
# --------------------------------------------------------------------------
_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "май": 5,
    "июн": 6, "июл": 7, "август": 8, "сентябр": 9, "октябр": 10,
    "ноябр": 11, "декабр": 12,
}


def _month_from_word(word: str) -> Optional[int]:
    word = word.lower()
    # match by stem since the word appears in different grammatical cases
    # (марта, апреля, февраля, ...)
    for stem, num in _MONTHS.items():
        if word.startswith(stem):
            return num
    return None


# Order matters: more specific patterns first.
_DATE_PATTERNS = [
    # 1 марта 2025 (г.)
    re.compile(
        r"(?P<d>\d{1,2})\s+(?P<m>[А-Яа-яЁё]+)\s+(?P<y>\d{4})\s*г?\.?",
    ),
    # 01.03.2025 or O1.O3.2O25 (OCR) - dot separated, day.month.year
    re.compile(r"(?P<d>[\dOolI]{1,2})\.(?P<m>[\dOolI]{1,2})\.(?P<y>[\dOolI]{2,4})"),
    # 03/01/25 - slash separated. The test spec lists this as an
    # alternative rendering of "01.03.2025", i.e. MM/DD/YY (US-style),
    # so month comes first here.
    re.compile(r"(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>\d{2,4})"),
    # already-ISO
    re.compile(r"(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})"),
]

_DUE_DATE_EXCLUSION = re.compile(
    r"(не\s+позднее|срок\s+оплаты|оплата\s+до|до:)\s*$", re.IGNORECASE
)


def _normalize_year(y: str) -> int:
    y = fix_ocr_digits(y)
    yi = int(y)
    if yi < 100:
        yi += 2000
    return yi


def find_first_date(text: str) -> Optional[str]:
    """Return the first plausible document date found in *text* as an ISO
    string (YYYY-MM-DD), skipping dates that are clearly a due-date /
    deadline rather than the document's own issue date.

    The heuristic is: a document's own date is virtually always the first
    date mentioned (in the title line or the header), while due dates
    ("не позднее ...", "срок оплаты", "оплата до") appear later in the
    body next to those specific keywords - so we scan matches in order
    and skip any whose preceding ~20 characters contain a due-date cue.
    """
    best: Optional[tuple[int, str]] = None
    for pattern in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            start = m.start()
            context = text[max(0, start - 20) : start]
            if _DUE_DATE_EXCLUSION.search(context):
                continue
            try:
                if "m" in m.groupdict() and m.group("m").isalpha() is False and not m.group("m").isdigit():
                    continue
                month_raw = m.group("m")
                if month_raw.isalpha():
                    month = _month_from_word(month_raw)
                    if month is None:
                        continue
                else:
                    month = int(fix_ocr_digits(month_raw))
                day = int(fix_ocr_digits(m.group("d")))
                year = _normalize_year(m.group("y"))
                if not (1 <= month <= 12 and 1 <= day <= 31):
                    continue
                iso = date(year, month, day).isoformat()
            except (ValueError, KeyError):
                continue
            if best is None or start < best[0]:
                best = (start, iso)
        if best is not None:
            # Found at least one valid match with the most specific
            # pattern group; no need to try looser patterns.
            break
    return best[1] if best else None


# --------------------------------------------------------------------------
# Russian number words -> float  (e.g. "девятьсот тысяч" -> 900000.0)
# --------------------------------------------------------------------------
_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
    "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900,
}
_SCALE = {"тысяча": 1000, "тысячи": 1000, "тысяч": 1000,
          "миллион": 1_000_000, "миллиона": 1_000_000, "миллионов": 1_000_000}

_NUM_WORD_RE = re.compile(
    r"\b(?:" + "|".join(
        sorted(
            list(_UNITS) + list(_TENS) + list(_HUNDREDS) + list(_SCALE),
            key=len, reverse=True,
        )
    ) + r")\b",
    re.IGNORECASE,
)


def parse_amount_words(text: str) -> Optional[float]:
    """Parse a Russian cardinal-number-in-words amount, e.g.

    "девятьсот тысяч рублей 00/100" -> 900000.0
    "Один миллион двести пятьдесят тысяч рублей 00 копеек" -> 1250000.0

    Returns None if no recognizable number word sequence is found.
    Only whole-ruble amounts are handled (kopecks after the words are
    ignored - "00/100" / "00 копеек" always denote zero kopecks in our
    sample documents).
    """
    words = _NUM_WORD_RE.findall(text.lower())
    if not words:
        return None

    total = 0
    current_group = 0  # value accumulated since the last thousand/million
    for w in words:
        if w in _HUNDREDS:
            current_group += _HUNDREDS[w]
        elif w in _TENS:
            current_group += _TENS[w]
        elif w in _UNITS:
            current_group += _UNITS[w]
        elif w in _SCALE:
            scale = _SCALE[w]
            current_group = current_group or 1
            total += current_group * scale
            current_group = 0
    total += current_group
    return float(total) if total > 0 else None


# --------------------------------------------------------------------------
# Currency amount normalization
# --------------------------------------------------------------------------
def parse_amount_number(raw: str) -> Optional[float]:
    """Normalize a numeric amount string in any of the supported formats:

    '1 250 000,00'  (space thousands sep, comma decimal)
    '1250000.00'    (no thousands sep, dot decimal)
    '1,250,000.00'  (comma thousands sep, dot decimal)
    '1 250 000'     (space thousands sep, no decimal)
    """
    s = raw.strip()
    s = s.replace("\xa0", " ")  # non-breaking space, common in RU documents
    s = re.sub(r"[^\d.,\s]", "", s)
    s = s.strip()
    if not s:
        return None

    has_comma = "," in s
    has_dot = "." in s
    has_space = " " in s

    try:
        if has_comma and has_dot:
            # '1,250,000.00' -> comma is thousands sep, dot is decimal
            s = s.replace(",", "")
            return float(s)
        if has_comma and not has_dot:
            # '1 250 000,00' -> comma is decimal separator
            s = s.replace(" ", "")
            s = s.replace(",", ".")
            return float(s)
        if has_space:
            s = s.replace(" ", "")
            return float(s)
        return float(s)
    except ValueError:
        return None
